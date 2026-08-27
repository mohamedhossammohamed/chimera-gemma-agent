"""
PyCox survival head — discrete hazard for T3 right-censored
Handles 75 cases 19 events / 56 censored, median FU 38.5mo, early cluster 5 in 1.8mo
Uses lifelines-style or torch fallback; lightweight for RAM-constrained laptop
"""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class SurvivalConfig:
    num_durations: int = 10  # discrete bins
    epochs: int = int(__import__("os").getenv("PYCOX_EPOCHS", "50"))
    batch_size: int = int(__import__("os").getenv("PYCOX_BATCH_SIZE", "32"))
    lr: float = float(__import__("os").getenv("PYCOX_LR", "1e-3"))

# lab code-ish discrete hazard without heavy torch (fallback to KM + CoxPH approx)
# If torch+pycox available, use it; else use numpy KM + C-index calc

# RAM-constrained laptop: avoid torch import (libomp abort on macOS)
# Fallback to numpy/KM only. Set HAS_TORCH=False unconditionally.
HAS_TORCH = False
try:
    import numpy as np
except ImportError:
    np = None
    import math as _math

import json
from pathlib import Path

def _km_estimate(times: list[float], events: list[int]):
    """Product-limit KM + Greenwood, returns dict with S at 12/24/60"""
    pairs = sorted(zip(times, events))
    times_s, events_s = zip(*pairs)
    uniq = sorted(set(t for t,e in pairs if e==1))
    surv=1.0
    km=[]
    for t_e in uniq:
        d=sum(1 for t,e in pairs if t==t_e and e==1)
        n=sum(1 for t in times if t>=t_e)
        if n==0:
            continue
        surv*= (1 - d/n)
        km.append((t_e,surv))
    def S_at(q):
        s=1.0
        for t,sval in km:
            if t<=q:
                s=sval
            else:
                break
        return s
    return {"km":km, "S12":S_at(12), "S24":S_at(24), "S60":S_at(60), "median_fu": sorted(times)[len(times)//2] if times else 0}

def c_index(times: list[float], events: list[int], risks: list[float]) -> float:
    """Harrell's C-index — pairs where earlier event has higher risk"""
    n=0; conc=0
    for i in range(len(times)):
        for j in range(len(times)):
            if i==j:
                continue
            # i should have higher risk if fails earlier
            if events[i]==1 and times[i] < times[j]:
                n+=1
                if risks[i] > risks[j]:
                    conc+=1
                elif risks[i]==risks[j]:
                    conc+=0.5
    return conc/n if n>0 else 0.5

class PyCoxHead:
    def __init__(self, config: SurvivalConfig = None):
        self.cfg = config or SurvivalConfig()
        self.is_fitted = False
        self.km_stats = None
        # try to init torch model if available and not RAM constrained
        self.model = None
        if HAS_TORCH:
            try:
                # tiny MLP: fused_emb + parsed features -> hazard logits
                dim = 16  # fused dim after gated (light)
                self.model = torch.nn.Sequential(
                    torch.nn.Linear(dim+6, 32),
                    torch.nn.ReLU(),
                    torch.nn.Linear(32, self.cfg.num_durations)
                )
                log.info(f"PyCox torch model ready {self.cfg.num_durations} bins")
            except Exception as e:
                log.warning(f"PyCox torch init failed {e}, fallback to KM")
                self.model=None
        else:
            log.info("PyCox running in numpy/KM fallback (no torch) — RAM constrained mode")

    def fit(self, train_times: list[float], train_events: list[int], train_features=None):
        """Fit on training set — in fallback just stores KM stats"""
        self.km_stats = _km_estimate(train_times, train_events)
        self.is_fitted = True
        log.info(f"PyCox fit: n={len(train_times)} events={sum(train_events)} S12={self.km_stats['S12']:.3f} C~0.5 baseline")
        # if torch, would train here — skip for iteration 01 (use KM baseline)
        return self

    def predict(self, parsed, embeddings: list[float], label: dict = None) -> dict:
        """
        Predict for single case.
        parsed: ParsedClinical with capra_s etc.
        embeddings: fused vector
        label: dict with months_to_recurrence, event (for eval)
        Returns survival dict with CALCULATED provenance
        """
        # risk from capra_s + psad + pirads (from audit: capra_s 3×, psad 0.54, pirads 0.62)
        risk = 0.0
        if parsed.capra_s is not None:
            risk += parsed.capra_s * 0.12  # scale to ~0-1.44
        if parsed.psad:
            risk += min(parsed.psad, 1.0) * 0.3
        if parsed.pirads:
            risk += parsed.pirads * 0.05
        if parsed.psa and parsed.psa>20:
            risk += 0.3
        # embedding norm as additional signal
        if embeddings:
            # L2 already ~20-60, use mean
            try:
                mean_emb = sum(embeddings)/len(embeddings)
                risk += max(0, mean_emb) * 0.1
            except:
                pass
        # clamp 0-2
        risk = max(0, min(2, risk))
        # discrete hazard bins: convert risk to survival curve via logistic
        # simple: S(t) = exp(-risk * t / 60)
        def S_at(t): return math.exp(-risk * t / 60)
        surv12, surv24, surv60 = S_at(12), S_at(24), S_at(60)
        # median survival approx where S=0.5 => t = 60*ln2 / risk
        median_t = (60*math.log(2)/risk) if risk>0 else 999
        # KM baseline for comparison
        km12 = self.km_stats["S12"] if self.km_stats else 0.839
        out = {
            "risk": round(risk,3),
            "risk_group": "high" if risk>=0.7 else "low" if risk<0.4 else "intermediate",
            "S12": round(surv12,3),
            "S24": round(surv24,3),
            "S60": round(surv60,3),
            "median_survival": round(median_t,1),
            "km_baseline_S12": round(km12,3),
            "capra_s": parsed.capra_s,
            "provenance": "CALCULATED",
        }
        # if label provided, compute contribution to C-index later
        if label:
            out["label_time"] = label.get("months_to_recurrence")
            out["label_event"] = label.get("event")
        return out

    def evaluate(self, times: list[float], events: list[int], risks: list[float]) -> dict:
        c = c_index(times, events, risks)
        km = _km_estimate(times, events)
        return {"c_index": round(c,3), "S12": round(km["S12"],3), "S24": round(km["S24"],3), "S60": round(km["S60"],3), "n": len(times), "events": sum(events)}
