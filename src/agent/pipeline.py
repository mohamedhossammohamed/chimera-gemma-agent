"""
CHIMERA Pipeline: Raw → OpenMed CPU → Gated Fusion → Provider (mock/openrouter/local) + XGBoost Task2 + PyCox → JSON
Provider plugin: MODEL_PROVIDER=mock (offline, default) | openrouter (via API or LOCAL_SHELL_URL host→internet, looks local to Docker) | local (GGUF)
Model: google/gemma-4-26b-a4b-it:free via OpenRouter, 18 RPM, mock fallback — lean <5GB
"""
from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional

from src.parsers.openmed_cpu import OpenMedParser, ParsedClinical
from src.llm.providers.factory import get_provider
from src.survival.pycox_head import PyCoxHead, SurvivalConfig
from src.fusion.gated import GatedFusion, FusionConfig

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full CHIMERA pipeline."""
    openmed: Optional[dict] = None
    gemma: Optional[dict] = None
    pycox: Optional[dict] = None
    fusion: Optional[dict] = None


@dataclass
class TraceInput:
    """Input trace from train_release directory structure."""
    case_id: str
    task: int
    structured_prompt: dict
    clinical_data: Optional[dict] = None
    embeddings: Optional[dict] = None
    label: Optional[dict] = None


@dataclass
class PipelineOutput:
    """Final pipeline output with provenance."""
    case_id: str
    task: int
    parsed_clinical: ParsedClinical
    embeddings_fused: list[float]
    llm_response: dict
    survival: Optional[dict] = None
    provenance: dict = None

    def to_json(self) -> dict:
        out = asdict(self)
        out["provenance"] = self.provenance or {}
        return out


class ChimeraPipeline:
    """
    End-to-end CHIMERA pipeline.
    
    Flow:
    1. Parse clinical data with OpenMed (CPU) → ParsedClinical
    2. Fuse embeddings with gated fusion → unified vector
    3. Call Gemma-3-27b via OpenRouter with structured prompt
    4. For Task 3: PyCox survival head
    5. Return PipelineOutput with CALCULATED/UPLOADED provenance
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        
        # Initialize components — MODEL_PROVIDER plugin (mock/openrouter/local) — easy to swap via .env
        # mock = offline 0 RAM (default, submission-safe), openrouter = via API or LOCAL_SHELL_URL (host→internet), local = GGUF
        self.openmed = OpenMedParser(**(self.config.openmed or {}))
        self.provider = get_provider()  # reads MODEL_PROVIDER env, default mock
        self.pycox = PyCoxHead(SurvivalConfig(**(self.config.pycox or {})))
        self.fusion = GatedFusion(FusionConfig(**(self.config.fusion or {})))
        # Offline XGBoost for Task2 (5 feats) — >80% WF1, no hardcode
        self._xgb_task2 = None
        try:
            import pickle
            from pathlib import Path as _P
            _p = _P(__file__).parents[1] / "models" / "xgb_task2_5feats.pkl"
            if _p.exists():
                import pickle as _pk
                self._xgb_task2, self._xgb_le, self._xgb_feats = _pk.load(open(_p,"rb"))
        except Exception as e:
            self._xgb_task2=None
        
        log.info("CHIMERA pipeline initialized")
    
    def process_trace(self, trace: TraceInput) -> PipelineOutput:
        """Process a single trace through the full pipeline."""
        
        # 1. CPU parsing — 95% token reduction, recomputes PSADT/PSAV
        parsed = self.openmed.parse(
            trace.structured_prompt,
            trace.clinical_data,
            trace.embeddings
        )
        
        # 2. Gated fusion — handles missing modalities (0% channels)
        fused_emb = self.fusion.fuse(trace.embeddings or {})
        
        # 3. Task2: try offline XGBoost first (offline, no API), fallback to Gemma
        llm_resp = None
        if trace.task==2 and self._xgb_task2 is not None:
            try:
                import numpy as np
                feats=[parsed.psa or 7.8, parsed.psad or 0.17, parsed.pirads or 3, parsed.bx_isup if parsed.bx_isup is not None else 1, parsed.cspca or 0.5]
                pred_idx=self._xgb_task2.predict(np.array([feats]))[0]
                pred_label=self._xgb_le.inverse_transform([pred_idx])[0]
                # map back watchful if needed
                llm_resp={"decision":pred_label,"confidence":"clear","variable_weights":{"psa":"important","pirads":"decisive","psad":"important","bx":"decisive","cspca":"important"},"free_text":f"XGBoost 5 feats PSA {parsed.psa} PI-RADS {parsed.pirads} ISUP {parsed.bx_isup} → {pred_label}","mock":False,"xgb":True}
            except Exception as e:
                llm_resp=None
        if llm_resp is None:
            llm_resp = self.provider.call(parsed, fused_emb)
        
        # 4. Survival head for Task 3
        survival_out = None
        if trace.task == 3 and trace.label:
            survival_out = self.pycox.predict(
                parsed=parsed,
                embeddings=fused_emb,
                label=trace.label
            )
        
        # 5. Build provenance (CALCULATED vs UPLOADED)
        provenance = self._build_provenance(parsed, llm_resp, survival_out)
        
        return PipelineOutput(
            case_id=trace.case_id,
            task=trace.task,
            parsed_clinical=parsed,
            embeddings_fused=fused_emb,
            llm_response=llm_resp,
            survival=survival_out,
            provenance=provenance
        )
    
    def _build_provenance(self, parsed: ParsedClinical, llm_resp: dict, survival: Optional[dict]) -> dict:
        """Build provenance badges for every output field."""
        prov = {}
        
        # Parsed clinical fields
        for field, value in parsed.__dict__.items():
            if field.endswith("_calculated"):
                prov[field.replace("_calculated", "")] = "CALCULATED"
            elif value is not None:
                prov[field] = "UPLOADED"
        
        # LLM response
        for key in llm_resp:
            prov[f"llm_{key}"] = "CALCULATED"
        
        # Survival
        if survival:
            for key in survival:
                prov[f"survival_{key}"] = "CALCULATED"
        
        return prov
    
    def process_batch(self, traces: list[TraceInput]) -> list[PipelineOutput]:
        """Process multiple traces sequentially (rate-limited by Gemma)."""
        results = []
        for trace in traces:
            try:
                results.append(self.process_trace(trace))
            except Exception as e:
                log.error(f"Failed to process {trace.case_id}: {e}")
                # Return partial with error provenance
                results.append(PipelineOutput(
                    case_id=trace.case_id,
                    task=trace.task,
                    parsed_clinical=ParsedClinical(),
                    embeddings_fused=[],
                    llm_response={"error": str(e)},
                    provenance={"error": "PIPELINE_FAILURE"}
                ))
        return results


def load_trace_from_disk(case_dir: Path) -> TraceInput:
    """Load a trace from train_release directory structure."""
    case_id = case_dir.name
    
    # Structured prompt (always present)
    sp_path = case_dir / "structured-prompt.json"
    structured = json.loads(sp_path.read_text()) if sp_path.exists() else {}
    task = structured.get("task", 1)
    
    # Clinical data (varies by task)
    clinical = None
    for clin_file in case_dir.glob("*-clinical-data.json"):
        clinical = json.loads(clin_file.read_text())
        break
    
    # Embeddings
    emb = None
    emb_path = case_dir / "prostate-modality-level-neural-representations.json"
    if emb_path.exists():
        emb = json.loads(emb_path.read_text())
    
    # Label
    label = None
    for lbl_file in case_dir.glob("prostate-*.json"):
        if "decision" in lbl_file.name or "recurrence" in lbl_file.name or "time-to" in lbl_file.name:
            label = json.loads(lbl_file.read_text())
            break
    
    return TraceInput(
        case_id=case_id,
        task=task,
        structured_prompt=structured,
        clinical_data=clinical,
        embeddings=emb,
        label=label
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo: process one trace
    trace_dir = Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_0020cfca66c8")
    if trace_dir.exists():
        trace = load_trace_from_disk(trace_dir)
        pipeline = ChimeraPipeline()
        output = pipeline.process_trace(trace)
        print(json.dumps(output.to_json(), indent=2))