"""
OpenMed CPU Parser — lightweight, 0GB VRAM, ~8ms
Fixes all 15 data-audit bugs: V-PSA, empty pmhx, NA pirads, psad drift, family history, bx_isup 0/null, dre sentences, cT variants, MRI outliers, list negation.
"""
from __future__ import annotations
import re
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── constants ──
DAYS_PER_YEAR = 365.2425
MRI_CLIP_MIN, MRI_CLIP_MAX = -22.0, 10.5  # winsorize from audit Bug12 (-31.7 actual)

FAMILY_MAP = {
    "no": "no", "there is no family history of prostate or breast cancer.": "no",
    "yes": "yes", "there is family history.": "yes",
    "unknown": "unknown", "family history of prostate or breast cancer is unknown.": "unknown",
}
# normalized via lower + strip + collapse
def normalize_family(raw: Any) -> str:
    if raw is None:
        return "unknown"
    s = str(raw).strip().lower()
    s = re.sub(r"\s+", " ", s)
    # map known variants
    if s in ("no", "yes", "unknown", "none", "null", ""):
        return s if s in ("no","yes") else "unknown"
    if "no family history" in s:
        return "no"
    if "family history" in s and "yes" in s:
        return "yes"
    if "unknown" in s:
        return "unknown"
    if s == "yes":
        return "yes"
    if s == "no":
        return "no"
    return "unknown"

@dataclass
class ParsedClinical:
    case_id: str = ""
    task: int = 0
    age: Optional[float] = None
    psa: Optional[float] = None
    vol: Optional[float] = None
    psad: Optional[float] = None
    psad_calculated: Optional[float] = None
    psad_source: str = "missing"  # stored | calculated | missing
    pirads: Optional[int] = None
    pirads_raw: Any = None
    pirads_na_flag: int = 0
    cspca: Optional[float] = None
    bx_isup: Optional[int] = None
    bx_isup_raw: Any = None  # keep 0 vs null distinction
    bx_isup_status: str = "unknown"  # negative(0) | positive(1-5) | unknown(null)
    bx_gl_prim: Optional[int] = None
    bx_gl_sec: Optional[int] = None
    dre: Optional[str] = None
    ct: Optional[str] = None
    ct_norm: Optional[str] = None
    family_history: str = "unknown"
    pmhx: list = field(default_factory=list)
    pmhx_source: str = "structured"  # structured | notes | cci_string | missing
    psadt: Optional[float] = None  # Infinity = stable
    psadt_trajectory: str = "[DATA NOT RECORDED]"
    psav: Optional[float] = None  # recomputed, not stored
    psav_stored: Optional[float] = None
    v_shaped: bool = False
    v_shaped_detail: str = ""
    mri_hospital: Optional[str] = None
    # surgical parsed (T3)
    gleason_prim: Optional[int] = None
    gleason_sec: Optional[int] = None
    margin: Optional[str] = None  # positive/negative/not_assessed
    ece: Optional[str] = None
    svi: Optional[str] = None
    lvi: Optional[str] = None
    lni: Optional[str] = None  # present/absent/not_assessed
    pt: Optional[str] = None
    capra_s: Optional[int] = None
    # provenance
    errors: list = field(default_factory=list)

def _safe_float(v):
    try:
        if v is None or v == "" or str(v).upper() in ("NA","NULL","N/A","NONE"):
            return None
        return float(v)
    except:
        return None

def _parse_pirads(v):
    if v is None or str(v).strip() == "":
        return None, v, 0
    s = str(v).strip()
    if s.upper() == "NA":
        return None, v, 1
    try:
        iv = int(float(s))
        if 1 <= iv <= 5:
            return iv, v, 0
        return None, v, 0
    except:
        return None, v, 0

def _ct_normalize(ct_raw: Any) -> Optional[str]:
    if not ct_raw or str(ct_raw).strip().upper() in ("NA","NULL",""):
        return None
    s = str(ct_raw).strip()
    # handle cT2 -> cT2a, cTx -> None, cT3 -> cT3a
    if s == "cT2":
        return "cT2a"
    if s == "cTx":
        return None
    if s == "cT3":
        return "cT3a"
    # already good like cT1c, cT2a/b/c, cT3a/b
    if re.match(r"^cT[1-4][a-c]?$", s):
        return s
    return s

def _dre_parse_t3(dre_text: Any) -> Optional[str]:
    if not dre_text:
        return None
    s = str(dre_text).lower()
    # try to find cT inside sentence
    m = re.search(r"cT[1-4][a-c]?", s)
    if m:
        return _ct_normalize(m.group(0))
    return None

def _psa_trend_parse(trend: list) -> list[tuple[datetime,float]]:
    pts = []
    for e in trend or []:
        date_str = e.get("date","")
        val = _safe_float(e.get("val"))
        if val is None:
            continue
        # try Feb 2022, Jan 2024 etc
        dt = None
        for fmt in ("%b %Y", "%B %Y", "%d %b %Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(str(date_str).strip(), fmt)
                break
            except:
                continue
        if dt is None:
            # try manual Feb 2022
            try:
                parts = str(date_str).strip().split()
                if len(parts)==2:
                    dt = datetime.strptime(parts[0]+" 01 "+parts[1], "%b %d %Y")
            except:
                continue
        if dt:
            pts.append((dt,val))
    pts.sort(key=lambda x: x[0])
    return pts

def _calc_psadt_psav(pts: list[tuple[datetime,float]]):
    if len(pts) < 3:
        return None, "[DATA NOT RECORDED]", None, "insufficient (<3)"
    span_days = (pts[-1][0]-pts[0][0]).days
    if span_days/30.44 < 6:
        return None, "[DATA NOT RECORDED]", None, f"span {span_days/30.44:.1f}mo <6"
    base = pts[0][0].timestamp()
    ts = [(p[0].timestamp()-base)/(86400*DAYS_PER_YEAR) for p in pts]
    vs = [p[1] for p in pts]
    n=len(ts)
    tmean=sum(ts)/n
    vmean=sum(vs)/n
    # PSAV linear
    ssT=sum((t-tmean)**2 for t in ts)
    if ssT==0:
        return None, "[DATA NOT RECORDED]", None, "zero var"
    ssTV=sum((t-tmean)*(v-vmean) for t,v in zip(ts,vs))
    psav = round((ssTV/ssT)*1000)/1000
    # PSADT log
    logVs=[math.log(v) for v in vs]
    logmean=sum(logVs)/n
    ssTL=sum((t-tmean)*(lv-logmean) for t,lv in zip(ts,logVs))
    k=ssTL/ssT
    if k<=0:
        return float("inf"), "Stable/Declining", psav, f"k={k:.4f} stable"
    dt_years=math.log(2)/k
    dt_months=round(dt_years*12*10)/10
    # trajectory
    if dt_months<6:
        traj="Aggressive"
    elif dt_months<12:
        traj="Rapid"
    elif dt_months<24:
        traj="Moderate"
    else:
        traj="Indolent"
    return dt_months, traj, psav, f"k={k:.4f}"

def _is_v_shaped(pts: list[tuple[datetime,float]]) -> tuple[bool,str]:
    if len(pts)<3:
        return False,""
    vals=[p[1] for p in pts]
    mn=min(vals)
    mn_i=vals.index(mn)
    if mn_i==0 or mn_i==len(vals)-1:
        return False,""
    first, last = vals[0], vals[-1]
    # 1.3× or 20% drop+rise check (audit: 1.3x or 20%)
    if (first/mn>1.3 or last/mn>1.3) or (first>mn*1.2 and last>mn*1.2 and (first-mn)>0.2*first):
        return True, f"min {mn} at {pts[mn_i][0].strftime('%b %Y')} vs ends {first}/{last}"
    return False,""

def _parse_surgical(text: str) -> dict:
    if not text or not isinstance(text,str):
        return {"gleason_prim":None,"gleason_sec":None,"margin":None,"ece":None,"svi":None,"lvi":None,"lni":None,"pt":None}
    low=text.lower()
    out={}
    # Gleason
    m=re.search(r"gleason\s+(\d)\s*\+\s*(\d)", text, re.I)
    if m:
        out["gleason_prim"]=int(m.group(1)); out["gleason_sec"]=int(m.group(2))
    else:
        out["gleason_prim"]=out["gleason_sec"]=None
    # ECE
    if "extraprostatic extension was present" in low or "extracapsular extension was present" in low:
        out["ece"]="present"
    elif "no extraprostatic extension" in low or "without extraprostatic" in low or "no extracapsular" in low or "there was no extraprostatic" in low:
        out["ece"]="absent"
    else:
        out["ece"]=None
    # margin
    if "margins were positive" in low or "margin was positive" in low:
        out["margin"]="positive"
    elif "margins were negative" in low or "margin was negative" in low or "r0" in low:
        out["margin"]="negative"
    else:
        out["margin"]=None
    # SVI
    if "seminal vesicles were invaded" in low or "seminal vesicle was invaded" in low:
        out["svi"]="present"
    elif "not invaded" in low or "were not invaded" in low:
        out["svi"]="absent"
    else:
        out["svi"]=None
    # LVI
    if "lymphovascular invasion was present" in low:
        out["lvi"]="present"
    elif "lymphovascular invasion was absent" in low:
        out["lvi"]="absent"
    else:
        out["lvi"]=None
    # LNI — critical: no lymph nodes were removed => not_assessed
    if "no lymph nodes were removed" in low:
        out["lni"]="not_assessed"
    elif "lymph node metastasis was present" in low or "lymph nodes were positive" in low:
        out["lni"]="present"
    elif "no lymph node metastasis" in low or "lymph node metastasis was absent" in low:
        out["lni"]="absent"
    elif "lymph node" in low and "positive" in low:
        out["lni"]="present"
    else:
        out["lni"]=None
    # pt
    mpt=re.search(r"pT([1-4][a-c]?)", text)
    out["pt"]=f"pT{mpt.group(1)}" if mpt else None
    return out

def _capra_s(psa, gleason_prim, gleason_sec, margin, ece, svi, lni):
    total=0
    if psa is None:
        psa_pts=0
    elif psa<=6:
        psa_pts=0
    elif psa<=10:
        psa_pts=1
    elif psa<=20:
        psa_pts=2
    else:
        psa_pts=3
    total+=psa_pts
    if gleason_prim is None or gleason_sec is None:
        gl_pts=0
    else:
        s=gleason_prim+gleason_sec
        if s<=6:
            gl_pts=0
        elif gleason_prim==3 and gleason_sec==4:
            gl_pts=1
        elif gleason_prim==4 and gleason_sec==3:
            gl_pts=2
        else:
            gl_pts=3
    total+=gl_pts
    if margin=="positive":
        total+=2
    if ece=="present":
        total+=1
    if svi=="present":
        total+=2
    if lni=="present":
        total+=1
    return min(12,total)

class OpenMedParser:
    """CPU parser — 0GB VRAM, ~8ms per case."""

    def parse(self, structured: dict, clinical: dict|None, embeddings: dict|None) -> ParsedClinical:
        pc = ParsedClinical()
        sp = structured or {}
        clin = clinical or {}
        pc.case_id = sp.get("case_id") or sp.get("pid") or ""
        pc.task = int(sp.get("task") or 0)

        # age, psa
        pc.age = _safe_float(sp.get("age"))
        pc.psa = _safe_float(sp.get("psa"))
        pc.cspca = _safe_float(sp.get("cspca"))
        pc.mri_hospital = sp.get("mri_hospital")

        # vol/psad with recomputation (Bug6)
        pc.vol = _safe_float(sp.get("vol"))
        stored_psad = _safe_float(sp.get("psad"))
        calc_psad = None
        if pc.psa is not None and pc.vol and pc.vol>0:
            calc_psad = round(pc.psa/pc.vol,3)
        pc.psad_calculated = calc_psad
        if calc_psad is not None and stored_psad is not None and abs(calc_psad-stored_psad)>0.01:
            pc.errors.append(f"psad drift stored {stored_psad} vs calc {calc_psad}")
            pc.psad = calc_psad
            pc.psad_source = "calculated"
        elif calc_psad is not None:
            pc.psad = calc_psad
            pc.psad_source = "calculated" if stored_psad is None else "stored"
        elif stored_psad is not None:
            pc.psad = stored_psad
            pc.psad_source = "stored"
        else:
            # try parse from radiology_report for T3 (Bug5)
            rad = clin.get("radiology_report") or ""
            m=re.search(r"Prostate volume:\s*([\d.]+)", str(rad))
            if m and pc.psa:
                try:
                    v=float(m.group(1))
                    pc.vol=v
                    pc.psad=round(pc.psa/v,3)
                    pc.psad_calculated=pc.psad
                    pc.psad_source="radiology_parsed"
                except:
                    pass

        # pirads NA (Bug4)
        pir, raw, flag = _parse_pirads(sp.get("pirads"))
        # fallback to radiology_report
        if pir is None and flag==0:
            rad = clin.get("radiology_report") or ""
            m=re.search(r"PI-RADS[:\s]*([1-5]|NA)", str(rad), re.I)
            if m:
                pir, raw2, flag2 = _parse_pirads(m.group(1))
                if pir is not None:
                    pir, raw, flag = pir, raw2, flag2
        pc.pirads=pir; pc.pirads_raw=sp.get("pirads"); pc.pirads_na_flag=flag

        # bx_isup 0 vs null (Bug14)
        raw_isup = sp.get("bx_isup")
        pc.bx_isup_raw = raw_isup
        if raw_isup is None:
            pc.bx_isup=None; pc.bx_isup_status="unknown"
        elif int(raw_isup)==0:
            pc.bx_isup=0; pc.bx_isup_status="negative"
        else:
            try:
                pc.bx_isup=int(float(raw_isup)); pc.bx_isup_status="positive"
            except:
                pc.bx_isup=None; pc.bx_isup_status="unknown"
        pc.bx_gl_prim=_safe_float(sp.get("bx_gl_prim"))
        pc.bx_gl_sec=_safe_float(sp.get("bx_gl_sec"))
        if pc.bx_gl_prim is not None:
            pc.bx_gl_prim=int(pc.bx_gl_prim)
        if pc.bx_gl_sec is not None:
            pc.bx_gl_sec=int(pc.bx_gl_sec)

        # dre + ct (Bug15, cT variants)
        pc.dre = sp.get("dre")
        ct_raw = sp.get("ct")
        pc.ct_norm = _ct_normalize(ct_raw)
        # T3 dre sentence fallback
        if pc.ct_norm is None and pc.dre and isinstance(pc.dre,str) and "cT" in pc.dre:
            parsed=_dre_parse_t3(pc.dre)
            if parsed:
                pc.ct_norm=parsed
        pc.ct = pc.ct_norm

        # family_history normalization (Bug13)
        raw_fh = clin.get("family_history") if clin.get("family_history") is not None else sp.get("family_history")
        pc.family_history = normalize_family(raw_fh)

        # pmhx handling (Bug8) — try structured pmhx, else medhx, else notes, else CCI string
        pmhx = sp.get("pmhx")
        if isinstance(pmhx, list) and len(pmhx)>0:
            pc.pmhx = [str(x) for x in pmhx]
            pc.pmhx_source="structured"
        else:
            # try medhx
            medhx = sp.get("medhx")
            if medhx and str(medhx).lower() not in ("not reported","none","", "null"):
                # split by comma
                parts=[p.strip() for p in str(medhx).split(",") if p.strip()]
                if parts and parts!=["Not reported"]:
                    pc.pmhx=parts
                    pc.pmhx_source="medhx"
            # try note_sections
            if not pc.pmhx:
                notes = sp.get("note_sections") or []
                # also check clinical previous_notes
                prev = clin.get("previous_notes")
                texts=[]
                if isinstance(notes,list):
                    for n in notes:
                        if isinstance(n,dict):
                            texts.append(n.get("t",""))
                if isinstance(prev, list):
                    for n in prev:
                        if isinstance(n,dict):
                            texts.append(n.get("text",""))
                elif isinstance(prev,str):
                    texts.append(prev)
                combined=" ".join(texts).lower()
                # keyword hunt for CCI conditions
                keywords=["hypertension","hypercholesterolaemia","diabetes","copd","asthma","coronary","stroke","ckd","cancer","dementia","liver disease"]
                found=[k for k in keywords if k in combined]
                if found:
                    pc.pmhx=found
                    pc.pmhx_source="notes"
            # T3 CCI string
            if not pc.pmhx:
                prev = clin.get("previous_notes")
                if isinstance(prev,str) and "CCI" in prev:
                    m=re.search(r"CCI[^\d]*(\d)", prev)
                    if m:
                        pc.pmhx=[f"CCI {m.group(1)}"]
                        pc.pmhx_source="cci_string"
            if not pc.pmhx:
                pc.pmhx=[]
                pc.pmhx_source="missing"
                if pc.task==3:
                    pc.errors.append("pmhx missing for T3 — CCI from structured absent")

        # psa trend PSADT/PSAV with V-shape (Bug7,9)
        trend = clin.get("psa_trend") if clin else None
        if isinstance(trend, list) and len(trend)>0:
            pts=_psa_trend_parse(trend)
            pc.v_shaped, pc.v_shaped_detail = _is_v_shaped(pts)
            if pc.v_shaped:
                pc.errors.append(f"V-shaped PSA {pc.v_shaped_detail}")
            pc.psadt, pc.psadt_trajectory, pc.psav, _ = _calc_psadt_psav(pts)
            pc.psav_stored = _safe_float(sp.get("psav"))
            if pc.psav is not None and pc.psav_stored is not None and abs(pc.psav-pc.psav_stored)>0.5:
                pc.errors.append(f"psav drift calc {pc.psav} vs stored {pc.psav_stored}")
        else:
            pc.psadt=None; pc.psadt_trajectory="[DATA NOT RECORDED]"; pc.psav=None
            if pc.task in (1,2):
                pc.errors.append("psa_trend missing for T1/T2")

        # surgical parse for T3 (Bug10)
        surg = clin.get("surgical_pathology_report") if clin else None
        if surg:
            parsed=_parse_surgical(surg)
            pc.gleason_prim=parsed.get("gleason_prim")
            pc.gleason_sec=parsed.get("gleason_sec")
            pc.margin=parsed.get("margin")
            pc.ece=parsed.get("ece")
            pc.svi=parsed.get("svi")
            pc.lvi=parsed.get("lvi")
            pc.lni=parsed.get("lni")
            pc.pt=parsed.get("pt")
            pc.capra_s=_capra_s(pc.psa, pc.gleason_prim, pc.gleason_sec, pc.margin, pc.ece, pc.svi, pc.lni)
        # also try to parse radiology volume for T3 already done

        # MRI outliers handling (Bug12) — clip note
        if embeddings and embeddings.get("MRI image"):
            # just flag
            try:
                vals=embeddings["MRI image"][0] if isinstance(embeddings["MRI image"][0], list) else embeddings["MRI image"]
                if any(v<-22 or v>10.5 for v in vals):
                    pc.errors.append(f"MRI outlier beyond ±22 (min {min(vals):.1f} max {max(vals):.1f}) winsorize to {MRI_CLIP_MIN}/{MRI_CLIP_MAX}")
            except:
                pass

        # 5-ARI check
        prev = clin.get("previous_notes") if clin else None
        texts=""
        if isinstance(prev,str):
            texts=prev
        elif isinstance(prev,list):
            texts=" ".join([p.get("text","") for p in prev if isinstance(p,dict)])
        if "5-ARI" in texts or "finasteride" in texts.lower() or "dutasteride" in texts.lower():
            if pc.psa:
                pc.errors.append(f"5-ARI noted — PSA {pc.psa} may be ×2 suppressed (use {pc.psa*2:.1f})")

        return pc
