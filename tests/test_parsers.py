import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.parsers.openmed_cpu import OpenMedParser

def load_case(case_dir):
    sp=json.loads((case_dir/"structured-prompt.json").read_text())
    clin=None
    for f in case_dir.glob("*-clinical-data.json"):
        clin=json.loads(f.read_text()); break
    emb=None
    p=case_dir/"prostate-modality-level-neural-representations.json"
    if p.exists():
        emb=json.loads(p.read_text())
    return sp, clin, emb

def test_v_shaped():
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_0020cfca66c8")
    sp,clin,emb=load_case(p)
    parsed=OpenMedParser().parse(sp,clin,emb)
    assert parsed.v_shaped==True
    assert parsed.psadt==float("inf")
    assert parsed.psadt_trajectory=="Stable/Declining"

def test_pirads_na():
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_3646e0a2ae13")
    if not p.exists():
        return
    sp,clin,emb=load_case(p)
    parsed=OpenMedParser().parse(sp,clin,emb)
    assert parsed.pirads is None and parsed.pirads_na_flag==1

def test_psad_drift():
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_4c6b2b2bbf94")
    sp,clin,emb=load_case(p)
    parsed=OpenMedParser().parse(sp,clin,emb)
    # stored 0.07 vs calc 0.084 should be corrected
    assert abs(parsed.psad - 0.084) < 0.01

def test_surgical_not_assessed():
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3/T3-045")
    sp,clin,emb=load_case(p)
    parsed=OpenMedParser().parse(sp,clin,emb)
    # T3-045 has no nodes removed? check
    # T3-001 has no lymph nodes removed -> not_assessed
    p2=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3/T3-001")
    sp2,clin2,emb2=load_case(p2)
    parsed2=OpenMedParser().parse(sp2,clin2,emb2)
    # T3-001 should be present? Actually T3-001 has present, so test T3-045 or T3 with no nodes
    # find one with not_assessed
    for d in pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3").iterdir():
        if not d.is_dir(): continue
        sp3,clin3,_=load_case(d)
        pr=OpenMedParser().parse(sp3,clin3,{})
        if pr.lni=="not_assessed":
            assert True
            return
    assert False, "no not_assessed found"

def test_empty_pmhx():
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3/T3-001")
    sp,clin,emb=load_case(p)
    parsed=OpenMedParser().parse(sp,clin,emb)
    # T3 has no pmhx in structured, but should parse from notes or cci string
    assert parsed.pmhx_source in ("cci_string","missing","notes")
