import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.fusion.gated import GatedFusion

def test_missing_mri():
    f=GatedFusion()
    # empty MRI
    emb={"MRI image":[], "Biopsy slide":[], "Prostatectomy slide":[]}
    fused=f.fuse(emb)
    assert len(fused)==16
    # should be zeros
    assert all(v==0 for v in fused)

def test_biopsy_zero_channel_task1():
    f=GatedFusion()
    # Task1 has MRI only
    emb=json.loads(pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_0020cfca66c8/prostate-modality-level-neural-representations.json").read_text())
    # biopsy is []
    fused=f.fuse(emb)
    assert len(fused)==16
    # Task2 with biopsy
    emb2=json.loads(pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task2/T2-001/prostate-modality-level-neural-representations.json").read_text())
    fused2=f.fuse(emb2)
    assert len(fused2)==16
    # norms should be similar despite missing channel (gated)
    import math
    n1=math.sqrt(sum(x*x for x in fused))
    n2=math.sqrt(sum(x*x for x in fused2))
    # after gating, both should be non-zero and Task1 not collapsed (gated handles 0% biopsy)
    assert n1>0.5 and n2>0.5
    assert abs(n1-n2) < 5  # not 2× difference

def test_prostatectomy_n3():
    f=GatedFusion()
    emb=json.loads(pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3/T3-001/prostate-modality-level-neural-representations.json").read_text())
    assert len(emb["Prostatectomy slide"])==3
    fused=f.fuse(emb)
    assert len(fused)==16

def test_site_norm():
    f=GatedFusion()
    emb=json.loads(pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task1/PT-pseudo_0020cfca66c8/prostate-modality-level-neural-representations.json").read_text())
    fused_a=f.fuse(emb, hospital="LUMC (Leiden)")
    fused_b=f.fuse(emb, hospital="Unknown Hospital")
    # site norm should make them slightly different
    assert fused_a!=fused_b
