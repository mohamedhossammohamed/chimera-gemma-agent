import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.survival.pycox_head import PyCoxHead, c_index

def test_km_stats():
    head=PyCoxHead()
    head.fit([1.8,65.7,38.5,1.3,52.0],[1,0,0,1,1])
    assert head.km_stats["S12"] < 1.0
    assert head.km_stats["S60"] < head.km_stats["S12"]

def test_c_index():
    # perfect ranking
    times=[1,2,3,4]
    events=[1,1,0,0]
    risks=[0.9,0.7,0.3,0.2]  # higher risk earlier -> perfect
    assert c_index(times,events,risks) > 0.8
    # inverted
    risks2=[0.2,0.3,0.7,0.9]
    assert c_index(times,events,risks2) < 0.3

def test_predict_task3():
    from src.parsers.openmed_cpu import OpenMedParser
    p=pathlib.Path("/Users/mohammedhossam/Downloads/train_release/task3/T3-001")
    sp=json.loads((p/"structured-prompt.json").read_text())
    clin=json.loads((p/"prostate-time-to-recurrence-or-last-follow-up-clinical-data.json").read_text())
    emb=json.loads((p/"prostate-modality-level-neural-representations.json").read_text())
    from src.fusion.gated import GatedFusion
    parser=OpenMedParser(); parsed=parser.parse(sp,clin,emb)
    fusion=GatedFusion(); fused=fusion.fuse(emb)
    head=PyCoxHead(); head.fit([1.8,65.7],[1,0])
    pred=head.predict(parsed, fused, label={"months_to_recurrence":1.8,"event":1})
    assert "risk" in pred and "S12" in pred
    assert pred["capra_s"]==parsed.capra_s
