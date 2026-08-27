import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.llm.gemma_client import GemmaClient, GemmaConfig
from src.parsers.openmed_cpu import OpenMedParser
import json

def test_mock_without_key():
    # ensure no key -> mock
    os.environ.pop("OPENROUTER_API_KEY", None)
    # also delete from config
    client=GemmaClient(GemmaConfig(api_key=None))
    # parse dummy
    parser=OpenMedParser()
    parsed=parser.parse({"case_id":"test","task":1,"age":66,"psa":7.8,"pirads":4,"bx_isup":2}, {"psa_trend":[{"date":"Jan 2022","val":2},{"date":"Jan 2023","val":3},{"date":"Jan 2024","val":4}]}, {"MRI image":[[0]*1024]})
    resp=client.call(parsed, [0]*16)
    assert resp.get("mock")==True
    assert "decision" in resp

def test_rate_limit_bucket():
    from src.llm.gemma_client import TokenBucket
    b=TokenBucket(rpm=18)
    # consume 18 should be ok
    for _ in range(18):
        assert b.consume(1) is True or b.consume(1)==True
    # 19th should wait
    w=b.consume(1)
    assert isinstance(w, float) and w>0

def test_no_hardcoded_key_in_repo():
    # Sentinel Shield: ensure no .py contains real key (skip this test file itself)
    import pathlib as pl
    root=pl.Path(__file__).parents[1]
    needle = "sk-" + "or-v1"
    real_fragment = "dc140aeb"
    for p in root.rglob("*.py"):
        if p.name == "test_gemma.py":
            continue
        txt=p.read_text()
        assert needle not in txt or real_fragment not in txt, f"hardcoded key in {p}"
    for p in root.rglob("*.md"):
        txt=p.read_text()
        if p.name==".env.example":
            assert "REPLACE" in txt
        else:
            assert real_fragment not in txt
