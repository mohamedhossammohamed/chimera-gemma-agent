#!/usr/bin/env python3
"""Batch all 423 train_release cases through pipeline — rate-limited, mock by default."""
import os, sys, json, pathlib, argparse
sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
from src.agent.pipeline import ChimeraPipeline, PipelineConfig, load_trace_from_disk

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", default="/Users/mohammedhossam/Downloads/train_release")
    ap.add_argument("--out", default="iterations/iter-01/outputs")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    args=ap.parse_args()
    data_root=pathlib.Path(args.data)
    out_root=pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # collect traces
    traces=[]
    for task in ["task1","task2","task3"]:
        for case_dir in (data_root/task).iterdir():
            if not case_dir.is_dir(): continue
            traces.append(load_trace_from_disk(case_dir))
    if args.limit:
        traces=traces[:args.limit]
    print(f"Running {len(traces)} traces with {os.getenv('OPENROUTER_MODEL','google/gemma-3-27b-it:free')} mock={not os.getenv('OPENROUTER_API_KEY')}")
    cfg=PipelineConfig()
    pipe=ChimeraPipeline(cfg)
    outputs=pipe.process_batch(traces)
    # write per-case
    for o in outputs:
        (out_root/f"{o.case_id}.json").write_text(json.dumps(o.to_json(), indent=2))
    # summary
    summary={"n":len(outputs), "tasks":{}, "capra_s_high_recur":0}
    for o in outputs:
        summary["tasks"].setdefault(str(o.task),0)
        summary["tasks"][str(o.task)]+=1
    (out_root/"_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done → {out_root} n={len(outputs)}")

if __name__=="__main__":
    main()
