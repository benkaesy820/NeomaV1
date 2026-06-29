from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows=[]
    with path.open("r",encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): rows.append(json.loads(line))
    return rows


def extract_choice(text: str) -> str | None:
    cleaned = text.strip().upper()
    match = re.search(r"(?<![A-Z0-9])([ABCD])(?![A-Z0-9])", cleaned)
    return match.group(1) if match else None


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--suite",type=Path,required=True)
    parser.add_argument("--outputs",type=Path,required=True,help="JSONL with id plus answer or output")
    parser.add_argument("--out",type=Path,default=Path("runs/stage_a_english_score.json"))
    args=parser.parse_args()
    suite={row["id"]:row for row in load_jsonl(args.suite)}
    outputs=load_jsonl(args.outputs)
    results=[]
    for row in outputs:
        rid=row.get("id")
        if rid not in suite: continue
        answer=str(row.get("answer") or row.get("output") or "")
        expected=suite[rid]["accepted_answers"]
        if suite[rid]["scoring"]=="choice":
            normalized=extract_choice(answer)
            ok=normalized in expected
        else:
            normalized=" ".join(answer.lower().split())
            ok=normalized in {" ".join(value.lower().split()) for value in expected}
        results.append({"id":rid,"correct":ok,"normalized_answer":normalized,"category":suite[rid]["category"],"language":suite[rid]["language"]})
    by_category={}
    for category in sorted({r["category"] for r in results}):
        subset=[r for r in results if r["category"]==category]
        by_category[category]={"correct":sum(r["correct"] for r in subset),"total":len(subset),"accuracy":sum(r["correct"] for r in subset)/len(subset) if subset else 0.0}
    report={"suite":str(args.suite),"scored":len(results),"expected":len(suite),"correct":sum(r["correct"] for r in results),"accuracy":sum(r["correct"] for r in results)/len(results) if results else 0.0,"by_category":by_category,"results":results}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8",newline="\n")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},indent=2))

if __name__=="__main__": main()
