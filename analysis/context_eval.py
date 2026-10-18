"""Does a context pack help the coder? Precision/recall with and without local vocabulary.

    python -m analysis.context_eval
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .annotate import ROOT, codes_of


def score(items: list[dict], context: str | None) -> dict:
    tp = fp = fn = 0
    rows = []
    for it in items:
        pred, gold = set(codes_of(it["text"], context)), set(it["gold"])
        tp, fp, fn = tp + len(pred & gold), fp + len(pred - gold), fn + len(gold - pred)
        rows.append({"text": it["text"], "gold": sorted(gold), "pred": sorted(pred)})
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "rows": rows}


def evaluate() -> dict:
    data = yaml.safe_load((ROOT / "data" / "context_eval.yaml").read_text(encoding="utf-8"))
    return {ctx: {"without": score(items, None), "with": score(items, ctx), "n": len(items)} for ctx, items in data.items()}


def main() -> None:
    res = evaluate()
    print(f"{'context':12} {'n':>3}   recall without -> with   precision without -> with")
    for ctx, r in res.items():
        print(f"{ctx:12} {r['n']:>3}   {r['without']['recall']:>8.0%} -> {r['with']['recall']:<6.0%}   "
              f"{r['without']['precision']:>8.0%} -> {r['with']['precision']:.0%}")
    out = ROOT / "figures" / "context_eval.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
