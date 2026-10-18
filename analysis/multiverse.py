"""Multiverse analysis over the decisions logged in config/decisions.yaml.

Every combination of each decision's main + plausible options is run; the
spread of results shows how much the conclusions depend on analytical choices.

    python -m analysis.multiverse                                  # synthetic population, offline
    python -m analysis.multiverse --input data/transcripts --llm   # include the LLM-annotator branch
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .annotate import DECISIONS, MAIN_SPEC, ROOT, annotate_transcript

INK, MUTED, RING, DOT, MAIN = "#1f1f1e", "#6b6a66", "#e6e5e1", "#8a93a1", "#0e6b5a"


def mean_answer_words(tr: dict) -> float:
    ans = [t["text"] for t in tr["turns"] if t["role"] == "respondent" and t.get("topic") not in (None, "closing")]
    return sum(len(a.split()) for a in ans) / max(len(ans), 1)


def specifications(include_llm: bool) -> list[dict]:
    axes = []
    for d in DECISIONS["decisions"]:
        opts = [d["main"], *d.get("plausible", [])]
        if d.get("run_offline") is False and not include_llm:
            opts = [d["main"]]
        axes.append([(d["id"], o) for o in opts])
    return [dict(combo) for combo in itertools.product(*axes)]


def run(transcripts: list[dict], spec: dict, llm=None) -> dict:
    sample = [t for t in transcripts if spec["sample"] == "keep_all" or mean_answer_words(t) >= 8]
    anns = [annotate_transcript(t, llm=llm if spec["annotator"] == "llm" else None, spec=spec) for t in sample]
    n = len(anns)
    action = sum(any(x["kind"] == "action" for x in a["tensions"]) for a in anns)
    gap = sum(any(x["kind"] == "dissatisfaction" for x in a["tensions"]) for a in anns)
    return {**spec, "n": n, "action_paradox": action / n if n else 0.0, "expectation_gap": gap / n if n else 0.0,
            "is_main": all(spec[k] == v for k, v in MAIN_SPEC.items())}


def spec_curve(df: pd.DataFrame, outcome: str, title: str, path: Path) -> None:
    """Specification curve: estimates sorted, with the choices behind each one underneath."""
    df = df.sort_values(outcome).reset_index(drop=True)
    decisions = [d for d in DECISIONS["decisions"] if df[d["id"]].nunique() > 1]
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 2.6 + 0.42 * sum(len(df[d["id"]].unique()) for d in decisions)), dpi=150,
                                 sharex=True, gridspec_kw={"height_ratios": [2.2, 0.42 * sum(len(df[d["id"]].unique()) for d in decisions)]})
    x = range(len(df))
    ax.scatter(x, df[outcome] * 100, s=22, color=[MAIN if m else DOT for m in df["is_main"]], zorder=3)
    main = df[df["is_main"]]
    if len(main):
        i, v = main.index[0], main[outcome].iloc[0] * 100
        ax.scatter([i], [v], s=70, facecolor="none", edgecolor=MAIN, linewidth=1.8, zorder=4)
        ax.annotate(f"main analysis: {v:.0f}%", (i, v), xytext=(8, 10), textcoords="offset points", fontsize=8.5, color=MAIN, fontweight="bold")
    lo, hi, med = df[outcome].min() * 100, df[outcome].max() * 100, df[outcome].median() * 100
    ax.axhline(med, color=RING, lw=1, zorder=1)
    ax.set_ylabel("% of respondents", fontsize=8.5, color=MUTED)
    ax.set_title(f"{title}: {lo:.0f}% to {hi:.0f}% across {len(df)} specifications (median {med:.0f}%)",
                 fontsize=10.5, color=INK, loc="left", fontweight="bold")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, colors=MUTED, length=0)
    ax.grid(axis="y", color=RING, lw=0.6)
    # choice matrix
    rows, labels = [], []
    for d in decisions:
        for o in df[d["id"]].unique():
            rows.append((d["id"], o))
            labels.append(f"{d['id'].replace('_', ' ')}: {o.replace('_', ' ')}")
    for r, (did, o) in enumerate(rows):
        on = df[did] == o
        bx.scatter([i for i in x if on[i]], [r] * int(on.sum()), s=14, marker="s",
                   color=[MAIN if df["is_main"][i] else INK for i in x if on[i]])
        bx.scatter([i for i in x if not on[i]], [r] * int((~on).sum()), s=14, marker="s", color=RING)
    bx.set_yticks(range(len(rows)), labels, fontsize=7.5, color=INK)
    bx.invert_yaxis()
    bx.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        bx.spines[s].set_visible(False)
    bx.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "synthetic_population"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--llm", action="store_true", help="also run the LLM-annotator branch")
    args = ap.parse_args()
    llm = None
    if args.llm:
        from interviewer.llm import LLMClient
        llm = LLMClient()
    transcripts = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(args.input).glob("*.json"))]
    if not transcripts:
        raise SystemExit(f"No transcripts in {args.input}. Run: python data/make_population.py")
    df = pd.DataFrame([run(transcripts, s, llm) for s in specifications(args.llm)])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "multiverse.csv", index=False)
    spec_curve(df, "action_paradox", "Action paradox", out / "multiverse_action_paradox.png")
    spec_curve(df, "expectation_gap", "Expectation gap", out / "multiverse_expectation_gap.png")
    m = df[df["is_main"]].iloc[0]
    print(f"{len(df)} specifications on {len(transcripts)} synthetic transcripts")
    for o in ("action_paradox", "expectation_gap"):
        print(f"  {o}: main {m[o]:.0%}, range {df[o].min():.0%}-{df[o].max():.0%}, median {df[o].median():.0%}")
    # which single decision moves the result most?
    for o in ("action_paradox", "expectation_gap"):
        swings = {d["id"]: df.groupby(d["id"])[o].mean().max() - df.groupby(d["id"])[o].mean().min()
                  for d in DECISIONS["decisions"] if df[d["id"]].nunique() > 1}
        top = max(swings, key=swings.get)
        print(f"  biggest lever for {o}: {top} ({swings[top]:.0%} difference in average)")


if __name__ == "__main__":
    main()
