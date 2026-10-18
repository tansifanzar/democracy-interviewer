"""Annotate a folder of transcripts, build belief networks, write figures and a table.

    python -m analysis.run_analysis                         # synthetic demo data, offline
    python -m analysis.run_analysis --input data/transcripts --llm
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .annotate import ROOT, annotate_transcript
from .layers_viz import gap_chart, three_layer_grid
from .network import INK, MUTED, build_network, draw_network, legend_handles, metrics

BAR = "#2a78d6"


def profile_chart(df: pd.DataFrame, path: Path) -> None:
    """Small multiples: one measure per panel, one bar per respondent (single series)."""
    panels = [("repertoire", "Distinct considerations", "{:.0f}"),
              ("abstract_share", "Share abstract (vs. tangible)", "{:.0%}"),
              ("justification_rate", "Answers with a reason", "{:.0%}"),
              ("contradictions", "Action paradoxes", "{:.0f}")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 0.55 * len(df) + 1.4), dpi=150, sharey=True)
    labels = list(df["respondent"])
    for ax, (col, title, fmt) in zip(axes, panels):
        ax.barh(labels, df[col], color=BAR, height=0.55)
        for y, v in enumerate(df[col]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.invert_yaxis() if ax is axes[0] else None
        for s in ("top", "right", "bottom"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#d4d3cf")
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", colors=INK, labelsize=8, length=0)
        ax.set_xlim(0, max(df[col].max() * 1.35, 1e-9) if col != "contradictions" else max(df[col].max() + 1, 2))
    fig.suptitle("Respondent profiles (synthetic demo data, dictionary baseline)", x=0.01, ha="left",
                 fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    ap.add_argument("--llm", action="store_true", help="annotate with the configured LLM instead of the dictionary")
    args = ap.parse_args()

    llm = None
    if args.llm:
        from interviewer.llm import LLMClient
        llm = LLMClient()

    files = sorted(Path(args.input).glob("*.json"))
    if not files:
        raise SystemExit(f"No transcripts found in {args.input}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    annotations, rows, graphs = [], [], []
    for f in files:
        tr = json.loads(f.read_text(encoding="utf-8"))
        ann = annotate_transcript(tr, llm=llm)
        G = build_network(ann)
        annotations.append(ann)
        graphs.append((ann, G))
        rows.append(metrics(G, ann))

    df = pd.DataFrame(rows)
    df.to_csv(out / "respondent_metrics.csv", index=False)
    (ROOT / "data" / "annotations.json").write_text(json.dumps(annotations, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = 3
    nrows = -(-len(graphs) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(5 * cols, 4.4 * nrows + 0.8), dpi=150)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (ann, G) in zip(axes, graphs):
        persona = ann["meta"].get("persona", "")
        draw_network(G, title=f"{ann['meta']['respondent_id']} · {persona}", ax=ax)
    for ax in axes[len(graphs):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Democratic belief networks from AI conversational interviews (synthetic demo data)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out / "belief_networks.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    profile_chart(df, out / "respondent_profiles.png")
    three_layer_grid(annotations, out / "three_layers.png")
    gap_chart(annotations, out / "expectation_gaps.png")
    print(df.round(2).to_string(index=False))
    print(f"\nWrote figures and metrics to {out}")


if __name__ == "__main__":
    main()
