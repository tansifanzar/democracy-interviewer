"""How did the agents behave across interviews? Prints a summary and draws figures/agent_activity.png.

    python data/simulate_interviews.py && python -m analysis.agent_report
    python -m analysis.agent_report --input data/transcripts        # your own interviews
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .agent_trace import TOPICS, summarise
from .annotate import ROOT

INK, MUTED, RING, ASKED, PRODUCTIVE = "#1f1f1e", "#6b6a66", "#e6e5e1", "#b9c2cf", "#0e6b5a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data" / "simulated"))
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args()
    trs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(Path(args.input).glob("*.json"))]
    if not trs:
        raise SystemExit(f"No transcripts in {args.input}. Run: python data/simulate_interviews.py")
    s = summarise(trs)
    rows = [(t, v) for t, v in s["per_topic"].items() if v["answers"]]
    print(f"{s['n']} interviews · safety flags: {s['safety_flags']}")
    print(f"{'topic':18} follow-up rate   productive follow-ups")
    for t, v in rows:
        n_main = s["n"]
        rate = v["followups"] / n_main
        prod = v["productive"] / v["followups"] if v["followups"] else 0
        print(f"{t:18} {rate:>8.0%}        {prod:>8.0%}")
    b = s["bridge"]
    print(f"bridge: {b['bridge_arm']} in bridge arm · {b['conflict']} with a principle-choice conflict · "
          f"{b['asked']} asked · reactions {b['reactions']}")

    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 1.6), dpi=150)
    labels = [TOPICS[t]["id"].replace("_", " ") for t, _ in rows]
    asked = [v["followups"] / s["n"] * 100 for _, v in rows]
    prod = [v["productive"] / s["n"] * 100 for _, v in rows]
    y = range(len(rows))
    ax.barh(y, asked, color=ASKED, height=0.6, label="follow-up asked")
    ax.barh(y, prod, color=PRODUCTIVE, height=0.6, label="follow-up added something")
    for i, (a, p) in enumerate(zip(asked, prod)):
        if a:
            ax.text(a + 1, i, f"{p / a:.0%} productive", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(list(y), labels, fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("follow-ups per 100 interviews", fontsize=8.5, color=MUTED)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RING)
    ax.tick_params(length=0, labelsize=8, colors=MUTED)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Where the probe agent follows up, and where it pays off (simulated interviews)",
                 fontsize=10.5, color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(args.out) / "agent_activity.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
