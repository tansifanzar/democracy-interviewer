"""Figures for the three-layer view: concept -> expectation (vs. reality) -> behaviour.

three_layer_grid : one "fingerprint" per respondent. Each consideration keeps a
                   fixed spoke; the three rings are what democracy IS to them
                   (inner), what they EXPECT (middle) and what they DO (outer).
                   A consistent democrat lines up along the spokes; a dashed
                   spoke marks an action paradox (named in the concept, acted
                   against in a scenario).
gap_chart        : expected importance vs. perceived delivery per consideration
                   (the dissatisfaction paradox), averaged over respondents.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .annotate import CATS

INK, MUTED, FAINT, RING = "#1f1f1e", "#6b6a66", "#b9b8b3", "#e6e5e1"
TYPE = {"abstract": "#2a78d6", "tangible": "#eb6834"}
EXPECTED, DELIVERED = "#2a78d6", "#eb6834"
R = {"concept": 0.42, "expectation": 0.74, "behaviour": 1.06}


def _angle(c: str) -> float:
    abstract = [k for k, s in CATS.items() if s["type"] == "abstract"]
    tangible = [k for k, s in CATS.items() if s["type"] == "tangible"]
    group, (a0, a1) = (abstract, (112, 248)) if c in abstract else (tangible, (68, -68))
    i = group.index(c)
    return math.radians(a0 + (a1 - a0) * i / (len(group) - 1))


def _xy(c: str, r: float):
    a = _angle(c)
    return r * math.cos(a), r * math.sin(a)


def draw_three_layers(ann: dict, ax, title: str = "") -> None:
    L = ann["layers"]
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.45, 1.45)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", fontweight="bold")
    for name, r in R.items():
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=RING, lw=1.2))
        ax.text(0, r + 0.03, {"concept": "is", "expectation": "expects", "behaviour": "does"}[name],
                ha="center", va="bottom", fontsize=6.5, color=MUTED)
    action = {t["category"] for t in ann["tensions"] if t["kind"] == "action"}
    for c, spec in CATS.items():
        col = TYPE[spec["type"]]
        x0, y0 = _xy(c, 0.28)
        x1, y1 = _xy(c, 1.18)
        ax.plot([x0, x1], [y0, y1], color=RING, lw=0.8, zorder=0)
        present = any(c in L[k] for k in ("concept", "expectation", "acted"))
        lx, ly = _xy(c, 1.3)
        ax.text(lx, ly, spec["label"], fontsize=6.6, va="center", color=INK if present else FAINT,
                ha="right" if lx < -0.05 else "left" if lx > 0.05 else "center")
        if c in action:   # action paradox: dashed spoke from concept to behaviour
            xa, ya = _xy(c, R["concept"])
            xb, yb = _xy(c, R["behaviour"])
            ax.plot([xa, xb], [ya, yb], color=INK, lw=1.4, ls=(0, (3, 2)), zorder=1)
        if c in L["concept"]:
            x, y = _xy(c, R["concept"])
            ax.scatter([x], [y], s=[30, 60, 105][L["concept"][c] - 1], color=col, edgecolor="white", linewidth=1.2, zorder=3)
        if c in L["expectation"]:
            x, y = _xy(c, R["expectation"])
            v = L["reality"].get(c, 0)
            size = [40, 80][L["expectation"][c] - 1]
            if v > 0:
                ax.scatter([x], [y], s=size, color=col, edgecolor="white", linewidth=1.2, zorder=3)
            elif v < 0:
                ax.scatter([x], [y], s=size, facecolor="white", edgecolor=col, linewidth=1.8, zorder=3)
            else:
                ax.scatter([x], [y], s=size, color=FAINT, edgecolor="white", linewidth=1.2, zorder=3)
        if c in L["acted"]:
            x, y = _xy(c, R["behaviour"])
            if L["acted"][c] > 0:
                ax.scatter([x], [y], s=55, marker="s", color=col, edgecolor="white", linewidth=1.2, zorder=3)
            else:
                ax.scatter([x], [y], s=60, marker="X", color=INK, edgecolor="white", linewidth=0.8, zorder=3)


def legend_handles():
    return [
        Line2D([], [], marker="o", ls="", color=TYPE["abstract"], markersize=7, label="Principle"),
        Line2D([], [], marker="o", ls="", color=TYPE["tangible"], markersize=7, label="Everyday politics"),
        Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, markeredgewidth=1.6, markersize=7, label="Expected, seen badly delivered"),
        Line2D([], [], marker="s", ls="", color=MUTED, markersize=7, label="Acted in line"),
        Line2D([], [], marker="X", ls="", color=INK, markersize=7, label="Acted against"),
        Line2D([], [], color=INK, ls=(0, (3, 2)), lw=1.4, label="Action paradox"),
    ]


def three_layer_grid(annotations: list[dict], path) -> None:
    cols = 3
    rows = -(-len(annotations) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.0 * rows + 0.9), dpi=150)
    axes = axes.flatten()
    for ax, ann in zip(axes, annotations):
        draw_three_layers(ann, ax, f"{ann['meta']['respondent_id']} · {ann['meta'].get('persona', '')}")
    for ax in axes[len(annotations):]:
        ax.axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, frameon=False, fontsize=8.5)
    fig.suptitle("What democracy is, what people expect, what they would do (synthetic demo data)",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def gap_chart(annotations: list[dict], path) -> None:
    rows = []
    for c, spec in CATS.items():
        exp = [a["layers"]["expectation"][c] for a in annotations if c in a["layers"]["expectation"]]
        judged = [a["layers"]["reality"][c] for a in annotations if c in a["layers"]["expectation"] and c in a["layers"]["reality"]]
        if not exp or not judged:
            continue
        importance = sum(e / 2 for e in exp) / len(exp)
        delivery = sum((v + 1) / 2 for v in judged) / len(judged)
        rows.append((spec["label"], importance, delivery, len(judged)))
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.4), dpi=150)
    for y, (label, imp, dlv, n) in enumerate(rows):
        ax.plot([dlv, imp], [y, y], color=RING, lw=4, solid_capstyle="round", zorder=1)
        # expected = ring, delivered = dot, so both stay visible when they coincide
        ax.scatter([imp], [y], s=150, facecolor="white", edgecolor=EXPECTED, linewidth=2.2, zorder=3)
        ax.scatter([dlv], [y], s=55, color=DELIVERED, edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(1.04, y, f"n = {n}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.12)
    ax.set_xticks([0, 0.5, 1], ["low / badly", "medium / mixed", "high / well"], fontsize=8, color=MUTED)
    ax.grid(axis="x", color=RING, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RING)
    ax.tick_params(length=0)
    if rows:
        ax.text(rows[0][1], -0.55, "expected", color=EXPECTED, fontsize=8.5, ha="center", fontweight="bold")
        ax.text(rows[0][2], -0.55, "delivered", color=DELIVERED, fontsize=8.5, ha="center", fontweight="bold")
    ax.set_title("Where expectations and perceived delivery part ways (synthetic demo data)", fontsize=10.5,
                 color=INK, loc="left", fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
