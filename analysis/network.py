"""Represent one respondent's considerations as a belief network and summarise it.

Nodes  = codebook categories the respondent raised (size = number of answers)
Edges  = categories raised together in the same answer, or within the same topic
Tension edges (dashed) link endorsed principles to an accepted violation.
"""
from __future__ import annotations

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from .annotate import CATS

# Reference categorical palette (dataviz skill), slots 1-2; text stays in ink colours.
COLOR = {"abstract": "#2a78d6", "tangible": "#eb6834", "violation": "#52514e"}
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"


def build_network(ann: dict) -> nx.Graph:
    G = nx.Graph()
    for t in ann["turns"]:
        for c in t["codes"]:
            if c not in G:
                G.add_node(c, kind=CATS[c]["type"], label=CATS[c]["label"], count=0)
            G.nodes[c]["count"] += 1
        for a, b in combinations(sorted(set(t["codes"])), 2):          # same answer: strong tie
            G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0})["weight"] + 1.0, tension=False)
    topics = {}
    for t in ann["turns"]:
        topics.setdefault(t["topic"], set()).update(t["codes"])
    for codes in topics.values():                                      # same topic: weak tie
        for a, b in combinations(sorted(codes), 2):
            if not G.has_edge(a, b):
                G.add_edge(a, b, weight=0.5, tension=False)
    for c in ann["contradictions"]:
        G.add_node(c["node"], kind="violation", label=c["node"].capitalize(), count=1)
        for p in c["principles"]:
            G.add_edge(p, c["node"], weight=1.0, tension=True)
    return G


def metrics(G: nx.Graph, ann: dict) -> dict:
    cat_nodes = [n for n, d in G.nodes(data=True) if d["kind"] != "violation"]
    H = G.subgraph(cat_nodes)
    mentions = sum(G.nodes[n]["count"] for n in cat_nodes)
    abstract = sum(G.nodes[n]["count"] for n in cat_nodes if G.nodes[n]["kind"] == "abstract")
    turns = ann["turns"]
    m = {
        "respondent": ann["meta"]["respondent_id"],
        "language": ann["meta"]["language"],
        "repertoire": len(cat_nodes),
        "abstract_share": abstract / mentions if mentions else 0.0,
        "density": nx.density(H) if len(cat_nodes) > 1 else 0.0,
        "justification_rate": sum(t["justified"] for t in turns) / len(turns) if turns else 0.0,
        "contradictions": len(ann["contradictions"]),
        "mean_words": sum(len(t["text"].split()) for t in turns) / len(turns) if turns else 0.0,
    }
    # Illustrative composite only - weights are arbitrary and NOT validated.
    m["sophistication_index"] = max(0.0, min(1.0,
        (m["repertoire"] / len(CATS) + m["justification_rate"] + m["density"]) / 3 - 0.15 * m["contradictions"]))
    return m


def fixed_layout(G: nx.Graph) -> dict:
    """Every category keeps the same position in every panel, so respondents can be
    compared at a glance: abstract principles on the left arc, tangible politics on
    the right arc, accepted violations in the centre."""
    import math
    abstract = [c for c, s in CATS.items() if s["type"] == "abstract"]
    tangible = [c for c, s in CATS.items() if s["type"] == "tangible"]
    pos = {}
    for group, (a0, a1) in ((abstract, (110, 250)), (tangible, (70, -70))):
        for i, c in enumerate(group):
            ang = math.radians(a0 + (a1 - a0) * i / max(len(group) - 1, 1))
            pos[c] = (math.cos(ang), math.sin(ang))
    viol = [n for n, d in G.nodes(data=True) if d["kind"] == "violation"]
    for i, n in enumerate(viol):
        pos[n] = (0.0, 0.25 - 0.55 * i)
    return {n: pos[n] for n in G.nodes}


def draw_network(G: nx.Graph, title: str = "", ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    fig = ax.figure
    ax.set_facecolor(SURFACE)
    ax.set_title(title, fontsize=10, color=INK, loc="left", fontweight="bold")
    ax.axis("off")
    if len(G) == 0:
        ax.text(0.5, 0.5, "no considerations coded", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return fig
    pos = fixed_layout(G)
    normal = [(a, b) for a, b, d in G.edges(data=True) if not d["tension"]]
    tension = [(a, b) for a, b, d in G.edges(data=True) if d["tension"]]
    nx.draw_networkx_edges(G, pos, edgelist=normal, ax=ax, edge_color="#b9b8b3",
                           width=[0.8 + 1.2 * G.edges[e]["weight"] for e in normal])
    nx.draw_networkx_edges(G, pos, edgelist=tension, ax=ax, edge_color=COLOR["violation"], width=2, style="dashed")
    for kind in ("abstract", "tangible", "violation"):
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == kind]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax, node_color=COLOR[kind],
                               node_shape="s" if kind == "violation" else "o",
                               node_size=[220 + 160 * G.nodes[n]["count"] for n in nodes],
                               edgecolors=SURFACE, linewidths=2)
    for n, (x, y) in pos.items():
        if G.nodes[n]["kind"] == "violation":
            ax.text(x, y - 0.13, G.nodes[n]["label"], ha="center", va="top", fontsize=7, color=INK)
        else:
            ax.text(x * 1.16, y * 1.16, G.nodes[n]["label"], fontsize=7, color=INK, va="center",
                    ha="right" if x < -0.05 else "left" if x > 0.05 else "center")
    ax.set_xlim(-1.9, 1.9)
    ax.set_ylim(-1.35, 1.25)
    ax.set_aspect("equal")
    return fig


def legend_handles():
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], marker="o", ls="", color=COLOR["abstract"], markersize=9, label="Abstract principle"),
        Line2D([], [], marker="o", ls="", color=COLOR["tangible"], markersize=9, label="Tangible / everyday politics"),
        Line2D([], [], marker="s", ls="", color=COLOR["violation"], markersize=8, label="Accepted norm violation"),
        Line2D([], [], color=COLOR["violation"], ls="--", lw=2, label="Tension with endorsed principle"),
    ]
