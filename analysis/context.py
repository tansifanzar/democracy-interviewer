"""Context packs: country-specific wording, vocabulary and conduct for the agents and the coder.

Each pack (config/contexts/<id>.yaml) keeps the construct a question measures
fixed and adapts only what should differ between societies:

    vignettes               locally meaningful wording of the behaviour scenarios
    glossary                local terms mapped to the shared codebook (for the coder)
    sensitive_topics        topics where a refusal is respected without probing
    discomfort_markers      phrases that signal a refusal or perceived risk
    reassurance             what the interviewer says instead of a follow-up
    notes_for_agents        guidance added to the LLM agents' prompts
    probing_style           how to probe in this context
    interpretation_guardrails  warnings for analysis and write-up

Every country gets a pack, Western ones included: no context is the default.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

CONTEXT_DIR = Path(__file__).resolve().parent.parent / "config" / "contexts"


@lru_cache(maxsize=None)
def load_context(context_id: str | None) -> dict | None:
    if not context_id:
        return None
    path = CONTEXT_DIR / f"{context_id}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown context '{context_id}'. Available: {available_contexts()}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def available_contexts() -> list[str]:
    return sorted(p.stem for p in CONTEXT_DIR.glob("*.yaml"))


def signals_discomfort(text: str, ctx: dict | None) -> bool:
    if not ctx:
        return False
    t = text.lower()
    return any(m in t for m in ctx.get("discomfort_markers", []))
