"""Effect specs: how a character's skills look, phase by phase, checked before they reach the VM.

A spec is the skill tree of one character in a fixed shape (``FIELDS``). It is written once per
character (by hand or by a language model, see ``scripts/effect_specs.py``), validated here, then
merged into the roster: phase texts feed the fine narration styles, ``pose`` drives the proxy
body and ``kind`` says whether the proxy shows a flying projectile.

What the checks enforce comes from hand-written effect prompts that worked on H3 and from our own
renders: effects are written as light with a colour and a shape, the caster's whole body takes part
in the cast, the hit is short and its light clears off the target at once, and nothing in the text
times, frames or moves the camera (the engine owns those).
"""

from __future__ import annotations

import re
from typing import Any

from .render import POSES

KINDS = ("projectile", "at_target")
# mechanic -> {field: (form, max words)}; "verb" fields are verb phrases of the character, "clause"
# fields carry their own subject.
FIELDS: dict[str, dict[str, tuple[str, int]]] = {
    "ranged_attack": {"windup": ("verb", 32), "release": ("verb", 24), "travel": ("clause", 24),
                      "impact": ("clause", 26), "decay": ("clause", 18), "again": ("verb", 34),
                      "text": ("verb", 45)},
    "melee_attack": {"windup": ("verb", 30), "impact": ("clause", 26), "decay": ("clause", 18),
                     "recover": ("verb", 16), "again": ("verb", 30), "text": ("verb", 40)},
    "dash": {"text": ("verb", 24)},
}
NEEDS_TARGET = {"ranged_attack": ("windup", "impact", "again", "text"),
                "melee_attack": ("impact", "again", "text"), "dash": ()}
LIGHT = re.compile(r"glow|blaz|bright|light|flash|lumin|radian|shin(e|ing)|gleam|incandesc|burn|fier|flame|"
                   r"spark|crackl|white-hot|shimmer|flare|ember|fire|beam|blinding", re.I)
COLOUR = re.compile(r"\b(red|orange|yellow|gold(en)?|white|blue|cyan|violet|purple|green|emerald|crimson|scarlet|"
                    r"silver|black|amber|azure|magenta|pink|teal|indigo|jade|ruby)\b", re.I)
GONE = re.compile(r"vanish|fade|dissolv|die(s)? (out|away)|wink(s)? out|disappear|gone|clear|snuff|flicker(s)? out",
                  re.I)
QUICK = re.compile(r"instant|split second|at once|moment|immediately|heartbeat|quickly|swiftly|in a blink|"
                   r"as fast as|just as fast", re.I)
FORBIDDEN = re.compile(r"\b(camera|screen|frame|frames|(?<!split )seconds?|slow[- ]motion|gore|wound\w*|bleed\w*|gash\w*|"
                       r"dismember\w*|corpse|entire courtyard|whole courtyard|fills the (yard|courtyard|air))\b",
                       re.I)
SUBJECT_START = re.compile(r"^(he|she|they|it|his|her|their|the|a|an)\b", re.I)
PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _words(text: str) -> int:
    return len(text.split())


def validate(spec: dict[str, Any], pronoun: str = "they") -> list[str]:
    """Problems with ``spec`` (empty when it can go into the roster)."""
    problems: list[str] = []
    other = {"he": r"\b(she|her|hers)\b", "she": r"\b(he|his|him)\b"}.get(pronoun)
    for mechanic, fields in FIELDS.items():
        skill = spec.get(mechanic)
        if not isinstance(skill, dict):
            problems.append(f"{mechanic}: missing")
            continue
        where = mechanic
        if not str(skill.get("name") or "").strip():
            problems.append(f"{where}.name: missing")
        if mechanic in POSES and skill.get("pose") not in POSES[mechanic]:
            problems.append(f"{where}.pose: must be one of {sorted(POSES[mechanic])}")
        if mechanic == "ranged_attack" and skill.get("kind") not in KINDS:
            problems.append(f"{where}.kind: must be one of {KINDS}")
        for name, (form, limit) in fields.items():
            text = str(skill.get(name) or "").strip()
            at = f"{where}.{name}"
            if not text:
                problems.append(f"{at}: missing")
                continue
            if _words(text) > limit:
                problems.append(f"{at}: {_words(text)} words, at most {limit}")
            if not text[0].islower():
                problems.append(f"{at}: must start lower-case (it continues a sentence)")
            if form == "verb" and SUBJECT_START.match(text):
                problems.append(f"{at}: a verb phrase of the character, without a subject")
            if text.endswith("."):
                problems.append(f"{at}: no final full stop")
            bad = [p for p in PLACEHOLDER.findall(text) if p != "{target}"]
            if bad:
                problems.append(f"{at}: only {{target}} may appear in braces, not {bad}")
            if name in NEEDS_TARGET[mechanic] and "{target}" not in text:
                problems.append(f"{at}: must name {{target}}")
            if FORBIDDEN.search(text):
                problems.append(f"{at}: drop {FORBIDDEN.search(text).group(0)!r} (the engine owns camera and "
                                f"timing; no gore or effects that cover the whole view)")
            if other and re.search(other, text, re.I):
                problems.append(f"{at}: the character is {pronoun!r}")
        if mechanic == "ranged_attack":
            for name in ("release", "travel", "impact"):
                if skill.get(name) and not LIGHT.search(skill[name]):
                    problems.append(f"{where}.{name}: describe the effect as light (glowing, blazing, flash ...)")
            if skill.get("kind") == "at_target" and "{target}" not in str(skill.get("travel") or ""):
                problems.append(f"{where}.travel: an at_target skill forms at {{target}}; name it")
        if mechanic in ("ranged_attack", "melee_attack"):
            for name in ("impact", ) + (("release", ) if mechanic == "ranged_attack" else ()):
                if skill.get(name) and not COLOUR.search(skill[name]):
                    problems.append(f"{where}.{name}: name the colour of the light")
            decay = str(skill.get("decay") or "")
            if decay and not (GONE.search(decay) and QUICK.search(decay)):
                problems.append(f"{where}.decay: say the effect vanishes at once (e.g. 'vanishes in a split second')")
            if mechanic == "melee_attack" and skill.get("impact") and not LIGHT.search(skill["impact"]):
                problems.append(f"{where}.impact: give the blow a visible light effect")
    return problems


def merge(entry: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The roster entry with its skill tree taken from ``spec`` (moves and scene stay)."""
    skills = {}
    for mechanic, fields in FIELDS.items():
        row = spec[mechanic]
        skill = {"name": row["name"], "text": row["text"]}
        if "again" in fields:
            skill["again"] = row["again"]
        phases = {k: row[k] for k in fields if k not in ("text", "again")}
        if phases:
            skill["phases"] = phases
        for key in ("pose", "kind"):
            if row.get(key):
                skill[key] = row[key]
        skills[mechanic] = skill
    return {**entry, "skills": skills}
