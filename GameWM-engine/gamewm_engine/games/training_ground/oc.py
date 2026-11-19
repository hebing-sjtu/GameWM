"""Character cards: an OC's identity sheet and a few lines about it, turned into a roster entry.

A roster entry is everything the training ground's captions say about a character: ``look`` (the
identity line), the themed arena the grey boxes are restyled into (``setting``, ``lighting``,
``props``), the skill tree phase by phase (``skills``, see ``effects``) and how it walks, turns and
waits (``moves``). The narration only places these texts around the engine's events, so a new
character's captions are as good as its entry. ``scripts/oc_author.py`` has a language model write
the entry from a card; ``validate`` holds it to what the captions and the layout rewrites need.

A card::

    {"id": "09_sand_sorcerer", "sheet": "sheets/09_sand_sorcerer.jpg", "pronoun": "he",
     "brief": "a desert sorcerer with a bronze sceptre; ranged Sun Lance, melee Dune Strike, dash Sand Step",
     "theme": "a desert fortress"}

``pronoun`` and ``theme`` are optional (the model reads the pronoun off the sheet and picks an arena
that suits the character).
"""

from __future__ import annotations

import re
from typing import Any

from ...protocol import MOVES
from . import effects
from .layout import LAYOUTS

ENTRY_KEYS = ("id", "look", "pronoun", "setting", "lighting", "props", "skills", "moves")
PRONOUNS = ("he", "she", "they")
# dash phases told after the dash itself in the ``detailed`` style
DASH_PHASES: dict[str, tuple[str, int]] = {"decay": ("clause", 18), "recover": ("verb", 16)}
MOVE_WORDS = 22
TARGETED = ("walk_to", "turn_to")
PROPS_PARTS = ("The walls become", "the ground is", "the three grey dummies become", "the boxes become",
               "the tall blocks become")
# the arena's name in a skill or move only reads right in every layout where a rewrite handles it
PLACE = re.compile(r"\b(courtyard|yard)s?\b", re.I)
SAFE_PLACE = re.compile(r"across the (court)?yard", re.I)
SETTLES = re.compile(effects.GONE.pattern + r"|melt|thin(s)? away|settl|go(es)? out|die(s)? away|drift", re.I)
META = re.compile(r"identity sheet|protagonist|turnaround|character sheet|\bOC\b", re.I)


def spec(entry: dict[str, Any]) -> dict[str, Any]:
    """The entry's skill tree in the flat shape ``effects.validate`` checks."""
    out = {}
    for mechanic, skill in (entry.get("skills") or {}).items():
        row = {k: v for k, v in skill.items() if k != "phases"}
        if mechanic != "dash":
            row.update(skill.get("phases") or {})
        out[mechanic] = row
    return out


def _other(pronoun: str) -> str | None:
    return {"he": r"\b(she|her|hers|herself)\b", "she": r"\b(he|his|him|himself)\b"}.get(pronoun)


def _phrase(text: Any, at: str, limit: int, form: str, pronoun: str, targets: bool = False) -> list[str]:
    """Problems with one text that continues a caption sentence (``verb``: of the character, without
    a subject; ``clause``: with its own subject; ``noun``: a noun phrase)."""
    text = str(text or "").strip()
    if not text:
        return [f"{at}: missing"]
    problems = []
    if len(text.split()) > limit:
        problems.append(f"{at}: {len(text.split())} words, at most {limit}")
    if not text[0].islower():
        problems.append(f"{at}: must start lower-case (it continues a sentence)")
    if text.endswith("."):
        problems.append(f"{at}: no final full stop")
    if form == "verb" and effects.SUBJECT_START.match(text):
        problems.append(f"{at}: a verb phrase of the character, without a subject")
    braces = effects.PLACEHOLDER.findall(text)
    if targets and "{target}" not in braces:
        problems.append(f"{at}: must name {{target}}")
    if [b for b in braces if not (targets and b == "{target}")]:
        problems.append(f"{at}: no placeholders here except {{target}} where it is asked for, not {braces}")
    if effects.FORBIDDEN.search(text):
        problems.append(f"{at}: drop {effects.FORBIDDEN.search(text).group(0)!r} (the engine owns camera and timing)")
    if META.search(text):
        problems.append(f"{at}: describe the character, not the sheet ({META.search(text).group(0)!r})")
    other = _other(pronoun)
    if other and re.search(other, text, re.I):
        problems.append(f"{at}: the character is {pronoun!r}")
    return problems


def _arena(entry: dict[str, Any]) -> list[str]:
    problems = []
    setting, lighting, props = (str(entry.get(k) or "").strip() for k in ("setting", "lighting", "props"))
    if not re.match(r"^a ([\w-]+ ){0,3}walled ", setting):
        problems.append("setting: must start 'a walled ...' (the layouts rewrite it into a corridor or two yards)")
    if len(setting.split()) > 48:
        problems.append(f"setting: {len(setting.split())} words, at most 48")
    for need, why in ((r"\b(courtyard|yard)\b", "name the place a courtyard or yard"),
                      (r"\bthree\b[^,]*\bon round stands\b", "say 'three ... training dummies on round stands'"),
                      (r"\bcrates?\b", "name the crates"), (r"\bpillars?\b", "name the pillars")):
        if not re.search(need, setting):
            problems.append(f"setting: {why}")
    if setting.endswith("."):
        problems.append("setting: no final full stop")
    if not lighting or not lighting[0].isupper() or lighting.endswith(".") or len(lighting.split()) > 18:
        problems.append("lighting: one sentence of at most 18 words, capitalised, without its final full stop")
    if effects.FORBIDDEN.search(lighting):
        problems.append(f"lighting: drop {effects.FORBIDDEN.search(lighting).group(0)!r}")
    for part in PROPS_PARTS:
        if part not in props:
            problems.append(f"props: must say '{part} ...'")
    if not props.endswith(".") or len(props.split()) > 75:
        problems.append("props: one or two sentences, at most 75 words, ending with a full stop")
    for layout in LAYOUTS.values():
        for name, pattern, _, count in layout.rewrites:
            text = {"setting": setting, "props": props}.get(name)
            if text is not None and count == 1 and not re.search(pattern, text):
                problems.append(f"{name}: the {layout.id} layout cannot rewrite it (no match for {pattern!r})")
    return problems


def validate(entry: dict[str, Any], card_id: str | None = None) -> list[str]:
    """Problems with a roster entry (empty when the captions can use it in every layout)."""
    problems = [f"{k}: missing" for k in ENTRY_KEYS if k not in entry]
    if problems:
        return problems
    if card_id and entry["id"] != card_id:
        problems.append(f"id: must be {card_id!r}")
    pronoun = entry["pronoun"]
    if pronoun not in PRONOUNS:
        return problems + [f"pronoun: one of {PRONOUNS}"]
    look = str(entry["look"]).strip()
    problems += _phrase(look, "look", 30, "noun", pronoun)
    if not re.match(r"^an? ", look):
        problems.append("look: a noun phrase starting 'a ...' / 'an ...'")
    problems += _arena(entry)
    problems += effects.validate(spec(entry), pronoun)
    dash = (entry["skills"].get("dash") or {}).get("phases") or {}
    for name, (form, limit) in DASH_PHASES.items():
        problems += _phrase(dash.get(name), f"dash.phases.{name}", limit, form, pronoun)
    if dash.get("decay") and not SETTLES.search(dash["decay"]):
        problems.append("dash.phases.decay: say how the trail fades or settles")
    for mechanic, skill in entry["skills"].items():
        texts = [skill.get("text"), skill.get("again"), *(skill.get("phases") or {}).values()]
        if any(PLACE.search(SAFE_PLACE.sub("", str(t or ""))) for t in texts):
            problems.append(f"{mechanic}: name the arena only as 'across the courtyard' / 'across the yard' "
                            "(other layouts rewrite that)")
    moves = entry["moves"]
    for kind in MOVES:
        text = moves.get(kind)
        problems += _phrase(text, f"moves.{kind}", MOVE_WORDS, "verb", pronoun, targets=kind in TARGETED)
        if text and PLACE.search(text):
            problems.append(f"moves.{kind}: do not name the courtyard (the same move plays in every layout)")
    unknown = sorted(set(moves) - set(MOVES))
    if unknown:
        problems.append(f"moves: unknown kinds {unknown}; known: {sorted(MOVES)}")
    return problems


def entry(raw: dict[str, Any]) -> dict[str, Any]:
    """A model's reply in roster order, without keys the roster does not use."""
    out = {k: raw[k] for k in ENTRY_KEYS if k in raw}
    skills = {}
    for mechanic in effects.FIELDS:
        skill = dict((raw.get("skills") or {}).get(mechanic) or {})
        keep = ("name", "text", "again", "phases", "pose", "kind")
        skills[mechanic] = {k: skill[k] for k in keep if k in skill}
    out["skills"] = skills
    out["moves"] = {k: (raw.get("moves") or {})[k] for k in MOVES if k in (raw.get("moves") or {})}
    return out
