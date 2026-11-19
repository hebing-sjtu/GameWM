"""Write a roster entry for every character card with a language model, validate it, merge them.

    python scripts/oc_author.py --cards cards.json --out oc_entries/ --merged roster_oc.json \\
        [--only 09_sand_sorcerer] [--backend claude|files] [--model opus] [--refs effect_refs.json]

A card is an identity sheet and a few lines about the character (``gamewm_engine/games/training_ground/oc.py``);
the entry is everything the captions say about it: look, themed arena, skill phases, moves. With
``--backend claude`` the model is the ``ask-claude`` CLI, allowed to read the sheet image; with
``--backend files`` the script writes ``<id>.prompt<N>.txt`` and checks the ``<id>.reply<N>.txt``
someone (or another model) writes next to it, so any model can stand in. A reply that fails
``oc.validate`` goes back with its problems, up to ``--tries``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import effect_specs  # noqa: E402
from gamewm_engine.games.training_ground import effects, oc  # noqa: E402
from gamewm_engine.games.training_ground import rules as R  # noqa: E402
from gamewm_engine.games.training_ground.render import POSES  # noqa: E402
from gamewm_engine.protocol import MOVES  # noqa: E402

EXAMPLE = json.loads((ROOT / "examples" / "training_ground" / "oc_example.json").read_text(encoding="utf-8"))

CAPTION = """\
Each 5-second segment of a take gets one caption, assembled by the engine from your entry:

  A 3D video game captures <setting>. <lighting>. The protagonist from the identity sheet, <look>, <action>.

<action> is the engine's events in order, each written with your texts: a move ("<moves.walk_to>"),
then every skill use phase by phase ("He <windup>. He <release>, and <travel>. <impact>, rocking it
back on its stand. <decay>."), a repeat as "He once more <windup>. He <again>", a dash as
"He <dash.text>. <dash.phases.decay>. He <dash.phases.recover>." The engine fills {target} with
"the first training dummy on the left" and the like, and moves the camera itself."""

ARENA = """\
The arena: a walled square yard with three grey dummies, some boxes and tall blocks, restyled into
the character's world by an image model (``props`` is that instruction) and named in every caption
(``setting``, ``lighting``). The same text is rewritten for a long corridor and for two yards joined
by a gateway, so keep to this shape:
  setting: "a walled <material> training courtyard (or yard) <of/in a place that suits the character>
    with <one wall detail>, three <kind of> training dummies on round stands, <crates> and <square
    pillars>" (<= 48 words, no final full stop).
  lighting: one capitalised sentence, <= 18 words, no final full stop; daylight or another bright,
    even light, so the dummies and the effects read (no night, no fog that hides the dummies).
  props: "The walls become ..., the ground is ..., the three grey dummies become ... on round ...
    stands, the boxes become ... and the tall blocks become ...." (<= 75 words, final full stop).
  Skill and move texts never name the arena, except a projectile's travel "across the courtyard"
  (or "across the yard")."""


def prompt(card: dict, sheet: Path, refs: str) -> str:
    fields = "\n".join(f"  skills.{m}: " + ", ".join(f"{k} ({form}, <= {n} words)" for k, (form, n) in f.items())
                       for m, f in effects.FIELDS.items())
    dash = ", ".join(f"{k} ({form}, <= {n} words)" for k, (form, n) in oc.DASH_PHASES.items())
    poses = "\n".join(f"  {m}: " + "; ".join(
        f"{p} = {effect_specs.POSE_NOTES.get(f'{m}.{p}', effect_specs.POSE_NOTES.get(p))}" for p in POSES[m])
        for m in POSES)
    moves = ", ".join(f"{k}" + (" (names {target})" if k in oc.TARGETED else "") for k in MOVES)
    given = "\n".join(f"  {k}: {card[k]}" for k in ("id", "brief", "pronoun", "theme") if card.get(k))
    refs_block = (f"\nHand-written effect references (Chinese; one-off video-to-video prompts, so ignore their "
                  f"seconds, camera and GTA wording, and keep what they pin down about shape, material, colour, "
                  f"count and decay):\n{refs}\n" if refs else "")
    return f"""You write the roster entry of one character in a third-person action game. Its English texts
become the captions of a video world model that repaints a grey-box game proxy into the character's
world, so every word you write must be something the video can show.

First read the character's identity sheet (front, side and back views of the same figure) with your
Read tool: {sheet}
Describe only what the sheet shows; the brief below names the role and the skills.

Character card
{given}

{CAPTION}

{ARENA}

The character
  look: a noun phrase, <= 30 words, starting "a ..." / "an ...": build or age only if striking, hair
    (colour, length, beard), the main garments with their colours and materials, and what is held and
    how ("holding a short wand", "carrying a heavy war hammer"). It is said from behind and at a
    distance: name the silhouette and the colours, not the eye colour.
  pronoun: he | she | they, as the sheet shows (or as the card says).
  moves: verb phrases of the character for {moves}; each <= {oc.MOVE_WORDS} words, in the character's
    own manner (gait, what the clothes and the held item do). No camera, no arena name.

Skills (engine timing, do not write numbers: ranged windup {R.RANGED.windup} s, then the shot flies
about half a second; melee windup {R.MELEE.windup} s, the blow lands at its end, recovery
{R.MELEE.recovery} s; dash {R.DASH.duration} s over {R.DASH.distance:.0f} m). Keep the card's skill
names; give each a look that belongs to this character.
{effect_specs.LESSONS}{refs_block}
Field forms ("verb" = verb phrase of the character without a subject, e.g. "raises his wand ...";
"clause" = own subject, e.g. "the fireball streaks ..."; all start lower-case, no final full stop,
{{target}} is the only placeholder):
{fields}
  skills.dash.phases: {dash}
  skills.ranged_attack also has kind (projectile | at_target) and pose; skills.melee_attack has pose.
  Poses (the grey-box body plays this gesture, so pick the one that matches the weapon):
{poses}
  `again` is a repeat of the skill in the same caption: compact, but with the full visible effect and
  its quick decay. `text` is the whole skill in one verb phrase. `phases` holds the other fields.

Format example for a different character (card: {json.dumps(EXAMPLE["card"], ensure_ascii=False)}):
{json.dumps(EXAMPLE["entry"], indent=1, ensure_ascii=False)}

Reply with only the JSON object of the entry for {card["id"]}, no markdown fences."""


def ask(text: str, sheet: Path, model: str) -> str:
    cmd = ["ask-claude", "-p", text, "--output-format", "text", "--allowedTools", "Read",
           "--add-dir", str(sheet.parent), "--max-turns", "4"] + (["--model", model] if model else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        raise RuntimeError(f"ask-claude failed ({out.returncode}): {(out.stderr or out.stdout).strip()[:400]}")
    return out.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", required=True, help="JSON list of cards, or {'sheets': DIR, 'cards': [...]}")
    ap.add_argument("--sheets", default="", help="directory of the cards' sheet files (default: next to --cards)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--base", default=str(ROOT / "examples" / "training_ground" / "roster_fx.json"),
                    help="roster whose header (game, sheet_pattern, medium) the merged roster keeps")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--backend", choices=("claude", "files"), default="claude")
    ap.add_argument("--model", default="")
    ap.add_argument("--refs", default="")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args()
    cards_path = Path(args.cards)
    raw = json.loads(cards_path.read_text(encoding="utf-8"))
    sheets = cards_path.parent / raw.get("sheets", ".") if isinstance(raw, dict) else cards_path.parent
    sheets = Path(args.sheets) if args.sheets else sheets
    cards = raw["cards"] if isinstance(raw, dict) else raw
    refs = effect_specs.references(Path(args.refs)) if args.refs else ""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calls = 0
    for card in cards:
        path = out / f"{card['id']}.json"
        if (args.only and card["id"] not in args.only) or path.is_file():
            continue
        sheet = (sheets / card["sheet"]).resolve()
        if not sheet.is_file():
            raise SystemExit(f"{card['id']}: no sheet at {sheet}")
        text = prompt(card, sheet, refs)
        for attempt in range(1, args.tries + 1):
            reply_path = out / f"{card['id']}.reply{attempt}.txt"
            if args.backend == "files":
                (out / f"{card['id']}.prompt{attempt}.txt").write_text(text, encoding="utf-8")
                if not reply_path.is_file():
                    print(f"{card['id']}: waiting for {reply_path}", flush=True)
                    break
                reply = reply_path.read_text(encoding="utf-8")
            else:
                reply = ask(text, sheet, args.model)
                calls += 1
                reply_path.write_text(reply, encoding="utf-8")
            try:
                entry = oc.entry(effect_specs.parse(reply))
                if card.get("pronoun") and entry.get("pronoun") != card["pronoun"]:
                    entry["pronoun"] = card["pronoun"]
                problems = oc.validate(entry, card["id"])
            except (ValueError, json.JSONDecodeError) as err:
                entry, problems = None, [f"reply is not one JSON object: {err}"]
            print(f"{card['id']} try {attempt}: {len(problems)} problems", flush=True)
            if not problems:
                path.write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                break
            text = (prompt(card, sheet, refs) + "\n\nYour previous reply:\n" + reply.strip() +
                    "\n\nIt has these problems; reply with the corrected JSON object only:\n- " + "\n- ".join(problems))
        else:
            print(f"{card['id']}: gave up after {args.tries} tries", flush=True)
    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    merged = {k: v for k, v in base.items() if k != "characters"}
    merged["characters"] = []
    for card in cards:
        path = out / f"{card['id']}.json"
        if path.is_file():
            entry = json.loads(path.read_text(encoding="utf-8"))
            problems = oc.validate(entry, card["id"])
            if problems:
                raise SystemExit(f"{path}: {problems}")
            merged["characters"].append(entry)
    Path(args.merged).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{calls} model calls; {len(merged['characters'])} entries -> {args.merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
