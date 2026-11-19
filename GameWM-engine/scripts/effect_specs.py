"""Write every roster character's effect spec with a language model, validate it, merge it.

    python scripts/effect_specs.py --roster examples/training_ground/roster.json \\
        --refs effect_refs.json --out examples/training_ground/effects \\
        --merged examples/training_ground/roster_fx.json [--only 01_fire_mage] [--model opus]

The model is the ``ask-claude`` CLI (``-p ... --output-format text``). ``--refs`` is a JSON list of
hand-written effect prompts that worked (``[{"id", "submissions": {"effect": {"prompt"}}}]``); their
choreography paragraphs are quoted as references for what a good effect description pins down. A
spec that fails ``effects.validate`` goes back to the model with the problems, up to ``--tries``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gamewm_engine.games.training_ground import effects  # noqa: E402
from gamewm_engine.games.training_ground import rules as R  # noqa: E402
from gamewm_engine.games.training_ground.render import POSES  # noqa: E402

POSE_NOTES = {
    "point": "one arm raised straight at the target (wand, open hand, sceptre)",
    "staff": "two hands on a long staff levelled at the target",
    "throw": "overarm throw: the hand cocks back above the shoulder, then snaps forward",
    "overhead": "the arm (and weapon) raised to the sky during the charge, then brought down to point at the target",
    "melee_attack.overhead": "a heavy weapon raised overhead with both hands, then slammed down onto the target",
    "rifle": "a long gun shouldered with both hands, held steady",
    "push": "both palms gathered at the chest, then pushed out together",
    "punch": "a fist drawn back to the hip, then driven straight out",
    "palm": "an open-palm strike from the hip",
    "thrust": "a two-handed jab (staff head, rifle stock) pulled back, then driven forward",
    "swing": "a weapon swung wide from behind the shoulder in a horizontal arc",
    "claw": "a clawed hand raised high outside, then raked down across the body",
}
LESSONS = """\
What made hand-written effect prompts work on this video model, and what our own renders showed:
1. Name the effect's material, shape, count and colour exactly ("eight thin rigid vine spears", "a thin layer
   of black flame with dark silver edges"), not a generic "magic attack".
2. Effects must be written as LIGHT: glowing, blazing, a bright flash, a streak of light. Solid objects
   (steel darts, stones) were invisible until rewritten as glowing streaks "like tiny comets".
3. The caster's whole body performs the cast: turning the shoulders, bending the elbow, swinging the arm,
   feet planted. A staff is swung, a hammer raised; never reuse a gun-aiming pose for a spell.
4. The charge builds during the windup and is complete before the release; detail goes into the charge,
   the release instant and the hit (solid outline, key light, fine lines, directional sparks).
5. The hit is short and decisive. The effect then clears off the target at once (within a split second) so
   the target stays fully visible; no lingering glow, no floating weapons, nothing covering the whole view.
6. Say where the effect happens: a projectile crossing the space, or an effect forming on/at the target
   (from the ground at its feet, from the sky above it). The engine shows the target rocking or falling.
7. No camera, timing, seconds or frame words (the engine owns them); no gore.
"""


def references(path: Path, limit: int = 3) -> str:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for row in rows:
        prompt = row["submissions"]["effect"].get("prompt") or ""
        body = prompt.split("detailed_description:", 1)[-1]
        body = body.split("目标身体动作是独立保留层，不能跟随新法术重新编排。", 1)[-1]
        for stop in ("统一时间线", "Camera original clock", "Camera clock", "overall_soundscape:"):
            body = body.split(stop, 1)[0]
        body = body.strip()
        quality = re.search(r"精致细节[^\n]*", prompt)
        if len(body) < 40:
            continue
        out.append(f"- {row.get('title') or row['id']}: {body}" + (f"\n  {quality.group(0)}" if quality else ""))
        if len(out) >= limit:
            break
    return "\n".join(out)


EXAMPLE = {
    "ranged_attack": {
        "name": "Sun Lance", "kind": "projectile", "pose": "point",
        "windup": ("plants his feet, turns his shoulders side-on and raises the bronze sceptre at {target}, a small "
                   "sun of golden light gathering at its head"),
        "release": "snaps the sceptre forward and a blazing golden lance of light shoots from its head",
        "travel": "the lance streaks across the courtyard as a bright white-gold line trailing shimmering heat",
        "impact": "it bursts against {target} in a sharp golden flash and a ring of glowing sand",
        "decay": "the golden light vanishes in a split second, leaving {target} in plain view",
        "again": ("swings the sceptre up again and fires another blazing golden lance that flashes against {target} "
                  "and vanishes at once"),
        "text": ("raises his bronze sceptre at {target}, golden light gathering at its head, and fires a blazing golden "
                 "lance that bursts in a sharp flash on it and vanishes at once"),
    },
    "melee_attack": {
        "name": "Dune Strike", "pose": "swing",
        "windup": "steps in and swings the sceptre wide behind his shoulder, its head glowing amber",
        "impact": "the glowing sceptre sweeps into {target} in a bright arc of amber light and a spray of sand",
        "decay": "the amber arc fades at once",
        "recover": "brings the sceptre back across his body",
        "again": "sweeps the glowing sceptre into {target} again in a bright amber arc that fades at once",
        "text": "swings his glowing sceptre into {target} in a bright arc of amber light that fades at once",
    },
    "dash": {"name": "Sand Step", "text": "dashes forward low in a swirl of glowing golden sand"},
}


def prompt(entry: dict, refs: str) -> str:
    fields = "\n".join(f"  {m}: " + ", ".join(f"{k} ({form}, <= {n} words)" for k, (form, n) in f.items())
                       for m, f in effects.FIELDS.items())
    poses = "\n".join(f"  {m}: " + "; ".join(f"{p} = {POSE_NOTES.get(f'{m}.{p}', POSE_NOTES.get(p))}" for p in POSES[m])
                      for m in POSES)
    current = {m: {"name": s["name"], "text": s["text"]} for m, s in entry["skills"].items()}
    return f"""You write the skill-effect spec of one character in a third-person action game. Its English text
becomes the caption of a video world model that repaints a grey-box game proxy into the character's world,
phase by phase: windup (cast pose + charge), release, travel, impact, decay; melee: windup, impact, decay,
recover. The engine fills {{target}} (a wooden training dummy 7-10 m away for ranged, at arm's reach for
melee), adds the dummy's reaction and moves the camera.

Engine timing (do not write numbers): ranged windup {R.RANGED.windup} s, then the shot flies about half a
second; melee windup {R.MELEE.windup} s, the blow lands at its end, recovery {R.MELEE.recovery} s; dash
{R.DASH.duration} s over {R.DASH.distance:.0f} m.

{LESSONS}
Hand-written references (Chinese; one-off video-to-video prompts, so ignore their seconds, camera and
GTA wording, and keep what they pin down about shape, material, colour, count and decay):
{refs}

Character
  id: {entry["id"]}
  look: {entry["look"]}
  pronoun: {entry.get("pronoun", "they")}
  setting: {entry["setting"]}
  current skills (keep each name and its theme; rewrite the text): {json.dumps(current, ensure_ascii=False)}

Fields ("verb" = verb phrase of the character without a subject, e.g. "raises his wand ..."; "clause" =
own subject, e.g. "the fireball streaks ..."; all start lower-case, no final full stop, {{target}} is the
only placeholder):
{fields}
  ranged_attack also has kind (projectile | at_target) and pose; melee_attack has pose. Poses (the grey-box
  body plays this gesture, so pick the one that matches the character's weapon):
{poses}
  `again` is a repeat of the skill in the same caption: compact, but with the full visible effect and its
  quick decay. `text` is the whole skill in one verb phrase.

Format example for a different character (a desert sorcerer with a bronze sceptre):
{json.dumps(EXAMPLE, indent=1)}

Reply with only the JSON object for {entry["id"]}, no markdown fences."""


def ask(text: str, model: str) -> str:
    cmd = ["ask-claude", "-p", text, "--output-format", "text"] + (["--model", model] if model else [])
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(f"ask-claude failed ({out.returncode}): {out.stderr.strip()[:400]}")
    return out.stdout


def parse(reply: str) -> dict:
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the reply")
    return json.loads(reply[start:end + 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--model", default="")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--files", action="store_true",
                    help="no model calls: write <id>.prompt<N>.txt and check the <id>.reply<N>.txt someone wrote")
    args = ap.parse_args()
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    refs = references(Path(args.refs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    calls = 0
    for entry in roster["characters"]:
        path = out / f"{entry['id']}.json"
        if (args.only and entry["id"] not in args.only) or (args.files and path.is_file()):
            continue
        text = prompt(entry, refs)
        for attempt in range(1, args.tries + 1):
            reply_path = out / f"{entry['id']}.reply{attempt}.txt"
            if args.files:
                (out / f"{entry['id']}.prompt{attempt}.txt").write_text(text, encoding="utf-8")
                if not reply_path.is_file():
                    print(f"{entry['id']}: waiting for {reply_path}", flush=True)
                    break
                reply = reply_path.read_text(encoding="utf-8")
            else:
                reply = ask(text, args.model)
                calls += 1
                reply_path.write_text(reply, encoding="utf-8")
            try:
                spec = parse(reply)
                problems = effects.validate(spec, entry.get("pronoun", "they"))
            except (ValueError, json.JSONDecodeError) as err:
                spec, problems = None, [f"reply is not one JSON object: {err}"]
            print(f"{entry['id']} try {attempt}: {len(problems)} problems", flush=True)
            if not problems:
                path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                break
            text = (prompt(entry, refs) + "\n\nYour previous reply:\n" + reply.strip() +
                    "\n\nIt has these problems; reply with the corrected JSON object only:\n- " + "\n- ".join(problems))
        else:
            print(f"{entry['id']}: gave up after {args.tries} tries", flush=True)
    merged = dict(roster)
    merged["characters"] = []
    for entry in roster["characters"]:
        path = out / f"{entry['id']}.json"
        if path.is_file():
            spec = json.loads(path.read_text(encoding="utf-8"))
            problems = effects.validate(spec, entry.get("pronoun", "they"))
            if problems:
                raise SystemExit(f"{path}: {problems}")
            entry = effects.merge(entry, spec)
        merged["characters"].append(entry)
    Path(args.merged).write_text(json.dumps(merged, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{calls} model calls; merged roster -> {args.merged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
