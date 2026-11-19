"""Re-caption recorded takes with another roster and compare them with the captions they were sent with.

    python scripts/oc_regress.py --roster roster_oc.json --runs RUNS_PARENT --report report.md \\
        [--layouts corridor courtyard rooms] [--prefix runs_dt_]

A take's ``request.json`` holds the engine's action text per segment with ``{skill:...}`` /
``{move:...}`` placeholders, so swapping the character (look, skills, moves) and the arena text
(setting, lighting), retexted for the take's layout, gives the captions that roster would have
produced for the very same play. ``RUNS_PARENT/<prefix><layout>/<id>/take/{request,plan}.json``
are the recorded takes; ``plan.json`` has the captions they were rendered with.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gamewm_engine import compose, protocol  # noqa: E402
from gamewm_engine.games.training_ground import oc  # noqa: E402
from gamewm_engine.games.training_ground.layout import LAYOUTS  # noqa: E402


def recaption(take: Path, entry: dict, layout: str) -> tuple[list[str], list[str]]:
    """(captions with ``entry``, captions the take was sent with)."""
    payload = json.loads((take / "request.json").read_text(encoding="utf-8"))
    old = json.loads((take / "plan.json").read_text(encoding="utf-8"))
    e = LAYOUTS[layout].retext(entry)
    character = payload["assets"]["character"]
    payload["assets"]["character"] = {**character, "look": e["look"], "pronoun": e["pronoun"], "skills": e["skills"],
                                      "moves": e["moves"]}
    payload["assets"]["scene"] = {**payload["assets"]["scene"], "setting": e["setting"], "lighting": e["lighting"]}
    request = protocol.parse(payload, base=take, check_files=False)
    new = compose.plan(request, old["proxy_frames"])
    return [s.caption for s in new.segments], [s["caption"] for s in old["segments"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roster", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--layouts", nargs="*", default=["corridor", "courtyard", "rooms"])
    ap.add_argument("--prefix", default="runs_dt_")
    ap.add_argument("--show", type=int, default=1, help="segments per take quoted in full")
    args = ap.parse_args()
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    lines = [f"# Re-captioned takes: `{Path(args.roster).name}` against the recorded captions", ""]
    lines += ["| layout | character | problems | segments | words per segment (new / recorded) |", "|---|---|---|---|---|"]
    quotes = []
    for layout in args.layouts:
        for entry in roster["characters"]:
            take = Path(args.runs) / f"{args.prefix}{layout}" / entry["id"] / "take"
            if not (take / "request.json").is_file():
                continue
            problems = oc.validate(entry, entry["id"])
            new, old = recaption(take, entry, layout)
            nw = [len(c.split()) for c in new]
            ow = [len(c.split()) for c in old]
            lines.append(f"| {layout} | {entry['id']} | {len(problems)} | {len(new)} | "
                         f"{min(nw)}-{max(nw)} / {min(ow)}-{max(ow)} |")
            for i in range(min(args.show, len(new))):
                quotes += [f"## {layout} · {entry['id']} · segment {i}", "", "New:", "", f"> {new[i]}", "",
                           "Recorded:", "", f"> {old[i]}", ""]
    Path(args.report).write_text("\n".join(lines + [""] + quotes) + "\n", encoding="utf-8")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
