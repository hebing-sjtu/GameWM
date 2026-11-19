"""The training ground's authored-once pass and a full play-through for every character.

    # 1. opening view, sheet cut-outs, collages and restyle prompts, one pack per character
    python -m gamewm_engine.games.training_ground.author prepare \\
        --roster examples/training_ground/roster.json --sheets DIR_WITH_SHEETS --out packs/
    #    (--layout corridor|rooms for another arena; after editing only the roster's text:
    #     author retext --roster ... --out packs/)
    # 2. paint packs/<id>/first_frame.png from packs/<id>/collage.png with packs/<id>/prompt.txt
    # 3. review every painting against the proxy layout
    python -m gamewm_engine.games.training_ground.author review --out packs/
    # 4. play the demo once per character and write its v1 take
    python -m gamewm_engine.games.training_ground.author play --out packs/ --runs runs/ --workers 32
    # 5. render the takes (python -m gamewm_engine submit-ui runs/*/take/request.json --out renders/), then
    python -m gamewm_engine.games.training_ground.author overview --out packs/ --runs runs/ \\
        --renders renders/ --video overview.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ... import authoring
from ...protocol import GEOMETRY
from ...stream import DirectorySink, PreviewSink, pump
from . import narrate
from . import rules as R
from .game import TrainingGround
from .inputs import CORRIDOR_LONG, DEMO, LONG, ROOMS_LONG, SHOWCASE
from .layout import LAYOUTS

OPENING_SIZE = (1344, 768)
SCRIPTS = {"demo": DEMO, "showcase": SHOWCASE, "long": LONG, "corridor": CORRIDOR_LONG, "rooms": ROOMS_LONG}


def _layout(pack: authoring.Pack) -> str:
    return pack.meta.get("authoring", {}).get("layout", "courtyard")


def _character(entry: dict) -> dict:
    return {"subject": "protagonist", "look": entry["look"], "pronoun": entry.get("pronoun", "they"),
            "skills": entry["skills"], "moves": entry.get("moves", {})}


def retext(args: argparse.Namespace) -> None:
    """Refresh the text of existing packs (look, skills, moves, scene) from the roster; the
    painted first frame and the sheet stay as they are."""
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    for entry in roster["characters"]:
        path = Path(args.out) / entry["id"] / "pack.json"
        if (args.only and entry["id"] not in args.only) or not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        entry = LAYOUTS[row.get("authoring", {}).get("layout", "courtyard")].retext(entry)
        row["character"] = {**_character(entry), "sheet": row["character"]["sheet"]}
        row["scene"] = {"medium": roster.get("medium", "A 3D video game"), "setting": entry["setting"],
                        "lighting": entry["lighting"]}
        path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"retexted {entry['id']}")


def prepare(args: argparse.Namespace) -> None:
    roster = json.loads(Path(args.roster).read_text(encoding="utf-8"))
    out = Path(args.out)
    shared = out / "_opening"
    shared.mkdir(parents=True, exist_ok=True)
    layout = LAYOUTS[args.layout]
    view = TrainingGround("h3-33b", layout=layout.id).opening(*OPENING_SIZE)
    Image.fromarray(view["style"]).save(shared / "opening.png")
    Image.fromarray(view["empty"]).save(shared / "opening_empty.png")
    np.save(shared / "ids.npy", view["ids"])
    box = view["player_box"]
    (shared / "opening.json").write_text(json.dumps({"size": OPENING_SIZE, "player_box": box}) + "\n")
    empty = Image.fromarray(view["empty"])
    for entry in roster["characters"]:
        if args.only and entry["id"] not in args.only:
            continue
        entry = layout.retext(entry)
        sheet = Path(args.sheets) / roster["sheet_pattern"].format(id=entry["id"])
        root = out / entry["id"]
        root.mkdir(parents=True, exist_ok=True)
        with Image.open(sheet) as raw:
            figure = authoring.cutout(raw)
        figure.save(root / "cutout.png")
        authoring.paste(empty, figure, box).save(root / "collage.png")
        prompt = authoring.restyle_prompt(entry["setting"], entry["lighting"], entry["look"], entry["props"])
        (root / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        scene = {"medium": roster.get("medium", "A 3D video game"), "setting": entry["setting"],
                 "lighting": entry["lighting"]}
        character = _character(entry)
        authoring.write_pack(root, entry["id"], scene, character, sheet, {
            "opening": "../_opening/opening.png", "collage": "collage.png", "prompt": "prompt.txt",
            "player_box": list(box), "opening_size": list(OPENING_SIZE), "layout": layout.id}, roster["game"])
        print(f"prepared {entry['id']}: collage + prompt in {root}", flush=True)


def review(args: argparse.Namespace) -> None:
    out = Path(args.out)
    proxy = Image.open(out / "_opening" / "opening.png")
    ids = np.load(out / "_opening" / "ids.npy")
    box = tuple(json.loads((out / "_opening" / "opening.json").read_text())["player_box"])
    for path in sorted(out.glob("*/pack.json")):
        pack = authoring.load_pack(path, require_first_frame=False)
        if pack.first_frame is None:
            print(f"{pack.id}: not painted yet")
            continue
        with Image.open(pack.first_frame) as painted, Image.open(pack.character["sheet"]) as sheet:
            authoring.review_panel(proxy, painted, ids, box, sheet).save(pack.root / "review.jpg", quality=90)
        print(f"{pack.id}: {pack.root / 'review.jpg'}")


def play(args: argparse.Namespace) -> None:
    out, runs = Path(args.out), Path(args.runs)
    packs = [authoring.load_pack(p) for p in sorted(out.glob("*/pack.json"))]
    packs = [p for p in packs if not args.only or p.id in args.only]
    script = SCRIPTS[args.script]
    seconds = script.seconds
    first = not args.style_only
    for pack in packs:
        if _layout(pack) != script.layout:
            raise SystemExit(f"{pack.id} was prepared for the {_layout(pack)} layout, "
                             f"script {args.script!r} plays in the {script.layout} layout")
        modalities = ("style", "depth", "semantic", "duv") if first else ("style", )
        game = TrainingGround("h3-33b", modalities, character=pack.character, scene=pack.scene,
                              first_frame=str(pack.first_frame), workers=args.workers, session_id=pack.id,
                              camera=args.camera, text_style=args.style, glow=args.glow, layout=script.layout)
        root = runs / (pack.id + args.tag)
        sinks = [DirectorySink(root / "stream"), game.take_sink(root / "take", "style", seed=args.seed)]
        if first:
            sinks.append(PreviewSink(runs / "proxy_preview.mp4"))
        pump(game.stream(script, seconds), *sinks)
        take = sinks[1]
        (root / "inputs.jsonl").write_text("".join(json.dumps(vars(i)) + "\n" for i in game.inputs))
        print(f"{pack.id}: {len(game.snapshots)} frames, {len(take.plan.segments)} segments -> {take.request_path}",
              flush=True)
        first = False


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


def _event_text(event: dict, skills: dict) -> str:
    mechanic, outcome, target = event["mechanic"], event.get("outcome", ""), event.get("target", "")
    if event["actor"] == "player":
        name = skills[mechanic]["name"] if mechanic in skills else mechanic.replace("_", " ")
        return {"cast": name, "start": name, "end": ""}.get(outcome, f"{name} {outcome} {target}".strip())
    return f"{event['actor']} " + {"knockdown": "knocked down", "respawn": "rises"}.get(mechanic, mechanic)


def overview(args: argparse.Namespace) -> None:
    """A labelled grid of the engine proxy and every character's render, with the engine's events."""
    import av
    from PIL import ImageDraw

    out, runs, renders = Path(args.out), Path(args.runs), Path(args.renders)
    packs = [authoring.load_pack(p) for p in sorted(out.glob("*/pack.json"))]
    first = runs / packs[0].id / "take"
    request = json.loads((first / "request.json").read_text())
    fps = GEOMETRY[request["model"]].fps
    events = request["take"]["events"]
    tiles = [("engine proxy", None, first / request["take"]["proxy"]["video"])]
    tiles += [(p.id.split("_", 1)[1].replace("_", " "), p.character["skills"], renders / p.id / "take.mp4")
              for p in packs]
    tiles = [t for t in tiles if t[2].is_file()]
    cols = 3
    rows = -(-len(tiles) // cols)
    tw, th = args.tile
    big, small = _font(max(12, th // 14)), _font(max(11, th // 18))
    readers = [av.open(str(t[2])).decode(video=0) for t in tiles]
    path = Path(args.video)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps, options={"crf": "18", "preset": "medium"})
        stream.pix_fmt, stream.width, stream.height = "yuv420p", cols * tw, rows * th
        last: list[Image.Image | None] = [None] * len(tiles)
        frame = 0
        while True:
            live = False
            for i, reader in enumerate(readers):
                decoded = next(reader, None)
                if decoded is not None:
                    live = True
                    last[i] = decoded.to_image().convert("RGB").resize((tw, th), Image.LANCZOS)
            if not live:
                break
            canvas = Image.new("RGB", (cols * tw, rows * th))
            t = frame / fps
            for i, (label, skills, _) in enumerate(tiles):
                tile = last[i].copy()
                draw = ImageDraw.Draw(tile)
                draw.text((10, 8), label, font=big, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                recent = [e for e in events if 0 <= t - e["t"] < 0.9]
                texts = [x for x in (_event_text(e, skills or {}) for e in recent) if x]
                for j, text in enumerate(texts[-2:]):
                    draw.text((10, th - 12 - (len(texts[-2:]) - j) * (small.size + 6)), text, font=small,
                              fill=(255, 220, 120), stroke_width=2, stroke_fill=(0, 0, 0))
                if i == 0:
                    clock = f"{t:5.2f}s"
                    draw.text((tw - 10 - draw.textlength(clock, font=small), 8), clock, font=small,
                              fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
                canvas.paste(tile, ((i % cols) * tw, (i // cols) * th))
            for packet in stream.encode(av.VideoFrame.from_image(canvas)):
                container.mux(packet)
            frame += 1
        for packet in stream.encode():
            container.mux(packet)
    print(f"{path}: {frame} frames, {len(tiles)} tiles")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamewm_engine.games.training_ground.author")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--roster", required=True)
    p.add_argument("--sheets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--layout", choices=sorted(LAYOUTS), default="courtyard")
    t = sub.add_parser("retext")
    t.add_argument("--roster", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--only", nargs="*")
    r = sub.add_parser("review")
    r.add_argument("--out", required=True)
    g = sub.add_parser("play")
    g.add_argument("--out", required=True)
    g.add_argument("--runs", required=True)
    g.add_argument("--only", nargs="*")
    g.add_argument("--workers", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--script", choices=sorted(SCRIPTS), default="demo")
    g.add_argument("--camera", choices=sorted(R.CAMERA_SHOTS), default="follow")
    g.add_argument("--style", choices=narrate.STYLES, default="chain")
    g.add_argument("--style-only", action="store_true", help="render only the style proxy, also for the first pack")
    g.add_argument("--glow", type=float, nargs=3, help="projectile colour in the proxy (RGB 0-1)")
    g.add_argument("--tag", default="", help="suffix for each run directory (variants of one character)")
    v = sub.add_parser("overview")
    v.add_argument("--out", required=True)
    v.add_argument("--runs", required=True)
    v.add_argument("--renders", required=True)
    v.add_argument("--video", required=True)
    v.add_argument("--tile", type=int, nargs=2, default=(640, 360))
    args = ap.parse_args(argv)
    {"prepare": prepare, "retext": retext, "review": review, "play": play, "overview": overview}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
