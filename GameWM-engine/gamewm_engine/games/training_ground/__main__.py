"""Run the training ground and write its stream.

    python -m gamewm_engine.games.training_ground --out runs/demo --modalities style,depth,semantic,duv \\
        --controller demo --take --preview --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ...stream import DirectorySink, pump
from .game import TrainingGround
from .inputs import DEMO, Bot


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gamewm_engine.games.training_ground")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="h3-33b")
    ap.add_argument("--modalities", default="style")
    ap.add_argument("--controller", choices=("demo", "bot"), default="demo")
    ap.add_argument("--seconds", type=float, default=None, help="default: the demo script's length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--take", action="store_true", help="also write a v1 request (take/request.json)")
    ap.add_argument("--take-modality", default=None)
    ap.add_argument("--preview", action="store_true", help="also write preview.mp4")
    args = ap.parse_args(argv)

    game = TrainingGround(args.model, tuple(args.modalities.split(",")), supersample=args.supersample,
                          workers=args.workers, session_id=args.out.name)
    controller = DEMO if args.controller == "demo" else Bot(args.seed)
    seconds = args.seconds if args.seconds is not None else DEMO.seconds
    sinks = [DirectorySink(args.out / "stream")]
    take = None
    if args.take:
        take = game.take_sink(args.out / "take", args.take_modality)
        sinks.append(take)
    if args.preview:
        from ...stream import PreviewSink

        sinks.append(PreviewSink(args.out / "preview.mp4"))
    began = time.time()
    metas = pump(game.stream(controller, seconds), *sinks)
    summary = {
        "frames": len(game.snapshots),
        "chunks": len(metas),
        "seconds_wall": round(time.time() - began, 1),
        "events": [f"{e.t:.2f}s {e.mechanic} {e.actor}->{e.target} {e.outcome}" for e in game.events],
        "chunk_text": [f"{m['index']} {m['frames']}: {m['text']['action']}" for m in metas],
    }
    if take is not None:
        summary["take"] = str(take.request_path)
        summary["segments"] = [s.caption for s in take.plan.segments]
    (args.out / "inputs.jsonl").write_text(
        "".join(json.dumps(vars(i)) + "\n" for i in game.inputs), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
