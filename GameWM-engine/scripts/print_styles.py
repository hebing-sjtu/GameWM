"""Print the SHOWCASE caption of every text style for roster characters (tiny proxy, no GPU)."""

import json
import sys
import tempfile
from pathlib import Path

from gamewm_engine.games.training_ground import narrate
from gamewm_engine.games.training_ground.author import _character
from gamewm_engine.games.training_ground.game import TrainingGround
from gamewm_engine.games.training_ground.inputs import SHOWCASE
from gamewm_engine import load, plan
from gamewm_engine.stream import pump

roster = json.loads(Path("examples/training_ground/roster.json").read_text())
ids = sys.argv[1:] or ["01_fire_mage"]
camera = "push"
for entry in roster["characters"]:
    if entry["id"] not in ids:
        continue
    scene = {"medium": roster.get("medium", "A 3D video game"), "setting": entry["setting"],
             "lighting": entry["lighting"]}
    for style in narrate.STYLES:
        game = TrainingGround("h3-33b", ("style", ), size=(64, 36), supersample=1, character=_character(entry),
                              scene=scene, camera=camera, text_style=style)
        with tempfile.TemporaryDirectory() as tmp:
            take = game.take_sink(tmp)
            pump(game.stream(SHOWCASE, SHOWCASE.seconds), take)
            layout = plan(load(Path(tmp) / "request.json"))
        caption = layout.segments[0].caption
        print(f"=== {entry['id']} {style} ({len(caption.split())} words, {len(layout.segments)} seg)")
        print(caption + "\n")
    events = [(round(e.t, 2), e.mechanic, e.outcome, e.target) for e in game.events]
    print("events:", events)
    closes = [round(s.camera.get("close", 0), 2) for s in game.snapshots[::6]]
    print("close every 6 frames:", closes)
