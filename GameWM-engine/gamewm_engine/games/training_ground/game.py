"""The training ground as a streaming engine.

``TrainingGround.feed(input)`` advances one frame and returns a ``Chunk`` whenever a VAE chunk is
complete; ``stream(controller, seconds)`` wraps that into the full ``session / chunk ... / end``
message sequence of ``gamewm.engine-stream.v1``.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ... import compose
from ...protocol import GEOMETRY, MODALITIES, Character, Event, Proxy, Request, Skill
from ...stream import CHUNKING, STREAM_SCHEMA, Chunk, TakeSink, event_dict
from . import narrate
from . import rules as R
from .inputs import Controller
from .render import HUMAN, c2w, cast, render
from .world import Input, Snapshot, World

SCENE = {
    "medium": "A 3D video game",
    "setting": ("a walled stone training courtyard with three wooden training dummies on round stands, "
                "stacked wooden crates and square stone pillars"),
    "lighting": "Soft overcast daylight lights the courtyard evenly",
}
# The fire mage of the paper's BD-Sheet sample; any skill tree with these three mechanics works.
CHARACTER = {
    "subject": "protagonist",
    "look": "a young dark-haired man in a dark red leather coat with a crimson hood",
    "pronoun": "he",
    "skills": {
        "melee_attack": {"name": "Flame Fist",
                         "text": ("draws back his flame-wrapped right fist and drives a burning punch into {target} "
                                  "in a shower of sparks"),
                         "again": "throws another flaming punch into {target}"},
        "ranged_attack": {"name": "Fireball",
                          "text": ("raises his right hand to shoulder height at {target}, a flame gathering in his "
                                   "palm, and hurls a fireball that bursts into flames on it"),
                          "again": "raises his hand again and hurls another fireball at {target}",
                          "phases": {
                              "windup": ("plants his feet and raises his right hand to shoulder height toward "
                                         "{target}, a flame flickering to life in his palm"),
                              "release": "thrusts his palm forward and a roaring fireball bursts from it",
                              "travel": "the fireball streaks across the courtyard trailing sparks and smoke",
                              "impact": "it explodes in a ball of flame against {target}"}},
        "dash": {"name": "Ember Dash", "text": "dashes forward low in a streak of embers that scorch the ground"},
    },
    "moves": {
        "walk_to": "walks toward {target} with an easy stride, arms swinging and his coat flapping",
        "back": "backs away with light, careful steps",
        "idle": "stands loose and ready, shifting his weight",
    },
}
CAMERA = "The camera follows from behind the protagonist."
DETAILED_WORDS = 350  # the action part; scene, subject and camera add about 70 more


def _r(value: float) -> float:
    return round(float(value), 4)


def state(s: Snapshot) -> dict[str, Any]:
    """The per-frame state a chunk message carries (pose, dummies, projectiles, camera, input)."""
    p = s.player
    return {
        "frame": s.tick,
        "player": {"x": _r(p["x"]), "z": _r(p["z"]), "yaw": _r(p["yaw"]), "speed": _r(p["speed"]),
                   "action": p["action"]},
        "dummies": [{"id": d["id"], "state": d["state"], "hp": d["hp"], "tilt": _r(d["tilt"])} for d in s.dummies],
        "projectiles": [[_r(v) for v in xyz] for xyz in s.projectiles],
        "camera": {"c2w": c2w(s.camera), "hfov_deg": s.camera["hfov_deg"]},
        "input": {k: (_r(v) if isinstance(v, float) else v) for k, v in s.locomotion.items()},
    }


class TrainingGround:

    def __init__(self, model: str = "h3-33b", modalities: tuple[str, ...] = ("style", ), *,
                 character: dict | None = None, scene: dict | None = None, first_frame: str | None = None,
                 supersample: int = 2, workers: int = 0, session_id: str | None = None,
                 size: tuple[int, int] | None = None, camera: str = "follow", text_style: str = "chain",
                 glow: tuple[float, float, float] | None = None, layout: str = "courtyard") -> None:
        unknown = [m for m in modalities if m not in MODALITIES]
        if unknown or not modalities:
            raise ValueError(f"modalities must be a non-empty subset of {MODALITIES}")
        self.model = model
        self.geometry = GEOMETRY[model]
        self.chunking = CHUNKING[model]
        self.fps = self.geometry.fps
        self.modalities = tuple(modalities)
        self.size = size or (self.geometry.proxy_width, self.geometry.proxy_height)
        self.character = character or CHARACTER
        self.scene = scene or SCENE
        self.first_frame = first_frame
        self.supersample = supersample
        self.session_id = session_id or uuid.uuid4().hex[:12]
        if text_style not in narrate.STYLES:
            raise ValueError(f"text_style must be one of {narrate.STYLES}")
        self.camera, self.text_style, self.glow = camera, text_style, glow
        self.world = World(self.fps, camera, layout)
        self.layout = layout
        self.entities = self.world.layout.entities()
        self.snapshots: list[Snapshot] = []
        self.events: list[Event] = []
        self.inputs: list[Input] = []
        self.index = 0
        self._pool = ProcessPoolExecutor(workers) if workers > 1 else None
        self._request = Request(
            model=model, first_frame=Path(first_frame or "first_frame.png"), medium=self.scene["medium"],
            setting=self.scene["setting"], lighting=self.scene["lighting"],
            character=Character(look=self.character["look"], subject=self.character.get("subject", "protagonist"),
                                sheet=Path(self.character["sheet"]) if self.character.get("sheet") else None,
                                skills={k: Skill(v["name"], v["text"], v.get("again", ""), dict(v.get("phases") or {}))
                                        for k, v in self.character["skills"].items()},
                                moves=dict(self.character.get("moves") or {}),
                                pronoun=self.character.get("pronoun", "they")),
            proxy=Proxy(Path("stream"), self.modalities[0], "capsule"), action=("stands still", ), camera=CAMERA,
            entities=dict(self.entities))
        missing = [m for m in R.MECHANICS if m not in self.character["skills"]]
        if missing:
            raise ValueError(f"the character's skill tree lacks {missing}")

    # -- messages -------------------------------------------------------------------------------

    def session(self) -> dict[str, Any]:
        g = self.geometry
        return {
            "schema": STREAM_SCHEMA,
            "type": "session",
            "session_id": self.session_id,
            "game": "training_ground",
            "model": self.model,
            "fps": self.fps,
            "proxy_size": list(self.size),
            "output_size": [g.width, g.height],
            "chunking": self.chunking.as_dict(),
            "modalities": list(self.modalities),
            "text_modality": self.modalities[0],
            "proxy_kind": "capsule",
            "assets": {"first_frame": self.first_frame, "scene": dict(self.scene), "character": dict(self.character)},
            "entities": dict(self.entities),
            "layout": self.layout,
            "camera": CAMERA,
            "camera_shot": self.camera,
            "text_style": self.text_style,
            "rules": {"melee_attack": asdict(R.MELEE), "ranged_attack": asdict(R.RANGED), "dash": asdict(R.DASH),
                      "dummy_hp": R.DUMMY_HP},
        }

    def feed(self, inp: Input) -> Chunk | None:
        """Advance one frame with ``inp``; returns the chunk this frame completes, if any."""
        self.inputs.append(inp)
        self.events += self.world.step(inp)
        self.snapshots.append(self.world.snapshot())
        _, end = self.chunking.bounds(self.index)
        return self._emit(end) if len(self.snapshots) >= end else None

    def flush(self) -> Chunk | None:
        """Emit the trailing partial chunk (a live session that stops between chunk boundaries)."""
        start, _ = self.chunking.bounds(self.index)
        return self._emit(len(self.snapshots)) if len(self.snapshots) > start else None

    def end(self, reason: str = "stopped") -> dict[str, Any]:
        if self._pool is not None:
            self._pool.shutdown()
        return {"schema": STREAM_SCHEMA, "type": "end", "session_id": self.session_id, "chunks": self.index,
                "frames": len(self.snapshots), "reason": reason}

    def stream(self, controller: Controller, seconds: float | None = None, frames: int | None = None,
               ) -> Iterator[dict[str, Any] | Chunk]:
        total = frames if frames is not None else round((seconds or 0.0) * self.fps)
        yield self.session()
        for _ in range(total):
            chunk = self.feed(controller(self.world))
            if chunk is not None:
                yield chunk
        tail = self.flush()
        if tail is not None:
            yield tail
        yield self.end("finished")

    # -- text -----------------------------------------------------------------------------------

    def action_for(self, start: int, end: int) -> str:
        """The action text for frames ``[start, end)``. A ``detailed`` window that would read longer
        than ``DETAILED_WORDS`` once its placeholders are filled is told ``phased`` instead."""
        phases = {k: set(v.get("phases") or ()) for k, v in self.character["skills"].items()}
        text = narrate.action(self.snapshots[start:end], self.events, self.fps, self.text_style, phases)
        if self.text_style == "detailed":
            filled = compose._fill(text, self._request, self._window_events(start, end), f"frames {start}-{end}")
            if len(filled.split()) > DETAILED_WORDS:
                text = narrate.action(self.snapshots[start:end], self.events, self.fps, "phased", phases)
        return text

    def _window_events(self, start: int, end: int) -> tuple[Event, ...]:
        return tuple(e for e in self.events if start <= e.detail["frame"] < end)

    def _emit(self, end: int) -> Chunk:
        k = self.index
        start, _ = self.chunking.bounds(k)
        snaps = self.snapshots[start:end]
        frames = self._render(snaps)
        w_start, _ = self.chunking.bounds(self.chunking.window_start(k))
        window_action = self.action_for(w_start, end)
        caption = compose.caption(self._request, window_action, self._window_events(w_start, end), f"chunk {k}")
        meta = {
            "schema": STREAM_SCHEMA,
            "type": "chunk",
            "session_id": self.session_id,
            "index": k,
            "frames": [start, end],
            "t": [round(start / self.fps, 6), round(end / self.fps, 6)],
            "latents": self.chunking.prefix_latents if k == 0 else self.chunking.latents,
            "events": [event_dict(e) for e in self._window_events(start, end)],
            "states": [state(s) for s in snaps],
            "text": {
                "action": self.action_for(start, end),
                "window": [self.chunking.window_start(k), k],
                "window_frames": [w_start, end],
                "window_action": window_action,
                "caption": caption,
                "document": compose.document(self._request, caption),
            },
        }
        self.index += 1
        return Chunk(meta, frames, tuple(snaps))

    # -- frames ---------------------------------------------------------------------------------

    def _render(self, snaps: list[Snapshot]) -> dict[str, np.ndarray]:
        skills = self.character["skills"]
        poses = {k: v["pose"] for k, v in skills.items() if v.get("pose")}
        flying = skills["ranged_attack"].get("kind", "projectile") != "at_target"
        job = partial(render, width=self.size[0], height=self.size[1], modalities=self.modalities,
                      supersample=self.supersample, glow=self.glow, poses=poses, projectiles=flying)
        outs = list(self._pool.map(job, snaps)) if self._pool is not None else [job(s) for s in snaps]
        return {m: np.stack([o[m] for o in outs]) for m in self.modalities}

    def take_sink(self, root: str | Path, modality: str | None = None, **kwargs) -> TakeSink:
        """A sink that writes this session as a v1 request (narrated per VM segment)."""
        return TakeSink(root, self.action_for, modality or self.modalities[0], first_frame=self.first_frame,
                        **kwargs)

    def opening(self, width: int = 1344, height: int = 768, supersample: int = 2) -> dict[str, Any]:
        """Frame 0 as the painter needs it: the grey-box view, the same view without the player,
        class ids, and the player's box. Same camera as the stream (hfov fixed), so a 16:9 painting
        centre-cropped to the VM's aspect lines up with the proxy."""
        world = World(self.fps, layout=self.layout)
        world.step(Input())
        snap = world.snapshot()
        s = max(1, supersample)
        full, empty = cast(snap, width * s, height * s), cast(snap, width * s, height * s, player=False)

        def down(image: np.ndarray) -> np.ndarray:
            return image.reshape(height, s, width, s, 3).mean(axis=(1, 3)).round().astype(np.uint8)

        ids = full.ids[s // 2::s, s // 2::s]
        ys, xs = np.nonzero(ids == HUMAN)
        return {"style": down(full.image), "empty": down(empty.image), "ids": ids,
                "player_box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)}

    def replay(self) -> TrainingGround:
        """A fresh game that reproduces this one frame for frame when fed ``self.inputs``."""
        return TrainingGround(self.model, self.modalities, character=self.character, scene=self.scene,
                              first_frame=self.first_frame, supersample=self.supersample, size=self.size,
                              camera=self.camera, text_style=self.text_style, glow=self.glow, layout=self.layout)
