"""The engine's streaming output, ``gamewm.engine-stream.v1``.

A request (``protocol.py``) describes a finished take. A stream is the same information produced
while the game runs, cut where the VM's video VAE cuts time, so a causal VM can start a chunk as
soon as the engine has finished it:

``session``  once: model, frame rate, chunking, modalities, assets, skill tree and entities.
``chunk``    per VAE chunk: its proxy frames, the events resolved in it, per-frame state and camera,
             and the text for the VM's attention window ending at this chunk.
``end``      once.

Chunk 0 is the VAE's prefix (H3: 5 frames -> 2 latents; Wan: 1 frame -> 1 latent); every later
chunk is one temporal block (H3: 17 frames -> 5 latents). A chunk's ``text`` covers the window of
the last ``window_chunks`` chunks, which is what a causal student attends to when it denoises it.

Messages are plain JSON. Frames travel beside them: in process as numpy arrays on ``Chunk.frames``,
and on disk as ``.npy`` files named in the message (``DirectorySink``). ``TakeSink`` turns a
stream back into a v1 request plus proxy video, so the offline VM path stays the reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import GEOMETRY, SCHEMA, Event

STREAM_SCHEMA = "gamewm.engine-stream.v1"


@dataclass(frozen=True)
class Chunking:
    """Frames per VAE chunk for one backbone. ``latents`` is what each chunk becomes."""

    prefix: int
    size: int
    window: int
    prefix_latents: int
    latents: int

    def bounds(self, index: int) -> tuple[int, int]:
        if index == 0:
            return 0, self.prefix
        start = self.prefix + (index - 1) * self.size
        return start, start + self.size

    def index_of(self, frame: int) -> int:
        return 0 if frame < self.prefix else 1 + (frame - self.prefix) // self.size

    def window_start(self, index: int) -> int:
        return max(0, index - self.window + 1)

    def as_dict(self) -> dict[str, int]:
        return {"prefix_frames": self.prefix, "chunk_frames": self.size, "window_chunks": self.window,
                "prefix_latents": self.prefix_latents, "chunk_latents": self.latents}


CHUNKING = {
    "h3-33b": Chunking(prefix=5, size=17, window=6, prefix_latents=2, latents=5),
    "wan2.2-5b": Chunking(prefix=1, size=20, window=6, prefix_latents=1, latents=5),
    "wan2.2-14b": Chunking(prefix=1, size=12, window=6, prefix_latents=1, latents=3),
}


def event_dict(e: Event) -> dict[str, Any]:
    row: dict[str, Any] = {"t": e.t, "mechanic": e.mechanic, "actor": e.actor}
    if e.target is not None:
        row["target"] = e.target
    if e.outcome is not None:
        row["outcome"] = e.outcome
    if e.detail:
        row["detail"] = dict(e.detail)
    return row


@dataclass
class Chunk:
    """One ``chunk`` message. ``frames`` holds ``{modality: uint8 [n, H, W, 3]}``; ``snapshots`` is
    the engine's own per-frame state, kept for in-process sinks and never serialised."""

    meta: dict[str, Any]
    frames: dict[str, np.ndarray]
    snapshots: tuple[Any, ...] = field(default=(), repr=False)

    @property
    def index(self) -> int:
        return self.meta["index"]

    @property
    def span(self) -> tuple[int, int]:
        return tuple(self.meta["frames"])

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.meta["events"]


Message = "dict[str, Any] | Chunk"


class Sink:
    """Receives the stream in order: ``session(dict)``, ``chunk(Chunk)`` ..., ``end(dict)``."""

    def session(self, message: dict[str, Any]) -> None:
        pass

    def chunk(self, chunk: Chunk) -> None:
        pass

    def end(self, message: dict[str, Any]) -> None:
        pass


def pump(messages: Iterable[dict[str, Any] | Chunk], *sinks: Sink) -> list[dict[str, Any]]:
    """Feed a stream into sinks; returns the chunk metadata (for summaries)."""
    metas = []
    for message in messages:
        for sink in sinks:
            if isinstance(message, Chunk):
                sink.chunk(message)
            elif message["type"] == "session":
                sink.session(message)
            else:
                sink.end(message)
        if isinstance(message, Chunk):
            metas.append(message.meta)
    return metas


class DirectorySink(Sink):
    """``session.json``, ``chunks.jsonl`` (one line per chunk, then the ``end`` line) and
    ``frames/<modality>/<index>.npy``. A consumer can tail ``chunks.jsonl`` while the game runs."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._log = None

    def session(self, message: dict[str, Any]) -> None:
        (self.root / "session.json").write_text(json.dumps(message, indent=2) + "\n", encoding="utf-8")
        self._log = open(self.root / "chunks.jsonl", "w", encoding="utf-8")

    def chunk(self, chunk: Chunk) -> None:
        names = {}
        for modality, frames in chunk.frames.items():
            path = self.root / "frames" / modality / f"{chunk.index:06d}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, frames)
            names[modality] = str(path.relative_to(self.root))
        self._write({**chunk.meta, "proxy": names})

    def end(self, message: dict[str, Any]) -> None:
        self._write(message)
        self._log.close()

    def _write(self, row: dict[str, Any]) -> None:
        self._log.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._log.flush()


def read_directory(root: str | Path):
    """Replay a ``DirectorySink`` directory as a stream (frames loaded lazily per chunk)."""
    root = Path(root)
    yield json.loads((root / "session.json").read_text(encoding="utf-8"))
    for line in (root / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["type"] == "chunk":
            frames = {m: np.load(root / p) for m, p in row.pop("proxy").items()}
            yield Chunk(row, frames)
        else:
            yield row


def write_video(path: str | Path, frames: np.ndarray, fps: int, lossless: bool = False) -> None:
    """``frames`` uint8 [n, H, W, 3]. Code modalities (semantic, duv) must stay exact, so they go
    through lossless RGB H.264; style and depth use ordinary yuv420p."""
    import av  # optional dependency, only needed to write video

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n, height, width, _ = frames.shape
    with av.open(str(path), "w") as container:
        if lossless:
            stream = container.add_stream("libx264rgb", rate=fps, options={"crf": "0", "preset": "medium"})
            stream.pix_fmt = "rgb24"
        else:
            stream = container.add_stream("libx264", rate=fps, options={"crf": "16", "preset": "medium"})
            stream.pix_fmt = "yuv420p"
        stream.width, stream.height = width, height
        for frame in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


LOSSLESS = {"semantic": True, "duv": True, "style": False, "depth": False}


class TakeSink(Sink):
    """Collect a stream into a v1 request: ``proxy/<modality>.mp4``, ``request.json``.

    ``action_for(start, end)`` writes the action text for frames ``[start, end)``; the game
    supplies it, because the narration needs the engine's state, not just the chunk messages.
    The take is cut into the VM's segments (H3: 124 frames, stride 85), each with its own action.
    """

    def __init__(self, root: str | Path, action_for: Callable[[int, int], str], modality: str = "style",
                 first_frame: str | Path | None = None, steps: int = 30, seed: int = 0) -> None:
        self.root = Path(root)
        self.action_for = action_for
        self.modality = modality
        self.first_frame = first_frame
        self.steps, self.seed = steps, seed
        self.frames: list[np.ndarray] = []
        self.events: list[dict[str, Any]] = []
        self.request_path: Path | None = None

    def session(self, message: dict[str, Any]) -> None:
        if self.modality not in message["modalities"]:
            raise ValueError(f"the stream carries {message['modalities']}, not {self.modality!r}")
        self.message = message

    def chunk(self, chunk: Chunk) -> None:
        self.frames.append(chunk.frames[self.modality])
        if chunk.index == 0:
            self._opening = chunk.frames.get("style", chunk.frames[self.modality])[0]
        self.events += chunk.events

    def end(self, message: dict[str, Any]) -> None:
        from .compose import plan, segment_count
        from .protocol import load, parse

        s = self.message
        geometry = GEOMETRY[s["model"]]
        frames = np.concatenate(self.frames)
        self.root.mkdir(parents=True, exist_ok=True)
        video = self.root / "proxy" / f"{self.modality}.mp4"
        first = self.first_frame
        if first is None:
            from PIL import Image

            first = self.root / "first_frame_proxy.png"
            Image.fromarray(self._opening).save(first)
        payload = {
            "schema": SCHEMA,
            "model": s["model"],
            "assets": {"first_frame": str(Path(first).resolve() if self.first_frame else first.name),
                       "scene": s["assets"]["scene"], "character": s["assets"]["character"]},
            "take": {"proxy": {"video": str(video.relative_to(self.root)), "modality": self.modality,
                               "kind": s.get("proxy_kind", "")},
                     "seconds": None, "entities": s["entities"], "events": self.events, "action": None,
                     "camera": s.get("camera", "")},
            "generation": {"steps": self.steps, "seed": self.seed},
        }
        payload["assets"]["character"] = {k: v for k, v in payload["assets"]["character"].items() if v is not None}
        probe = parse({**payload, "take": {**payload["take"], "action": "stands still", "events": []}},
                      check_files=False)
        count, output = segment_count(len(frames), probe)
        frames = frames[:output]
        stride = geometry.stride or geometry.frames
        actions = [self.action_for(i * stride, min(i * stride + geometry.frames, output)) for i in range(count)]
        payload["take"]["seconds"] = round(output / geometry.fps, 6)
        payload["take"]["action"] = actions[0] if count == 1 else actions
        payload["take"]["events"] = [e for e in self.events if e["t"] * geometry.fps < output]
        write_video(video, frames, geometry.fps, LOSSLESS[self.modality])
        self.request_path = self.root / "request.json"
        self.request_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.plan = plan(load(self.request_path), output)
        (self.root / "plan.json").write_text(json.dumps(
            {"proxy_frames": self.plan.proxy_frames, "output_frames": self.plan.output_frames,
             "segments": [{"index": g.index, "frames": [g.start, g.end], "role": g.role, "caption": g.caption}
                          for g in self.plan.segments]}, indent=2) + "\n", encoding="utf-8")


class PreviewSink(Sink):
    """A side-by-side mp4 of the streamed modalities, with the chunk index and events burnt in."""

    def __init__(self, path: str | Path, scale: float = 1.0) -> None:
        self.path = Path(path)
        self.scale = scale
        self.tiles: list[np.ndarray] = []
        self.events: list[dict[str, Any]] = []

    def session(self, message: dict[str, Any]) -> None:
        self.fps = message["fps"]
        self.modalities = message["modalities"]

    def chunk(self, chunk: Chunk) -> None:
        try:
            from PIL import Image, ImageDraw
        except ImportError:  # pragma: no cover
            Image = None
        f0, _ = chunk.span
        self.events += chunk.events
        for i in range(len(next(iter(chunk.frames.values())))):
            row = np.concatenate([chunk.frames[m][i] for m in self.modalities], axis=1)
            if Image is not None:
                image = Image.fromarray(row)
                draw = ImageDraw.Draw(image)
                frame = f0 + i
                recent = [e for e in self.events if 0 <= frame - round(e["t"] * self.fps) < self.fps]
                label = f"chunk {chunk.index}  frame {frame}  t={frame / self.fps:5.2f}s"
                draw.rectangle((0, 0, 8 + 7 * len(label), 16), fill=(0, 0, 0))
                draw.text((4, 2), label, fill=(255, 255, 255))
                for j, e in enumerate(recent[-3:]):
                    text = f"{e['mechanic']} {e.get('target') or e['actor']} {e.get('outcome', '')}"
                    draw.rectangle((0, 18 + 16 * j, 8 + 7 * len(text), 34 + 16 * j), fill=(120, 40, 0))
                    draw.text((4, 20 + 16 * j), text, fill=(255, 230, 200))
                row = np.asarray(image)
            if self.scale != 1.0:
                h, w = row.shape[:2]
                row = np.asarray(Image.fromarray(row).resize((int(w * self.scale) // 2 * 2,
                                                              int(h * self.scale) // 2 * 2)))
            self.tiles.append(row)

    def end(self, message: dict[str, Any]) -> None:
        write_video(self.path, np.stack(self.tiles), self.fps)


def latents(frames: int, chunking: Chunking) -> int:
    """Latent frames the VAE makes from ``frames`` streamed frames (partial last chunk rounds up)."""
    if frames <= chunking.prefix:
        return chunking.prefix_latents
    per = chunking.size / chunking.latents
    return chunking.prefix_latents + math.ceil((frames - chunking.prefix) / per)
