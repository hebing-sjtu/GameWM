"""The GameWM-engine -> GameWM-VM request, version 1.

A request is what the engine hands the VM for one take: the assets authored once for the game
(first frame, character sheet, skill tree), the proxy video the engine rendered from the player's
camera, the events it resolved, and the text that describes the take. Everything here is plain
data; nothing imports torch, so an engine can build and validate requests without a GPU.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA = "gamewm.engine-request.v1"

MODALITIES = ("style", "depth", "semantic", "duv")
SUBJECTS = ("protagonist", "vehicle")
PLACEHOLDER = re.compile(r"\{(skill|target|move):([A-Za-z0-9_\-]+(?:\.[a-z_]+)?)(?:>([A-Za-z0-9_\-]+))?\}")
# Parts of a skill a fine-grained caption can name one at a time: ``{skill:ranged_attack.windup>x}``.
# ``name`` is the skill's name; a phase the character doesn't author reads as nothing (``impact``
# falls back to the whole ``text``). ``decay`` is how the effect clears off the target after it lands.
PHASES = ("windup", "release", "travel", "impact", "decay", "recover", "name")
PRONOUNS = {"he": ("he", "him", "his"), "she": ("she", "her", "her"), "it": ("it", "it", "its"),
            "they": ("they", "them", "their")}
# How a subject moves when its character gives no ``moves`` entry; ``{target}`` is the aimed entity.
MOVES = {
    "walk": "walks forward",
    "walk_to": "walks toward {target}",
    "back": "steps backward",
    "strafe_left": "sidesteps to the left",
    "strafe_right": "sidesteps to the right",
    "turn_left": "turns to the left",
    "turn_right": "turns to the right",
    "turn_to": "turns to face {target}",
    "idle": "stands still",
}


@dataclass(frozen=True)
class Geometry:
    """The clip one VM forward generates, and how long takes are cut into segments."""

    fps: int
    frames: int
    width: int
    height: int
    proxy_width: int
    proxy_height: int
    stride: int | None
    max_segments: int

    @property
    def seconds(self) -> float:
        return self.frames / self.fps


GEOMETRY = {
    "h3-33b": Geometry(fps=24, frames=124, width=1280, height=704, proxy_width=640, proxy_height=352, stride=85,
                       max_segments=16),
    "wan2.2-5b": Geometry(fps=24, frames=121, width=1280, height=704, proxy_width=640, proxy_height=352,
                          stride=None, max_segments=1),
    "wan2.2-14b": Geometry(fps=16, frames=81, width=832, height=480, proxy_width=416, proxy_height=240,
                           stride=None, max_segments=1),
}

# The identity line placed before the caption when a sheet is given; the VM was trained on these
# exact sentences (FastVideo-oc scripts/h3_proxy/prepare_data/oc_sheets.py).
SHEET_PROMPTS = {
    "protagonist":
    "<Picture 2> is the protagonist identity sheet (front / left profile / back). The character the "
    "camera follows is the <Picture 2> character in every frame, also while turning or entering and "
    "leaving the frame. Identity only: camera, pose, motion and framing come from <Video 1>.",
    "vehicle":
    "<Picture 2> is the identity sheet of the enclosed vehicle the player drives; the driver stays hidden "
    "inside. The followed vehicle is the <Picture 2> vehicle in every frame, also while turning or entering "
    "and leaving the frame. Identity only: camera, pose, motion and framing come from <Video 1>.",
}


class ProtocolError(ValueError):
    """The request breaks the engine -> VM contract."""


@dataclass(frozen=True)
class Skill:
    name: str
    text: str
    again: str = ""  # shorter text for a repeat of the skill within one caption
    phases: dict[str, str] = field(default_factory=dict)

    def phase(self, name: str) -> str:
        if name == "name":
            return self.name
        return self.phases.get(name) or (self.text if name == "impact" else "")


@dataclass(frozen=True)
class Character:
    look: str
    sheet: Path | None = None
    subject: str = "protagonist"
    skills: dict[str, Skill] = field(default_factory=dict)
    moves: dict[str, str] = field(default_factory=dict)
    pronoun: str = "they"

    def move(self, kind: str) -> str:
        return self.moves.get(kind) or MOVES[kind]


@dataclass(frozen=True)
class Proxy:
    video: Path
    modality: str
    kind: str = ""
    near_is_bright: bool = True


@dataclass(frozen=True)
class Event:
    t: float
    mechanic: str
    actor: str
    target: str | None = None
    outcome: str | None = None
    detail: dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class Request:
    model: str
    first_frame: Path
    medium: str
    setting: str
    lighting: str
    character: Character
    proxy: Proxy
    action: tuple[str, ...]
    camera: str = ""
    entities: dict[str, str] = field(default_factory=dict)
    events: tuple[Event, ...] = ()
    seconds: float | None = None
    steps: int = 30
    seed: int = 0
    source: Path | None = None

    @property
    def geometry(self) -> Geometry:
        return GEOMETRY[self.model]


def _need(row: dict[str, Any], key: str, where: str) -> Any:
    if key not in row or row[key] in (None, ""):
        raise ProtocolError(f"{where}.{key} is required")
    return row[key]


def _path(value: Any, base: Path, where: str, check: bool) -> Path:
    path = Path(str(value)).expanduser()
    path = path if path.is_absolute() else base / path
    if check and not path.is_file():
        raise ProtocolError(f"{where} does not exist: {path}")
    return path


def parse(payload: dict[str, Any], *, base: Path = Path("."), check_files: bool = True) -> Request:
    """Read a request dict; relative paths resolve against ``base``."""

    if payload.get("schema") != SCHEMA:
        raise ProtocolError(f"schema must be {SCHEMA!r}")
    model = str(payload.get("model") or "h3-33b")
    if model not in GEOMETRY:
        raise ProtocolError(f"model must be one of {sorted(GEOMETRY)}")

    assets = _need(payload, "assets", "request")
    scene = _need(assets, "scene", "assets")
    char = _need(assets, "character", "assets")
    subject = str(char.get("subject") or "protagonist")
    if subject not in SUBJECTS:
        raise ProtocolError(f"assets.character.subject must be one of {SUBJECTS}")
    skills = {
        str(mechanic): Skill(str(_need(row, "name", f"skills.{mechanic}")), str(_need(row, "text", f"skills.{mechanic}")),
                             str(row.get("again") or ""), {str(k): str(v) for k, v in (row.get("phases") or {}).items()})
        for mechanic, row in (char.get("skills") or {}).items()
    }
    for mechanic, skill in skills.items():
        unknown = sorted(set(skill.phases) - set(PHASES) - {"name"})
        if unknown:
            raise ProtocolError(f"skills.{mechanic}.phases has unknown phases {unknown}; known: {PHASES[:-1]}")
    pronoun = str(char.get("pronoun") or "they")
    if pronoun not in PRONOUNS:
        raise ProtocolError(f"assets.character.pronoun must be one of {sorted(PRONOUNS)}")
    moves = {str(k): str(v) for k, v in (char.get("moves") or {}).items()}
    unknown = sorted(set(moves) - set(MOVES))
    if unknown:
        raise ProtocolError(f"assets.character.moves has unknown kinds {unknown}; known: {sorted(MOVES)}")
    sheet = char.get("sheet")
    character = Character(
        look=str(_need(char, "look", "assets.character")),
        sheet=_path(sheet, base, "assets.character.sheet", check_files) if sheet else None,
        subject=subject,
        skills=skills,
        moves=moves,
        pronoun=pronoun,
    )

    take = _need(payload, "take", "request")
    proxy_row = _need(take, "proxy", "take")
    modality = str(_need(proxy_row, "modality", "take.proxy"))
    if modality not in MODALITIES:
        raise ProtocolError(f"take.proxy.modality must be one of {MODALITIES}")
    proxy = Proxy(
        video=_path(_need(proxy_row, "video", "take.proxy"), base, "take.proxy.video", check_files),
        modality=modality,
        kind=str(proxy_row.get("kind") or ""),
        near_is_bright=bool(proxy_row.get("near_is_bright", True)),
    )
    events = tuple(
        Event(t=float(_need(row, "t", f"take.events[{i}]")), mechanic=str(_need(row, "mechanic", f"take.events[{i}]")),
              actor=str(_need(row, "actor", f"take.events[{i}]")), target=row.get("target"), outcome=row.get("outcome"),
              detail=dict(row.get("detail") or {}))
        for i, row in enumerate(take.get("events") or ()))
    action = _need(take, "action", "take")
    action = (action, ) if isinstance(action, str) else tuple(str(text) for text in action)
    generation = payload.get("generation") or {}

    request = Request(
        model=model,
        first_frame=_path(_need(assets, "first_frame", "assets"), base, "assets.first_frame", check_files),
        medium=str(scene.get("medium") or "A 3D video game"),
        setting=str(_need(scene, "setting", "assets.scene")),
        lighting=str(_need(scene, "lighting", "assets.scene")),
        character=character,
        proxy=proxy,
        action=action,
        camera=str(take.get("camera") or ""),
        entities={str(k): str(v) for k, v in (take.get("entities") or {}).items()},
        events=tuple(sorted(events, key=lambda e: e.t)),
        seconds=float(take["seconds"]) if take.get("seconds") is not None else None,
        steps=int(generation.get("steps", 30)),
        seed=int(generation.get("seed", 0)),
    )
    _check(request)
    return request


def _check(request: Request) -> None:
    geometry = request.geometry
    for event in request.events:
        if event.t < 0:
            raise ProtocolError(f"event {event.mechanic} at t={event.t} is before the take starts")
        for who in (event.actor, event.target):
            if who and who not in request.entities and who != "player":
                raise ProtocolError(f"event participant {who!r} is not in take.entities")
    for text in request.action:
        for kind, key, aim in PLACEHOLDER.findall(text):
            mechanic, _, phase = key.partition(".")
            if kind == "skill" and mechanic not in request.character.skills:
                raise ProtocolError(f"action references {{skill:{key}}} but the character has no such skill")
            if kind == "skill" and phase and phase not in PHASES:
                raise ProtocolError(f"action references {{skill:{key}}}; known phases: {PHASES}")
            if kind != "skill" and phase:
                raise ProtocolError(f"only skills have phases: {{{kind}:{key}}}")
            if kind == "move" and key not in MOVES:
                raise ProtocolError(f"action references {{move:{key}}}; known moves: {sorted(MOVES)}")
            if kind == "move" and "{target}" in request.character.move(key) and not aim:
                raise ProtocolError(f"{{move:{key}}} needs a target: {{move:{key}><entity>}}")
            if kind == "target" and key not in request.entities:
                raise ProtocolError(f"action references {{target:{key}}} but take.entities has no {key!r}")
            if aim and aim not in request.entities:
                raise ProtocolError(f"action aims {{{kind}:{key}>{aim}}} at {aim!r}, which is not in take.entities")
    if request.seconds is not None and request.seconds <= 0:
        raise ProtocolError("take.seconds must be positive")
    if len(request.action) > geometry.max_segments:
        raise ProtocolError(f"{request.model} takes at most {geometry.max_segments} segments")
    if not 1 <= request.steps <= 100:
        raise ProtocolError("generation.steps must be in [1, 100]")


def load(path: str | Path, *, check_files: bool = True) -> Request:
    path = Path(path)
    request = parse(json.loads(path.read_text(encoding="utf-8")), base=path.parent, check_files=check_files)
    return replace(request, source=path)
