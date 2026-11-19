"""Turn a request into the VM's per-segment inputs: windows, captions and text documents."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .protocol import PLACEHOLDER, PRONOUNS, SHEET_PROMPTS, Event, ProtocolError, Request

# Every training caption opens with this span, whatever the backbone's clip length; keep it verbatim.
SPAN = "[0.00s-5.17s]"
OPENING_ROLE, CONTINUATION_ROLE = "w0_omni", "wn_omni"


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    role: str
    caption: str
    document: str
    events: tuple[Event, ...]


@dataclass(frozen=True)
class Plan:
    request: Request
    proxy_frames: int
    output_frames: int
    segments: tuple[Segment, ...]


def frame_of(t: float, fps: int) -> int:
    """The frame an event time falls in; times written as ``frame / fps`` round back to ``frame``."""
    return math.floor(t * fps + 1e-3)


def segment_count(frames: int, request: Request) -> tuple[int, int]:
    """``(segments, output frames)`` for a proxy of ``frames`` frames at the VM's fps."""

    g = request.geometry
    if frames <= g.frames or g.stride is None:
        return 1, min(frames, g.frames)
    count = min(1 + math.ceil((frames - g.frames) / g.stride), g.max_segments)
    return count, min(frames, g.frames + g.stride * (count - 1))


def _subject(request: Request) -> str:
    noun = "The protagonist" if request.character.subject == "protagonist" else "The vehicle"
    return f"{noun} from the identity sheet" if request.character.sheet else noun


def _fill(text: str, request: Request, events: tuple[Event, ...], where: str) -> str:
    """Expand placeholders. A skill already used earlier in the same text takes its ``again`` form
    (or its full text followed by "again"), so a combo reads as one attack and its repeats."""
    used: set[str] = set()

    def swap(match) -> str:
        kind, key, aim = match.group(1), match.group(2), match.group(3)
        if kind == "target":
            return request.entities[key]
        if kind == "move":
            return request.character.move(key).replace("{target}", request.entities[aim] if aim else "")
        mechanic, _, phase = key.partition(".")
        hits = [e for e in events if e.mechanic == mechanic]
        if aim:
            hits = [e for e in hits if e.target == aim]
        if not hits:
            wanted = f"{mechanic} event at {aim}" if aim else f"{mechanic} event"
            raise ProtocolError(f"{where} uses {match.group(0)} but no {wanted} falls inside it")
        aimed = [e for e in hits if e.target in request.entities]
        target = request.entities[aimed[0].target] if aimed else "the target"
        skill = request.character.skills[mechanic]
        if phase:
            used.add(mechanic)
            return skill.phase(phase).replace("{target}", target)
        if mechanic not in used:
            used.add(mechanic)
            return skill.text.replace("{target}", target)
        return (skill.again or skill.text + " again").replace("{target}", target)

    filled = PLACEHOLDER.sub(swap, text)
    he, him, his = PRONOUNS[request.character.pronoun]
    for token, word in (("{he}", he), ("{him}", him), ("{his}", his)):
        filled = filled.replace(token, word).replace(token.title(), word.capitalize())
    filled = re.sub(r"\s+,", ",", re.sub(r"\s{2,}", " ", filled))
    return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), filled)


SENTENCE_START = re.compile(r"((?:[.!?]|\]) +)([a-z])")


def caption(request: Request, action: str, events: tuple[Event, ...], where: str = "segment 0") -> str:
    body = _fill(action, request, events, where).strip().rstrip(".")
    parts = [f"{request.medium} captures {request.setting}.", f"{request.lighting}.",
             f"{_subject(request)}, {request.character.look}, {body}."]
    if request.camera:
        parts.append(request.camera)
    return f"{SPAN} " + " ".join(parts)


def document(request: Request, text: str) -> str:
    """The text the VM encodes: modality tag, optional identity line, then the caption."""

    lines = [f"Reference modality: {request.proxy.modality} video."]
    if request.character.sheet:
        lines.append(SHEET_PROMPTS[request.character.subject])
    lines.append(text)
    return "\n".join(lines)


def plan(request: Request, proxy_frames: int | None = None) -> Plan:
    """Cut the take into VM segments; ``proxy_frames`` is the proxy's length at the VM's fps."""

    g = request.geometry
    if proxy_frames is None:
        if request.seconds is None:
            raise ProtocolError("give take.seconds or the proxy's frame count")
        proxy_frames = round(request.seconds * g.fps)
    elif request.seconds is not None:
        proxy_frames = min(proxy_frames, round(request.seconds * g.fps))
    if proxy_frames < 1:
        raise ProtocolError("the proxy has no frames")
    count, output = segment_count(proxy_frames, request)
    actions = request.action * count if len(request.action) == 1 else request.action
    if len(actions) != count:
        raise ProtocolError(f"the take makes {count} segments but take.action has {len(request.action)} entries")
    stride = g.stride or g.frames
    segments = []
    for index, action in enumerate(actions):
        start = index * stride
        end = start + g.frames
        events = tuple(e for e in request.events if start <= frame_of(e.t, g.fps) < end)
        text = caption(request, action, events, f"segment {index}")
        segments.append(Segment(index, start, end, OPENING_ROLE if index == 0 else CONTINUATION_ROLE, text,
                                document(request, text), events))
    return Plan(request, proxy_frames, output, tuple(segments))
