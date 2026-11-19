"""Write the protocol's ``action`` text for a window of frames from the engine's own log.

Every clause comes from state the engine computed: locomotion from the player's input and speed,
attacks and their outcomes from resolved events. Skills appear only as placeholders
(``{skill:ranged_attack>dummy_left}``), so the caption names an attack only where the engine
resolved one, and the character's skill tree decides how it looks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ...protocol import Event
from . import rules as R
from .layout import COURTYARD
from .world import Snapshot, _forward

ENTITIES = COURTYARD.entities()
MIN_RUN_SECONDS = 0.4


def _label(s: Snapshot) -> str:
    loco = s.locomotion
    if loco["dashing"]:
        return "dash"
    if s.player["speed"] < 0.3:
        if abs(loco["turn"]) > 0.3:
            return "turn_right" if loco["turn"] > 0 else "turn_left"
        return "still"
    if loco["forward"] < -0.3:
        return "back"
    if abs(loco["strafe"]) > 0.3 and abs(loco["strafe"]) >= abs(loco["forward"]):
        return "strafe_right" if loco["strafe"] > 0 else "strafe_left"
    return "walk"


def _runs(snaps: Sequence[Snapshot], fps: int) -> list[tuple[str, int, int]]:
    runs: list[list] = []
    for s in snaps:
        label = _label(s)
        if runs and runs[-1][0] == label:
            runs[-1][2] = s.tick + 1
        else:
            runs.append([label, s.tick, s.tick + 1])
    short = MIN_RUN_SECONDS * fps
    merged: list[list] = []
    for run in runs:
        if merged and (run[2] - run[1] < short or merged[-1][0] == run[0]):
            merged[-1][2] = run[2]
        else:
            merged.append(run)
    return [tuple(run) for run in merged]


def _heading(s: Snapshot) -> str | None:
    fx, fz = _forward(s.player["yaw"])
    best = None
    for d in s.dummies:
        if d["state"] != "up":
            continue
        dx, dz = d["x"] - s.player["x"], d["z"] - s.player["z"]
        dist = math.hypot(dx, dz)
        if dist < 1.2:
            continue
        cos = (dx * fx + dz * fz) / dist
        if cos > math.cos(math.radians(25.0)) and (best is None or cos > best[0]):
            best = (cos, d["id"])
    return best[1] if best else None


IDLE_SECONDS = 0.6  # a pause this long with no attack under way gets the character's idle move


def _move(label: str, end: Snapshot) -> str | None:
    """The ``{move:...}`` placeholder for a locomotion run; walks and turns aim at the dummy ahead."""
    if label == "dash":
        return None
    if label in ("walk", "turn_left", "turn_right"):
        target = _heading(end)
        if target:
            return f"{{move:{'walk_to' if label == 'walk' else 'turn_to'}>{target}}}"
    return f"{{move:{label}}}"


STYLES = ("chain", "prose", "phased", "detailed", "timed", "spans", "coarse")
"""How a window's beats become text, from a single comma chain to one timed sentence per phase:

- ``chain``: one clause per beat joined by ", then " (the first style; each skill as its ``text``);
- ``prose``: one sentence per beat, the character by pronoun, the window's first push-in folded into
  its attack;
- ``phased``: like ``prose`` but the first use of a skill in the window is told phase by phase
  (windup, release, flight, impact, decay as separate sentences), repeats by their ``again`` phrase;
- ``detailed``: every use of a skill told with its windup (the first in full phase by phase, a
  repeat as "once more" + windup + its ``again`` form), a dash followed by its ``decay`` /
  ``recover`` phases when it has them;
- ``timed``: ``phased`` with each sentence anchored to its time in the window ("At 2.2 seconds, ");
- ``spans``: ``phased`` with each sentence under its own ``[a-b]`` window stamp;
- ``coarse``: only the skill names ("uses Fireball on ...") and plain locomotion.
"""
CAMERA_IN = "the camera pushes in close behind {him}"


@dataclass
class Beat:
    start: int
    end: int
    kind: str  # move, idle, skill, miss, respawn, camera
    text: str = ""
    mechanic: str = ""
    target: str | None = None
    cast: int | None = None  # ranged: the frame the projectile leaves the hand
    hit: int | None = None  # the frame the attack resolves
    down: bool = False
    camera_in: bool = False
    outcome: str = ""


def beats(snaps: Sequence[Snapshot], events: Sequence[Event], fps: int) -> list[Beat]:
    """What happens in the window, in order: locomotion runs, attacks with the frames of their
    phases, respawns, and the skill camera moving in and out."""
    if not snaps:
        return [Beat(0, 0, "idle", "{move:idle}")]
    first, last = snaps[0].tick, snaps[-1].tick
    by_tick = {s.tick: s for s in snaps}
    frame_of = lambda e: e.detail.get("frame", round(e.t * fps))  # noqa: E731
    window = [e for e in events if first <= frame_of(e) <= last]
    out: list[Beat] = []
    for label, start, end in _runs(snaps, fps):
        if label == "still":
            quiet = all(by_tick[t].player.get("action") is None for t in range(start, end))
            if quiet and end - start >= IDLE_SECONDS * fps:
                out.append(Beat(start, end, "idle", "{move:idle}"))
            continue
        text = _move(label, by_tick[min(end - 1, last)])
        if text:
            out.append(Beat(start, end, "move", text))
    downs = {(e.actor, frame_of(e)) for e in window if e.mechanic == "knockdown"}
    casts = sorted(frame_of(e) for e in events if e.mechanic == "ranged_attack" and e.outcome == "cast")
    windup_r, windup_m = R.ticks(R.RANGED.windup, fps), R.ticks(R.MELEE.windup, fps)
    for e in window:
        f = frame_of(e)
        if e.mechanic == "melee_attack":
            if e.outcome == "miss":
                out.append(Beat(f - windup_m, f, "miss", "throws a strike at the empty air"))
            else:
                out.append(Beat(f - windup_m, f + R.ticks(R.MELEE.recovery, fps), "skill", mechanic=e.mechanic,
                                target=e.target, hit=f, down=(e.target, f) in downs, outcome="hit"))
        elif e.mechanic == "ranged_attack" and e.outcome in ("hit", "miss") and e.target:
            cast = max([c for c in casts if c <= f], default=f)
            out.append(Beat(cast - windup_r, f, "skill", mechanic=e.mechanic, target=e.target, cast=cast, hit=f,
                            down=(e.target, f) in downs, outcome=e.outcome))
        elif e.mechanic == "ranged_attack" and e.outcome == "miss":
            out.append(Beat(f, f, "miss", "sends a shot flying off out of reach"))
        elif e.mechanic == "ranged_attack" and e.outcome == "cast":
            resolved = any(r.mechanic == "ranged_attack" and r.outcome in ("hit", "miss") and r.t >= e.t
                           for r in window)
            if not resolved:  # still in flight when the window ends; aimed where it is headed
                out.append(Beat(f - windup_r, f, "skill", mechanic=e.mechanic, target=e.target, cast=f))
        elif e.mechanic == "dash" and e.outcome == "start":
            out.append(Beat(f, f + R.ticks(R.DASH.duration, fps), "skill", mechanic="dash"))
        elif e.mechanic == "respawn":
            out.append(Beat(f, f, "respawn", f"watches {{target:{e.actor}}} swing back upright"))
    # a shot that keeps the follow framing (``steady``) has no push-in to tell
    close = [s.camera.get("close", 0.0) > 0.5 and s.camera.get("pushes", True) for s in snaps]
    for i in range(1, len(snaps)):
        if close[i] != close[i - 1]:
            out.append(Beat(snaps[i].tick, snaps[i].tick, "camera", "in" if close[i] else "out"))
    out.sort(key=lambda b: (b.start, b.kind != "move"))
    for b in out:  # the push-in lands a few frames after the windup starts: tell it with the attack
        if b.kind == "camera" and b.text == "in":
            attack = next((a for a in out if a.kind == "skill" and a.mechanic != "dash"
                           and abs(a.start - b.start) <= fps // 3), None)
            if attack is not None:
                attack.camera_in, b.kind = True, "merged"
    return [b for b in out if b.kind != "merged"]


def action(snaps: Sequence[Snapshot], events: Sequence[Event], fps: int, style: str = "chain",
           phases: dict[str, set[str]] | None = None) -> str:
    """The action text for the frames of ``snaps`` (consecutive ticks) and the events among them.

    Locomotion is written as ``{move:...}`` and attacks as ``{skill:...}`` placeholders, so the
    character's own moves and skill tree decide how a walk or a fireball reads. ``phases`` says which
    phases each skill has text for (fine styles fall back to the whole skill without them).
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    bs = beats(snaps, events, fps)
    first = snaps[0].tick if snaps else 0
    if style == "chain":
        return _chain(bs)
    fine = style in ("phased", "detailed", "timed", "spans")
    told: set[str] = set()
    parts: list[tuple[int, int, str]] = []  # (start, end, sentence without its subject or with it)
    for b in bs:
        parts += _sentences(b, style, fine, told, phases or {}, fps)
    parts = [p for i, p in enumerate(parts) if i == 0 or p[2] != parts[i - 1][2]]
    if not parts:
        return "{move:idle}"
    out = []
    for i, (start, end, text) in enumerate(parts):
        t0, t1 = (start - first) / fps, (end - first) / fps
        if style == "timed" and i > 0:
            text = "the" + text[3:] if text.startswith("The ") else text
            text = f"at {max(0.0, t0):.1f} seconds, " + text.replace("{He} ", "{he} ", 1).replace("As ", "as ", 1)
        elif style == "spans":
            text = f"[{max(0.0, t0):.2f}s-{max(t0, t1) + 0.01:.2f}s] " + text
        out.append(text)
    if style == "spans":
        return "trains alone in the courtyard. " + " ".join(s.rstrip(".") + "." for s in out)
    head = out[0]
    lead = "As " + CAMERA_IN + ", {he} "
    if head.startswith(lead):
        head = head[len(lead):].rstrip(".") + ", as " + CAMERA_IN
    elif head.startswith("{He} "):
        head = head[len("{He} "):]
    return " ".join([head.rstrip(".") + "."] + [s.rstrip(".") + "." for s in out[1:]])


def _chain(bs: list[Beat]) -> str:
    texts, previous = [], None
    for b in bs:
        if b.kind == "camera":
            text = CAMERA_IN if b.text == "in" else "the camera eases back out"
        elif b.kind == "skill":
            text = _placeholder(b) + _after(b)
            if b.camera_in:
                text = CAMERA_IN + " as {he} " + text
        else:
            text = b.text
        if text == previous and not text.startswith("{skill:"):
            continue
        texts.append(text)
        previous = text
    return ", then ".join(texts) if texts else "{move:idle}"


def _placeholder(b: Beat, phase: str = "") -> str:
    key = b.mechanic + (f".{phase}" if phase else "")
    return f"{{skill:{key}>{b.target}}}" if b.target else f"{{skill:{key}}}"


def _sentences(b: Beat, style: str, fine: bool, told: set[str], phases: dict[str, set[str]],
               fps: int) -> list[tuple[int, int, str]]:
    # the proxy shows every push-in and pull-back; the text names the first push-in of the window only
    lead = "As " + CAMERA_IN + ", {he} " if b.camera_in and "camera" not in told else "{He} "
    if b.camera_in:
        told.add("camera")
    if b.kind == "camera":
        return []
    if b.kind != "skill":
        return [(b.start, b.end, "{He} " + b.text)]
    if style == "coarse":
        name = _placeholder(b, "name")
        aim = f" on {{target:{b.target}}}" if b.target and b.hit is not None else ""
        return [(b.start, b.end, f"{lead}uses {name}{aim}{_after(b)}")]
    have = phases.get(b.mechanic, set())
    every = style == "detailed"
    repeat = b.mechanic in told
    told.add(b.mechanic)
    if every and repeat:
        lead += "once more "
    if b.mechanic == "dash":
        out = [(b.start, b.end, lead + _placeholder(b))]
        if every and "decay" in have:
            out.append((b.end, b.end + fps // 3, _placeholder(b, "decay")))
        if every and "recover" in have:
            out.append((b.end, b.end + fps // 2, "{He} " + _placeholder(b, "recover")))
        return out
    if fine and b.cast is not None and b.hit is None and "release" in have:  # lands after the window
        flight = ", and " + _placeholder(b, "travel") if "travel" in have else ""
        out = [] if (repeat and not every) or "windup" not in have else [(b.start, b.cast, lead + _placeholder(b, "windup"))]
        return out + [(b.cast, b.end, ("{He} " if out else lead) + _placeholder(b, "release") + flight)]
    if not fine or (repeat and not every) or "windup" not in have:
        return [(b.start, b.end, lead + _placeholder(b) + _after(b))]
    if repeat:  # the body's windup, then the skill's short ``again`` form for the rest
        return [(b.start, b.cast or b.hit or b.end, lead + _placeholder(b, "windup")),
                (b.cast or b.hit or b.end, b.end, "{He} " + _placeholder(b) + _after(b))]
    out = [(b.start, b.cast or b.hit or b.end, lead + _placeholder(b, "windup"))]
    if b.cast is not None and "release" in have:
        flight = ", and " + _placeholder(b, "travel") if "travel" in have else ""
        out.append((b.cast, b.hit or b.cast, "{He} " + _placeholder(b, "release") + flight))
    if b.hit is not None:
        out.append((b.hit, b.hit + fps // 3, _placeholder(b, "impact") + _after(b)))
        if "decay" in have:
            out.append((b.hit + fps // 6, b.hit + fps // 2, _placeholder(b, "decay")))
    if "recover" in have and b.mechanic == "melee_attack":
        out.append((max(b.hit or b.end, b.end - fps // 6), b.end, "{He} " + _placeholder(b, "recover")))
    return out


def _after(b: Beat) -> str:
    if b.outcome != "hit":
        return ""
    return ", knocking it to the ground" if b.down else ", rocking it back on its stand"
