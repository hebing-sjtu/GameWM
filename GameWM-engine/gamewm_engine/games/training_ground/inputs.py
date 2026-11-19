"""Controllers: a function from the world (before its next tick) to that tick's ``Input``.

A live engine plugs a gamepad in here; the demo and the tests use a ``Script`` (timed holds,
presses and "face that dummy" turns) or the ``Bot``. Inputs are logged by the game, so any run can
be replayed exactly.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from . import rules as R
from .layout import LAYOUTS
from .world import Input, World, _forward

Controller = Callable[[World], Input]
BUTTONS = ("melee", "ranged", "dash")


@dataclass(frozen=True)
class Hold:
    """Hold sticks from ``start`` to ``end`` seconds."""
    start: float
    end: float
    forward: float = 0.0
    strafe: float = 0.0
    turn: float = 0.0


@dataclass(frozen=True)
class Press:
    """Press a button for one tick at ``at`` seconds."""
    at: float
    button: str


@dataclass(frozen=True)
class Face:
    """Turn toward a dummy from ``start`` until facing it (or ``end``)."""
    start: float
    end: float
    target: str


@dataclass(frozen=True)
class Walk:
    """Walk toward a dummy or a layout mark from ``start`` to ``end``, turning first if it is far
    off to the side, and stop ``stop`` metres from it."""
    start: float
    end: float
    target: str
    stop: float = 0.4


def _bearing(world: World, x: float, z: float) -> float:
    """Signed yaw error (radians, > 0 means the target is to the right)."""
    want = math.atan2(x - world.x, z - world.z)
    return (want - world.yaw + math.pi) % (2.0 * math.pi) - math.pi


def _turn_toward(world: World, x: float, z: float) -> float:
    err = _bearing(world, x, z)
    per_tick = math.radians(R.TURN_SPEED_DEG) / world.fps
    return max(-1.0, min(1.0, err / per_tick)) if abs(err) > 1e-3 else 0.0


class Script:
    """Timed inputs for one layout (its ``Face`` targets are that layout's dummies)."""

    def __init__(self, items: Sequence[Hold | Press | Face | Walk], layout: str = "courtyard") -> None:
        arena = LAYOUTS[layout]
        for item in items:
            if isinstance(item, Press) and item.button not in BUTTONS:
                raise ValueError(f"unknown button {item.button!r}")
            if isinstance(item, (Face, Walk)) and item.target not in arena.dummies and item.target not in arena.marks:
                raise ValueError(f"{layout} has no dummy or mark {item.target!r}")
        self.items = tuple(items)
        self.layout = layout

    def __call__(self, world: World) -> Input:
        tick = world.tick + 1
        t = tick / world.fps
        forward = strafe = turn = 0.0
        buttons = dict.fromkeys(BUTTONS, False)
        for item in self.items:
            if isinstance(item, Hold) and item.start <= t < item.end:
                forward, strafe, turn = forward + item.forward, strafe + item.strafe, turn + item.turn
            elif isinstance(item, Face) and item.start <= t < item.end:
                turn += _turn_toward(world, *world.layout.point(item.target))
            elif isinstance(item, Walk) and item.start <= t < item.end:
                x, z = world.layout.point(item.target)
                if math.hypot(x - world.x, z - world.z) > item.stop:
                    turn += _turn_toward(world, x, z)
                    forward += 1.0 if abs(_bearing(world, x, z)) < math.radians(60.0) else 0.0
            elif isinstance(item, Press) and round(item.at * world.fps) == tick:
                buttons[item.button] = True
        return Input(forward, strafe, turn, **buttons)

    @property
    def seconds(self) -> float:
        return max(getattr(i, "end", getattr(i, "at", 0.0)) for i in self.items)


# A 12.25 s showcase (294 frames = three H3 segments): fireballs on the centre dummy, an ember dash
# to the right one and a flame-fist combo that knocks it over, then the centre dummy falls too.
DEMO = Script((
    Hold(0.6, 2.4, forward=1.0),
    Face(2.2, 2.5, "dummy_center"),
    Press(2.5, "ranged"),
    Press(3.8, "ranged"),
    Face(4.9, 5.4, "dummy_right"),
    Press(5.45, "dash"),
    Face(5.75, 5.85, "dummy_right"),
    Press(5.85, "melee"),
    Press(6.7, "melee"),
    Press(7.55, "melee"),
    Hold(8.3, 9.1, forward=-1.0),
    Face(8.3, 9.3, "dummy_center"),
    Press(9.35, "ranged"),
    Press(10.6, "ranged"),
    Hold(11.3, 12.25, strafe=-0.6),
))


# One H3 segment (124 frames, 5.17 s) with each skill once, for comparing text and camera variants:
# the demo's walk-in and first fireball, its turn, ember dash and first flame fist.
SHOWCASE = Script((
    Hold(0.3, 2.1, forward=1.0),
    Face(1.9, 2.2, "dummy_center"),
    Press(2.2, "ranged"),
    Face(3.1, 3.6, "dummy_right"),
    Press(3.65, "dash"),
    Face(3.95, 4.05, "dummy_right"),
    Press(4.05, "melee"),
    Hold(4.4, 124 / 24),
))

# A 29.96 s bout (719 frames = eight H3 segments) with a skill in every segment: the centre dummy is
# worn down with ranged shots between melee combos that knock the side dummies over, and the
# dummies get back up in time for a second round.
LONG = Script((
    Hold(0.3, 1.9, forward=1.0),
    Face(1.7, 2.1, "dummy_center"),
    Press(2.1, "ranged"),
    Face(3.0, 3.7, "dummy_left"),
    Press(3.8, "dash"),
    Face(4.1, 4.2, "dummy_left"),
    Press(4.2, "melee"),
    Press(5.05, "melee"),
    Hold(5.8, 6.6, forward=-1.0),
    Face(5.8, 6.8, "dummy_center"),
    Press(6.9, "ranged"),
    Hold(7.8, 8.6, strafe=0.7),
    Face(8.4, 9.0, "dummy_right"),
    Press(9.1, "ranged"),
    Hold(10.0, 11.0, forward=1.0),
    Face(10.0, 11.2, "dummy_right"),
    Press(11.3, "dash"),
    Face(11.6, 11.7, "dummy_right"),
    Press(11.7, "melee"),
    Press(12.55, "melee"),
    Hold(13.2, 14.0, forward=-1.0),
    Face(13.2, 14.2, "dummy_left"),
    Press(14.3, "ranged"),
    Face(15.0, 15.6, "dummy_center"),
    Press(15.7, "ranged"),
    Press(17.0, "ranged"),
    Face(17.8, 18.4, "dummy_right"),
    Press(18.5, "ranged"),
    Hold(19.4, 20.3, forward=1.0),
    Face(19.4, 20.5, "dummy_left"),
    Press(20.6, "dash"),
    Face(20.9, 21.0, "dummy_left"),
    Press(21.0, "melee"),
    Hold(21.8, 22.7, forward=-1.0),
    Face(21.8, 22.9, "dummy_center"),
    Press(23.1, "ranged"),
    Face(23.9, 24.5, "dummy_right"),
    Press(24.6, "ranged"),
    Hold(25.5, 26.4, forward=1.0),
    Face(25.5, 26.6, "dummy_right"),
    Press(26.6, "dash"),
    Face(26.9, 27.0, "dummy_right"),
    Press(27.0, "melee"),
    Press(27.85, "melee"),
    Hold(28.5, 29.3, forward=-1.0),
    Face(28.5, 29.5, "dummy_center"),
    Hold(29.5, 719 / 24),
))


# The same 29.96 s for the corridor: shots down the lane at the next dummy, a dash in, a combo that
# knocks it over, then on to the next one; the dummies behind get back up in view of the camera.
CORRIDOR_LONG = Script((
    Hold(0.3, 1.6, forward=1.0),
    Face(1.5, 2.0, "dummy_l1"),
    Press(2.0, "ranged"),
    Face(2.9, 3.5, "dummy_r1"),
    Press(3.5, "ranged"),
    Face(4.3, 4.9, "dummy_l1"),
    Press(4.9, "dash"),
    Face(5.2, 5.3, "dummy_l1"),
    Press(5.3, "melee"),
    Press(6.15, "melee"),
    Press(7.0, "melee"),
    Face(7.6, 8.3, "dummy_r1"),
    Hold(8.0, 8.9, forward=1.0),
    Face(8.9, 9.3, "dummy_r1"),
    Press(9.3, "dash"),
    Face(9.6, 9.7, "dummy_r1"),
    Press(9.7, "melee"),
    Press(10.55, "melee"),
    Press(11.4, "melee"),
    Face(12.0, 12.6, "dummy_l2"),
    Hold(12.2, 13.8, forward=1.0),
    Face(13.8, 14.3, "dummy_l2"),
    Press(14.3, "ranged"),
    Press(15.6, "ranged"),
    Hold(16.2, 16.7, forward=1.0),
    Face(16.6, 17.0, "dummy_l2"),
    Press(17.0, "dash"),
    Face(17.3, 17.4, "dummy_l2"),
    Press(17.4, "melee"),
    Press(18.25, "melee"),
    Face(19.0, 19.6, "dummy_r2"),
    Press(19.7, "ranged"),
    Hold(20.5, 21.4, forward=1.0),
    Face(20.5, 21.5, "dummy_r2"),
    Press(21.5, "dash"),
    Face(21.8, 21.9, "dummy_r2"),
    Press(21.9, "melee"),
    Press(22.75, "melee"),
    Press(23.6, "melee"),
    Face(24.3, 24.9, "dummy_l3"),
    Hold(24.5, 26.1, forward=1.0),
    Face(26.0, 26.3, "dummy_l3"),
    Press(26.3, "ranged"),
    Face(27.0, 27.3, "dummy_l3"),
    Press(27.3, "dash"),
    Face(27.6, 27.7, "dummy_l3"),
    Press(27.7, "melee"),
    Press(28.55, "melee"),
    Hold(29.2, 719 / 24),
), layout="corridor")


# The same 29.96 s across two yards: the first yard's dummies (one knocked over by a combo, one worn
# down from range), a walk through the gateway, then the second yard's.
ROOMS_LONG = Script((
    Hold(0.3, 1.5, forward=1.0),
    Face(1.3, 1.9, "dummy_a_left"),
    Press(1.9, "ranged"),
    Face(2.6, 3.2, "dummy_a_right"),
    Press(3.2, "ranged"),
    Face(3.9, 4.2, "dummy_a_right"),
    Press(4.2, "dash"),
    Face(4.5, 4.6, "dummy_a_right"),
    Press(4.6, "melee"),
    Press(5.45, "melee"),
    Press(6.3, "melee"),
    Face(6.9, 7.4, "dummy_a_back"),
    Press(7.4, "ranged"),
    Face(8.1, 8.4, "dummy_a_back"),
    Press(8.4, "dash"),
    Face(8.7, 8.8, "dummy_a_back"),
    Press(8.8, "melee"),
    Press(9.65, "melee"),
    Press(10.5, "melee"),
    Walk(11.1, 13.6, "gate_in", stop=0.2),
    Face(13.6, 14.1, "gate_out"),  # square up and let a slow camera settle behind before the dash
    Press(14.6, "dash"),
    Walk(15.0, 16.7, "gate_out"),
    Face(16.7, 17.2, "dummy_b_left"),
    Press(17.2, "ranged"),
    Walk(17.9, 19.2, "dummy_b_left", stop=3.0),
    Face(19.1, 19.3, "dummy_b_left"),
    Press(19.3, "dash"),
    Face(19.6, 19.7, "dummy_b_left"),
    Press(19.7, "melee"),
    Press(20.55, "melee"),
    Press(21.4, "melee"),
    Face(21.9, 22.7, "dummy_b_right"),
    Press(22.7, "ranged"),
    Walk(23.4, 24.0, "dummy_b_right", stop=4.3),
    Face(23.9, 24.0, "dummy_b_right"),
    Press(24.0, "dash"),
    Face(24.3, 24.4, "dummy_b_right"),
    Press(24.4, "melee"),
    Press(25.25, "melee"),
    Press(26.1, "melee"),
    Face(26.6, 27.4, "dummy_b_back"),
    Press(27.4, "ranged"),
    Press(28.7, "ranged"),
    Hold(29.4, 719 / 24),
), layout="rooms")


class Bot:
    """Goes after the nearest standing dummy: fireballs from afar, dashes in, then melee."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def __call__(self, world: World) -> Input:
        tick = world.tick + 1
        targets = [d for d in world.dummies if d.state == "up"]
        if not targets:
            return Input(turn=0.25)
        d = min(targets, key=lambda d: math.hypot(d.x - world.x, d.z - world.z))
        dist = math.hypot(d.x - world.x, d.z - world.z) - R.DUMMY_RADIUS
        err = abs(_bearing(world, d.x, d.z))
        turn = _turn_toward(world, d.x, d.z)
        busy = world.action is not None
        if busy or err > math.radians(12.0):
            return Input(forward=0.3 if err < math.radians(45.0) and dist > 1.0 else 0.0, turn=turn)
        if dist <= R.MELEE.reach - 0.15:
            return Input(turn=turn, melee=tick >= world.ready["melee_attack"])
        if dist > 5.0 and tick >= world.ready["ranged_attack"] and self.rng.random() < 0.6:
            return Input(turn=turn, ranged=True)
        if 2.5 < dist < 4.6 and tick >= world.ready["dash"] and self.rng.random() < 0.15:
            return Input(forward=1.0, turn=turn, dash=True)
        return Input(forward=1.0, turn=turn)


def facing(world: World, target: str) -> float:
    """Degrees between the player's facing and a dummy (for tests)."""
    x, z, _ = world.layout.dummies[target]
    fx, fz = _forward(world.yaw)
    dx, dz = x - world.x, z - world.z
    return math.degrees(math.acos(max(-1.0, min(1.0, (fx * dx + fz * dz) / math.hypot(dx, dz)))))
