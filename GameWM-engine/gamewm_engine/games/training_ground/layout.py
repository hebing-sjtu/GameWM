"""Where the walls, pillars, crates and dummies stand: one ``Layout`` per arena.

The rules (skills, hit points, camera shots) are the same everywhere; a layout only places things.
``courtyard`` is the original square yard; ``corridor`` is a long lane lined with dummies on both
sides; ``rooms`` is two yards joined by a wide gateway between two tall pillars. Every box is
(centre x, centre z, half x, half z, height). The follow camera stays inside ``bounds``, a union of
rectangles (x min, x max, z min, z max) that the player can see from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import rules as R

Box = tuple[float, float, float, float, float]
Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class Layout:
    id: str
    walls: tuple[Box, ...]
    pillars: tuple[Box, ...]
    crates: tuple[Box, ...]
    dummies: dict[str, tuple[float, float, str]]  # id -> (x, z, noun phrase used in captions)
    bounds: tuple[Rect, ...]
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)  # player x, z, yaw
    wall_phrase: str = "the courtyard wall"
    marks: dict[str, tuple[float, float]] = field(default_factory=dict)  # named spots a script can walk to
    camera_collide: bool = False  # pull the camera in front of a tall box between it and the player
    # roster text rewrites for this arena: (field, pattern, replacement, count); count 0 = all
    rewrites: tuple[tuple[str, str, str, int], ...] = field(default=())

    def blockers(self) -> list[tuple[float, float, float, float, float, str]]:
        """Boxes as (cx, cz, hx, hz, height, entity); pillars block like walls."""
        return ([(*box, "wall") for box in self.walls] + [(*box, "wall") for box in self.pillars]
                + [(*box, "crate") for box in self.crates])

    def clamp(self, x: float, z: float) -> tuple[float, float]:
        """The point of ``bounds`` nearest to (x, z)."""
        best = None
        for x0, x1, z0, z1 in self.bounds:
            px, pz = min(max(x, x0), x1), min(max(z, z0), z1)
            d = (px - x) ** 2 + (pz - z) ** 2
            if best is None or d < best[0]:
                best = (d, px, pz)
        return best[1], best[2]

    def sight(self, px: float, pz: float, cx: float, cz: float, margin: float = 0.3) -> float:
        """How far (0-1) along the line from the player at (px, pz) to the camera at (cx, cz) the
        camera can go before a wall or pillar taller than 2 m is closer than ``margin``."""
        dx, dz, best = cx - px, cz - pz, 1.0
        for bx, bz, hx, hz, h, _ in self.blockers():
            if h < 2.0:
                continue
            t0, t1 = 0.0, 1.0
            for o, d, lo, hi in ((px, dx, bx - hx - margin, bx + hx + margin), (pz, dz, bz - hz - margin, bz + hz + margin)):
                if abs(d) < 1e-9:
                    if not lo <= o <= hi:
                        t0, t1 = 1.0, 0.0
                    continue
                a, b = (lo - o) / d, (hi - o) / d
                t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 <= t1 and t0 > 1e-6:
                best = min(best, t0)
        return best

    def point(self, name: str) -> tuple[float, float]:
        """Where a dummy stands or a mark is."""
        if name in self.dummies:
            return self.dummies[name][:2]
        return self.marks[name]

    def entities(self) -> dict[str, str]:
        out = {key: phrase for key, (_, _, phrase) in self.dummies.items()}
        out.update({"wall": self.wall_phrase, "crate": "a stack of wooden crates"})
        return out

    def retext(self, entry: dict) -> dict:
        """A roster entry with its setting, props and skill texts rewritten for this arena."""

        def sub(text: str, key: str) -> str:
            for name, pattern, replacement, count in self.rewrites:
                if name == key:
                    text = re.sub(pattern, replacement, text, count=count)
            return text

        out = dict(entry)
        out["setting"] = sub(entry["setting"], "setting")
        out["props"] = sub(entry["props"], "props")
        if isinstance(entry.get("lighting"), str):
            out["lighting"] = sub(entry["lighting"], "lighting")
        skills = {}
        for mechanic, skill in entry["skills"].items():
            skill = dict(skill)
            for key in ("text", "again"):
                if isinstance(skill.get(key), str):
                    skill[key] = sub(skill[key], "skill")
            if skill.get("phases"):
                skill["phases"] = {k: sub(v, "skill") for k, v in skill["phases"].items()}
            skills[mechanic] = skill
        out["skills"] = skills
        return out


def _ring(x0: float, x1: float, z0: float, z1: float, height: float = R.WALL_HEIGHT) -> list[Box]:
    """Four walls whose inner faces are the rectangle's edges."""
    w = R.WALL_THICKNESS / 2
    cx, cz, hx, hz = (x0 + x1) / 2, (z0 + z1) / 2, (x1 - x0) / 2 + 2 * w, (z1 - z0) / 2 + 2 * w
    return [(cx, z1 + w, hx, w, height), (cx, z0 - w, hx, w, height),
            (x1 + w, cz, w, hz, height), (x0 - w, cz, w, hz, height)]


_A = R.ARENA_HALF - R.WALL_THICKNESS / 2
COURTYARD = Layout(
    id="courtyard",
    walls=tuple(_ring(-_A, _A, -_A, _A)),
    pillars=R.PILLARS,
    crates=R.CRATES,
    dummies=dict(R.DUMMIES),
    bounds=((-(R.ARENA_HALF - 0.4), R.ARENA_HALF - 0.4, -(R.ARENA_HALF - 0.4), R.ARENA_HALF - 0.4), ),
)

# A 6 m lane, 51 m long; the dummies stand 1.4 m off the centre line, so the player walks between them.
CORRIDOR = Layout(
    id="corridor",
    walls=tuple(_ring(-3.0, 3.0, -6.0, 45.0)),
    pillars=tuple((side * 2.85, z, 0.45, 0.45, 3.2) for z in (4.0, 14.0, 24.0, 34.0) for side in (-1.0, 1.0)),
    crates=((-2.3, 18.5, 0.5, 0.5, 1.0), (2.2, 30.5, 0.6, 0.6, 1.2), (-2.3, 40.0, 0.5, 0.5, 0.9)),
    dummies={
        "dummy_l1": (-1.4, 7.0, "the first training dummy on the left"),
        "dummy_r1": (1.4, 11.0, "the first training dummy on the right"),
        "dummy_l2": (-1.4, 17.0, "the second training dummy on the left"),
        "dummy_r2": (1.4, 22.0, "the second training dummy on the right"),
        "dummy_l3": (-1.4, 28.0, "the third training dummy on the left"),
        "dummy_r3": (1.4, 35.0, "the third training dummy on the right"),
    },
    bounds=((-2.0, 2.0, -5.6, 44.6), ),
    wall_phrase="the corridor wall",
    rewrites=(
        ("setting", r"^a ", "a long ", 1),
        ("setting", r"\b(courtyard|yard)\b", "corridor", 1),
        ("setting", r"\bthree\b", "six", 1),
        ("setting", r"on round stands", "on round stands along both sides", 1),
        ("props", r"the three grey dummies", "the grey dummies", 1),
        ("lighting", r"\b(courtyard|yard)\b", "corridor", 0),
        ("skill", r"across the (court)?yard", "down the corridor", 0),
    ),
)

# Yard A (16 x 18 m) and yard B (16 x 20 m), joined by a 4 m gateway between two tall pillars.
_DIVIDER = 10.3
ROOMS = Layout(
    id="rooms",
    walls=tuple(_ring(-8.0, 8.0, -8.0, 30.6)) + (
        (-5.3, _DIVIDER, 3.3, R.WALL_THICKNESS / 2, R.WALL_HEIGHT),
        (5.3, _DIVIDER, 3.3, R.WALL_THICKNESS / 2, R.WALL_HEIGHT),
    ),
    pillars=((-2.45, _DIVIDER, 0.45, 0.45, 3.4), (2.45, _DIVIDER, 0.45, 0.45, 3.4),
             (-7.55, 1.0, 0.45, 0.45, 3.2), (7.55, 20.5, 0.45, 0.45, 3.2)),
    crates=((6.4, -5.0, 0.6, 0.6, 1.1), (-6.5, 7.4, 0.6, 0.6, 1.2), (-6.4, 28.0, 0.7, 0.7, 1.3),
            (5.6, 28.4, 0.5, 0.5, 0.9)),
    dummies={
        "dummy_a_left": (-3.0, 4.5, "the left training dummy in the first courtyard"),
        "dummy_a_right": (3.0, 4.5, "the right training dummy in the first courtyard"),
        "dummy_a_back": (5.0, 8.0, "the far training dummy in the first courtyard"),
        "dummy_b_left": (-3.0, 21.0, "the left training dummy in the second courtyard"),
        "dummy_b_right": (3.0, 21.0, "the right training dummy in the second courtyard"),
        "dummy_b_back": (0.0, 25.0, "the far training dummy in the second courtyard"),
    },
    bounds=((-7.6, 7.6, -7.6, 9.6), (-1.4, 1.4, 9.6, 11.0), (-7.6, 7.6, 11.0, 30.2)),
    start=(0.0, -1.0, 0.0),
    marks={"gate_in": (0.0, 8.2), "gate_out": (0.0, 15.0)},
    camera_collide=True,
    rewrites=(
        ("setting", r"^a ", "two ", 1),
        ("setting", r"\b(courtyard|yard)\b", r"\1s joined by a wide gateway between two tall pillars", 1),
        ("setting", r"on round stands", "on round stands in each", 1),
        ("props", r"the three grey dummies", "the grey dummies", 1),
        ("lighting", r"\bthe (courtyard|yard)\b", r"both \1s", 0),
    ),
)

LAYOUTS = {layout.id: layout for layout in (COURTYARD, CORRIDOR, ROOMS)}
