"""Every number the training ground's outcomes depend on.

Times are in seconds and become whole ticks at the stream's frame rate, so an event's frame is
exact and the same at 24 and 16 fps up to rounding. Distances are metres.
"""

from __future__ import annotations

from dataclasses import dataclass

# Courtyard: a square of paving inside four walls, the player starts at the origin facing +z.
ARENA_HALF = 12.0
WALL_HEIGHT = 2.4
WALL_THICKNESS = 0.6

PLAYER_RADIUS = 0.35
WALK_SPEED = 2.2
TURN_SPEED_DEG = 150.0
ACTION_MOVE_SCALE = 0.3  # movement while winding up or recovering from an attack

# Follow camera, framed like the capsule proxies the VM was trained on: full body, lower left of
# centre, high enough that what the player faces stays visible over the right shoulder.
CAMERA_DISTANCE = 6.0
CAMERA_HEIGHT = 2.5
CAMERA_SHOULDER = 0.6
CAMERA_LOOK_AHEAD = 6.0
CAMERA_LOOK_HEIGHT = 0.9
CAMERA_HFOV_DEG = 70.0


@dataclass(frozen=True)
class Shot:
    """Where the follow camera goes while a skill plays (``World(camera=...)``): ``zoom`` narrows the
    lens, ``push`` dollies in and swings ``orbit_deg`` round to the player's right so the target in
    front stays clear of the body; ``close`` eases from 0 to 1 when an attack starts and back after
    it resolves. ``yaw_tau`` / ``yaw_rate_deg`` (0 = uncapped) set how the camera turns after the
    player, ``follow_tau`` how its position trails the goal."""

    distance: float
    height: float
    shoulder: float
    look_height: float
    hfov_deg: float
    orbit_deg: float = 0.0
    look_ahead: float = CAMERA_LOOK_AHEAD
    yaw_tau: float = 0.3
    yaw_rate_deg: float = 0.0
    follow_tau: float = 0.12


CAMERA_SHOTS = {
    "follow": Shot(CAMERA_DISTANCE, CAMERA_HEIGHT, CAMERA_SHOULDER, CAMERA_LOOK_HEIGHT, CAMERA_HFOV_DEG),
    "zoom": Shot(CAMERA_DISTANCE, CAMERA_HEIGHT, CAMERA_SHOULDER, CAMERA_LOOK_HEIGHT, 42.0),
    "push": Shot(3.4, 2.2, 0.3, 1.1, 60.0, orbit_deg=35.0, look_ahead=2.5),
    # the follow framing with no skill shot, a slow capped pan and a soft trail: for footage to cut
    "steady": Shot(CAMERA_DISTANCE, CAMERA_HEIGHT, CAMERA_SHOULDER, CAMERA_LOOK_HEIGHT, CAMERA_HFOV_DEG,
                   yaw_tau=0.9, yaw_rate_deg=45.0, follow_tau=0.2),
}
LENS = ("distance", "height", "shoulder", "look_height", "hfov_deg", "orbit_deg", "look_ahead")


def frames_tighter(shot: Shot) -> bool:
    """Whether the skill shot changes the framing at all (``steady`` only changes how it trails)."""
    follow = CAMERA_SHOTS["follow"]
    return any(getattr(shot, name) != getattr(follow, name) for name in LENS)


SHOT_HOLD_SECONDS = 0.45  # stay close this long after the hit lands or the strike recovers
SHOT_IN_SECONDS = 0.12  # easing time constants
SHOT_OUT_SECONDS = 0.3

DUMMY_RADIUS = 0.32
DUMMY_HEIGHT = 1.8
DUMMY_HP = 100
DUMMY_DOWN_SECONDS = 3.0
DUMMY_FALL_SECONDS = 0.5
DUMMY_RISE_SECONDS = 0.75

# id -> (x, z, noun phrase used in captions)
DUMMIES = {
    "dummy_left": (-2.6, 7.0, "the left training dummy"),
    "dummy_center": (0.0, 9.5, "the center training dummy"),
    "dummy_right": (2.6, 7.0, "the right training dummy"),
}

# Axis-aligned props inside the courtyard: (center x, z, half x, half z, height)
CRATES = (
    (-8.5, 9.0, 0.6, 0.6, 1.2),
    (-7.4, 9.4, 0.5, 0.5, 0.9),
    (8.6, 4.0, 0.7, 0.7, 1.4),
    (8.5, -6.0, 0.6, 0.6, 1.0),
)
PILLARS = (
    (-6.0, 11.0, 0.45, 0.45, 3.2),
    (6.0, 11.0, 0.45, 0.45, 3.2),
    (-11.0, 0.0, 0.45, 0.45, 3.2),
    (11.0, 0.0, 0.45, 0.45, 3.2),
)


@dataclass(frozen=True)
class Melee:
    windup: float = 0.25  # the hit lands at the end of the windup
    recovery: float = 0.35
    cooldown: float = 0.8  # from the press
    reach: float = 1.3  # from the player's centre to the dummy's surface
    half_angle_deg: float = 50.0
    damage: int = 34


@dataclass(frozen=True)
class Ranged:
    windup: float = 0.4  # the projectile leaves the hand at the end of the windup
    recovery: float = 0.3
    cooldown: float = 1.2
    speed: float = 14.0
    radius: float = 0.15
    max_range: float = 25.0
    release_height: float = 1.4
    release_forward: float = 0.55
    damage: int = 25


@dataclass(frozen=True)
class Dash:
    duration: float = 0.25
    distance: float = 4.0
    cooldown: float = 2.0


MELEE, RANGED, DASH = Melee(), Ranged(), Dash()

# Mechanic names are the keys of a character's skill tree (protocol `assets.character.skills`).
MECHANICS = ("melee_attack", "ranged_attack", "dash")


def ticks(seconds: float, fps: int) -> int:
    return max(1, round(seconds * fps))
