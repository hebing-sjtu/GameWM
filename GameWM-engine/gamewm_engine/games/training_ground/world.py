"""The training ground's state and its fixed-step update.

One tick is one output frame. ``World.step(input)`` advances the world by one tick and returns the
events it resolved there; ``World.snapshot()`` is everything the renderer and the narrator need for
that frame. The update is deterministic: the same inputs give the same frames and events.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ...protocol import Event
from . import rules as R
from .layout import LAYOUTS, Layout


@dataclass(frozen=True)
class Input:
    """One tick of player input. ``turn`` > 0 turns right; ``strafe`` > 0 steps right."""

    forward: float = 0.0
    strafe: float = 0.0
    turn: float = 0.0
    melee: bool = False
    ranged: bool = False
    dash: bool = False


@dataclass
class Action:
    kind: str  # melee_attack | ranged_attack | dash
    start: int
    windup: int
    end: int
    direction: tuple[float, float] = (0.0, 1.0)


@dataclass
class Dummy:
    id: str
    x: float
    z: float
    hp: int = R.DUMMY_HP
    state: str = "up"  # up | falling | down | rising
    since: int = 0
    fall_dir: tuple[float, float] = (0.0, 1.0)
    impulses: list[tuple[int, float, float]] = field(default_factory=list)  # (tick, dir x, dir z)


@dataclass
class Projectile:
    x: float
    y: float
    z: float
    dx: float
    dz: float
    travelled: float = 0.0


@dataclass(frozen=True)
class Snapshot:
    tick: int
    t: float
    player: dict
    dummies: tuple[dict, ...]
    projectiles: tuple[tuple[float, float, float], ...]
    camera: dict
    locomotion: dict
    layout: str = "courtyard"


def _forward(yaw: float) -> tuple[float, float]:
    return math.sin(yaw), math.cos(yaw)


def _right(yaw: float) -> tuple[float, float]:
    return math.cos(yaw), -math.sin(yaw)


def _smooth(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class World:

    def __init__(self, fps: int = 24, camera: str = "follow", layout: str = "courtyard") -> None:
        if camera not in R.CAMERA_SHOTS:
            raise ValueError(f"camera must be one of {sorted(R.CAMERA_SHOTS)}")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {sorted(LAYOUTS)}")
        self.layout: Layout = LAYOUTS[layout]
        self.fps = fps
        self.shot = R.CAMERA_SHOTS[camera]
        self.close = 0.0
        self.focus_until = -1
        self.dt = 1.0 / fps
        self.tick = -1
        self.x, self.z, self.yaw = self.layout.start
        self.phase = 0.0
        self.speed = 0.0
        self.action: Action | None = None
        self.ready = {name: 0 for name in R.MECHANICS}
        self.dummies = [Dummy(key, x, z) for key, (x, z, _) in self.layout.dummies.items()]
        self._blockers = self.layout.blockers()
        self.projectiles: list[Projectile] = []
        self.last_input = Input()
        self.cam_yaw = self.yaw
        self.cam_x, self.cam_z = self._camera_goal()

    # -- geometry -------------------------------------------------------------------------------

    def blockers(self) -> list[tuple[float, float, float, float, float, str]]:
        """The layout's boxes as (cx, cz, hx, hz, height, entity)."""
        return self._blockers

    def _collide(self, x: float, z: float) -> tuple[float, float]:
        r = R.PLAYER_RADIUS
        for cx, cz, hx, hz, _, _ in self.blockers():
            px, pz = min(max(x, cx - hx), cx + hx), min(max(z, cz - hz), cz + hz)
            dx, dz = x - px, z - pz
            d = math.hypot(dx, dz)
            if d < r:
                if d < 1e-9:
                    dx, dz, d = 0.0, -1.0, 1.0
                x, z = px + dx / d * r, pz + dz / d * r
        for dummy in self.dummies:
            if dummy.state != "up":
                continue
            dx, dz = x - dummy.x, z - dummy.z
            d, reach = math.hypot(dx, dz), r + R.DUMMY_RADIUS
            if d < reach:
                if d < 1e-9:
                    dx, dz, d = 0.0, -1.0, 1.0
                x, z = dummy.x + dx / d * reach, dummy.z + dz / d * reach
        return x, z

    # -- update ---------------------------------------------------------------------------------

    def step(self, inp: Input) -> list[Event]:
        self.tick += 1
        k, t = self.tick, self.tick / self.fps
        events: list[Event] = []
        self.last_input = inp
        self._dummies(events)
        self._actions(inp, events)
        self._move(inp)
        self._projectiles(events)
        self._camera()
        return events

    def _event(self, events: list[Event], mechanic: str, actor: str, target: str | None = None,
               outcome: str | None = None, **detail) -> None:
        events.append(Event(round(self.tick / self.fps, 6), mechanic, actor, target, outcome,
                            {"frame": self.tick, **detail}))

    def _dummies(self, events: list[Event]) -> None:
        fall, down, rise = (R.ticks(s, self.fps) for s in (R.DUMMY_FALL_SECONDS, R.DUMMY_DOWN_SECONDS,
                                                             R.DUMMY_RISE_SECONDS))
        for d in self.dummies:
            age = self.tick - d.since
            if d.state == "falling" and age >= fall:
                d.state, d.since = "down", self.tick
            elif d.state == "down" and age >= down:
                d.state, d.since = "rising", self.tick
                self._event(events, "respawn", d.id, outcome="rise")
            elif d.state == "rising" and age >= rise:
                d.state, d.since, d.hp, d.impulses = "up", self.tick, R.DUMMY_HP, []

    def _damage(self, d: Dummy, amount: int, direction: tuple[float, float], events: list[Event]) -> None:
        d.hp = max(0, d.hp - amount)
        d.impulses.append((self.tick, *direction))
        if d.hp == 0:
            d.state, d.since, d.fall_dir = "falling", self.tick, direction
            self._event(events, "knockdown", d.id, outcome="down")

    def _actions(self, inp: Input, events: list[Event]) -> None:
        k = self.tick
        a = self.action
        if a is not None and k >= a.end:
            if a.kind == "dash":
                self._event(events, "dash", "player", outcome="end")
            self.action = a = None
        if a is None:
            for kind, pressed in (("dash", inp.dash), ("melee_attack", inp.melee), ("ranged_attack", inp.ranged)):
                if pressed and k >= self.ready[kind]:
                    self.action = a = self._start(kind, inp, events)
                    break
        if a is None:
            return
        if a.kind == "melee_attack" and k == a.windup:
            self._melee(events)
        elif a.kind == "ranged_attack" and k == a.windup:
            fx, fz = _forward(self.yaw)
            x0, z0 = self.x + fx * R.RANGED.release_forward, self.z + fz * R.RANGED.release_forward
            self.projectiles.append(Projectile(x0, R.RANGED.release_height, z0, fx, fz))
            self._event(events, "ranged_attack", "player", self._in_line(x0, z0, fx, fz), "cast")

    def _in_line(self, x: float, z: float, dx: float, dz: float) -> str | None:
        """The standing dummy a shot from (x, z) along (dx, dz) is headed for, if any."""
        best = None
        for d in self.dummies:
            along = (d.x - x) * dx + (d.z - z) * dz
            off = abs((d.x - x) * dz - (d.z - z) * dx)
            if d.state == "up" and 0.0 < along <= R.RANGED.max_range and off < R.DUMMY_RADIUS + R.RANGED.radius:
                if best is None or along < best[0]:
                    best = (along, d.id)
        return best[1] if best else None

    def _start(self, kind: str, inp: Input, events: list[Event]) -> Action:
        k, fps = self.tick, self.fps
        if kind == "melee_attack":
            m = R.MELEE
            self.ready[kind] = k + R.ticks(m.cooldown, fps)
            hit = k + R.ticks(m.windup, fps)
            return Action(kind, k, hit, hit + R.ticks(m.recovery, fps))
        if kind == "ranged_attack":
            m = R.RANGED
            self.ready[kind] = k + R.ticks(m.cooldown, fps)
            release = k + R.ticks(m.windup, fps)
            return Action(kind, k, release, release + R.ticks(m.recovery, fps))
        self.ready[kind] = k + R.ticks(R.DASH.cooldown, fps)
        fx, fz = _forward(self.yaw)
        rx, rz = _right(self.yaw)
        dx, dz = fx * inp.forward + rx * inp.strafe, fz * inp.forward + rz * inp.strafe
        norm = math.hypot(dx, dz)
        direction = (dx / norm, dz / norm) if norm > 1e-6 else (fx, fz)
        self._event(events, "dash", "player", outcome="start")
        return Action(kind, k, k, k + R.ticks(R.DASH.duration, fps), direction)

    def _melee(self, events: list[Event]) -> None:
        m = R.MELEE
        fx, fz = _forward(self.yaw)
        best = None
        for d in self.dummies:
            if d.state != "up":
                continue
            dx, dz = d.x - self.x, d.z - self.z
            dist = math.hypot(dx, dz)
            if dist - R.DUMMY_RADIUS > m.reach or dist < 1e-9:
                continue
            if (dx * fx + dz * fz) / dist < math.cos(math.radians(m.half_angle_deg)):
                continue
            if best is None or dist < best[0]:
                best = (dist, d, (dx / dist, dz / dist))
        if best is None:
            self._event(events, "melee_attack", "player", outcome="miss")
            return
        _, d, direction = best
        self._event(events, "melee_attack", "player", d.id, "hit", damage=m.damage, hp=max(0, d.hp - m.damage))
        self._damage(d, m.damage, direction, events)

    def _move(self, inp: Input) -> None:
        a = self.action
        x0, z0 = self.x, self.z
        if a is not None and a.kind == "dash":
            step = R.DASH.distance / max(1, a.end - a.start)
            self.x, self.z = self._collide(self.x + a.direction[0] * step, self.z + a.direction[1] * step)
        else:
            scale = R.ACTION_MOVE_SCALE if a is not None else 1.0
            self.yaw += math.radians(R.TURN_SPEED_DEG) * max(-1.0, min(1.0, inp.turn)) * self.dt
            fx, fz = _forward(self.yaw)
            rx, rz = _right(self.yaw)
            f, s = max(-1.0, min(1.0, inp.forward)), max(-1.0, min(1.0, inp.strafe))
            norm = max(1.0, math.hypot(f, s))
            v = R.WALK_SPEED * scale * self.dt / norm
            self.x, self.z = self._collide(self.x + (fx * f + rx * s) * v, self.z + (fz * f + rz * s) * v)
        moved = math.hypot(self.x - x0, self.z - z0)
        self.speed = moved / self.dt
        self.phase = (self.phase + moved / 1.3 * 2.0 * math.pi) % (2.0 * math.pi)

    def _projectiles(self, events: list[Event]) -> None:
        m = R.RANGED
        alive = []
        substeps = 4
        for p in self.projectiles:
            hit = None
            for _ in range(substeps):
                step = m.speed * self.dt / substeps
                p.x, p.z, p.travelled = p.x + p.dx * step, p.z + p.dz * step, p.travelled + step
                for d in self.dummies:
                    if d.state == "up" and math.hypot(p.x - d.x, p.z - d.z) < R.DUMMY_RADIUS + m.radius \
                            and p.y < R.DUMMY_HEIGHT:
                        hit = ("dummy", d)
                        break
                if hit is None:
                    for cx, cz, hx, hz, h, entity in self.blockers():
                        if abs(p.x - cx) < hx + m.radius and abs(p.z - cz) < hz + m.radius and p.y < h:
                            hit = ("block", entity)
                            break
                if hit is None and p.travelled >= m.max_range:
                    hit = ("expire", None)
                if hit is not None:
                    break
            if hit is None:
                alive.append(p)
            elif hit[0] == "dummy":
                d = hit[1]
                self._event(events, "ranged_attack", "player", d.id, "hit", damage=m.damage,
                            hp=max(0, d.hp - m.damage))
                self._damage(d, m.damage, (p.dx, p.dz), events)
            else:
                self._event(events, "ranged_attack", "player", hit[1], "miss")
        self.projectiles = alive

    def _camera(self) -> None:
        """A follow camera: behind and above the player, its yaw easing toward the player's; it moves
        in to the skill shot while an attack winds up, flies and lands."""
        a, hold = self.action, R.ticks(R.SHOT_HOLD_SECONDS, self.fps)
        if a is not None and a.kind != "dash":
            self.focus_until = max(self.focus_until, a.end + hold)
        if self.projectiles:
            self.focus_until = max(self.focus_until, self.tick + hold)
        want = 1.0 if self.tick <= self.focus_until else 0.0
        tau = R.SHOT_IN_SECONDS if want > self.close else R.SHOT_OUT_SECONDS
        self.close += (want - self.close) * (1.0 - math.exp(-self.dt / tau))
        ease = 1.0 - math.exp(-self.dt / self.shot.yaw_tau)
        diff = (self.yaw - self.cam_yaw + math.pi) % (2.0 * math.pi) - math.pi
        turn = diff * ease
        if self.shot.yaw_rate_deg:
            cap = math.radians(self.shot.yaw_rate_deg) * self.dt
            turn = max(-cap, min(cap, turn))
        self.cam_yaw += turn
        want_x, want_z = self._camera_goal()
        follow = 1.0 - math.exp(-self.dt / self.shot.follow_tau)
        self.cam_x += (want_x - self.cam_x) * follow
        self.cam_z += (want_z - self.cam_z) * follow
        if self.layout.camera_collide:
            # the trail can cut a corner the goal avoided
            reach = max(0.6, self.layout.sight(self.x, self.z, self.cam_x, self.cam_z))
            self.cam_x, self.cam_z = self.x + (self.cam_x - self.x) * reach, self.z + (self.cam_z - self.z) * reach

    def _camera_goal(self) -> tuple[float, float]:
        fx, fz = _forward(self._orbit_yaw())
        rx, rz = _right(self._orbit_yaw())
        distance, shoulder = self._lens("distance"), self._lens("shoulder")
        x = self.x - fx * distance + rx * shoulder
        z = self.z - fz * distance + rz * shoulder
        x, z = self.layout.clamp(x, z)  # stay inside the walls
        if self.layout.camera_collide:
            reach = max(0.6, self.layout.sight(self.x, self.z, x, z))
            x, z = self.x + (x - self.x) * reach, self.z + (z - self.z) * reach
        return x, z

    def _orbit_yaw(self) -> float:
        return self.cam_yaw - math.radians(self._lens("orbit_deg"))

    def _lens(self, name: str) -> float:
        """A camera parameter between the follow shot and the skill shot, at the current closeness."""
        base = getattr(R.CAMERA_SHOTS["follow"], name)
        return base + (getattr(self.shot, name) - base) * _smooth(self.close)

    # -- views ----------------------------------------------------------------------------------

    def _tilt(self, d: Dummy) -> tuple[float, tuple[float, float]]:
        fps = self.fps
        if d.state == "falling":
            return math.radians(86.0) * _smooth((self.tick - d.since) / R.ticks(R.DUMMY_FALL_SECONDS, fps)), d.fall_dir
        if d.state == "down":
            return math.radians(86.0), d.fall_dir
        if d.state == "rising":
            return math.radians(86.0) * (1.0 - _smooth((self.tick - d.since) / R.ticks(R.DUMMY_RISE_SECONDS, fps))), \
                d.fall_dir
        tx = tz = 0.0
        for tick, dx, dz in d.impulses:
            age = (self.tick - tick) / fps
            if age < 2.0:
                amp = math.radians(14.0) * math.exp(-age / 0.35) * math.sin(2.0 * math.pi * 1.8 * age + 0.6)
                tx, tz = tx + dx * amp, tz + dz * amp
        angle = math.hypot(tx, tz)
        return (angle, (tx / angle, tz / angle)) if angle > 1e-6 else (0.0, (0.0, 1.0))

    def snapshot(self) -> Snapshot:
        a = self.action
        action = None
        if a is not None:
            action = {"kind": a.kind, "age": self.tick - a.start, "windup": a.windup - a.start,
                      "length": a.end - a.start}
        dummies = []
        for d in self.dummies:
            angle, direction = self._tilt(d)
            dummies.append({"id": d.id, "x": d.x, "z": d.z, "hp": d.hp, "state": d.state, "tilt": angle,
                            "tilt_dir": direction})
        fx, fz = _forward(self.cam_yaw)
        rx, rz = _right(self._orbit_yaw())
        ahead, side = self._lens("look_ahead"), self._lens("shoulder")
        camera = {"eye": (self.cam_x, self._lens("height"), self.cam_z),
                  "target": (self.x + fx * ahead + rx * side, self._lens("look_height"),
                             self.z + fz * ahead + rz * side),
                  "hfov_deg": self._lens("hfov_deg"), "close": self.close,
                  "pushes": R.frames_tighter(self.shot)}
        inp = self.last_input
        return Snapshot(
            tick=self.tick, t=self.tick / self.fps,
            player={"x": self.x, "z": self.z, "yaw": self.yaw, "phase": self.phase, "speed": self.speed,
                    "action": action},
            dummies=tuple(dummies),
            projectiles=tuple((p.x, p.y, p.z) for p in self.projectiles),
            camera=camera,
            locomotion={"forward": inp.forward, "strafe": inp.strafe, "turn": inp.turn,
                        "dashing": a is not None and a.kind == "dash"},
            layout=self.layout.id,
        )
