"""A grey-box ray caster for the training ground.

One pass gives the shaded image (the ``style`` proxy, a white-box mannequin world like the capsule
proxies the VM was trained on), metric z-depth and class ids, so every proxy modality comes from
the same geometry. Pure numpy; a frame of 640x352 takes a fraction of a second on one core.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ... import proxy
from .layout import LAYOUTS
from .world import Snapshot, _forward, _right

HUMAN, PROP, WALL, GROUND, SKY = (proxy.CLASS_ID[n] for n in ("human", "prop", "building_structure", "terrain",
                                                                   "sky"))
SKY_TOP, SKY_HORIZON = np.array([0.74, 0.75, 0.78]), np.array([0.86, 0.86, 0.87])
LIGHT = np.array([0.45, 0.8, -0.35]) / np.linalg.norm([0.45, 0.8, -0.35])
GLOW = np.array([1.0, 0.78, 0.42])


@dataclass(frozen=True)
class Prim:
    kind: str  # capsule | sphere | box
    a: tuple[float, float, float]
    b: tuple[float, float, float]  # capsule end / box max; unused for spheres
    r: float
    albedo: float
    cls: int
    emissive: bool = False


# -- poses ----------------------------------------------------------------------------------------


def _rot(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return (v * math.cos(angle) + np.cross(axis, v) * math.sin(angle) + axis * np.dot(axis, v) *
            (1.0 - math.cos(angle)))


def _limb(origin, swing: float, bend: float, upper: float, lower: float, forward, down, side_out=0.0, side=None):
    """Two-segment limb hanging from ``origin``: ``swing`` rotates it toward ``forward``, ``bend``
    folds the lower segment back (legs) or up (arms, negative)."""
    direction = -down * math.cos(swing) + forward * math.sin(swing)
    if side is not None and side_out:
        direction = direction * math.cos(side_out) + side * math.sin(side_out)
    joint = origin + direction * upper
    lower_angle = swing - bend
    lower_dir = -down * math.cos(lower_angle) + forward * math.sin(lower_angle)
    if side is not None and side_out:
        lower_dir = lower_dir * math.cos(side_out) + side * math.sin(side_out)
    return joint, joint + lower_dir * lower


# Gestures the proxy body plays for a skill (a skill's ``pose``). Each key pose is
# (twist, lean, right arm, left arm) with an arm as (swing, elbow, out) in degrees: swing 0 hangs,
# 90 points ahead, 180 points up, negative reaches back; elbow < 0 folds the forearm up; out > 0
# opens the arm sideways, < 0 brings it across the body. ``charge`` is held at the end of the
# windup, ``strike`` a few frames after the release or hit; recovery eases back to the walk pose.
REST = (0.0, 0.0, (0.0, -15.0, 8.0), (0.0, -15.0, 8.0))
POSES = {
    "ranged_attack": {
        "point": ((35.0, 0.0, (88.0, 0.0, -25.0), (15.0, -30.0, 10.0)),
                  (35.0, 0.0, (92.0, 0.0, -25.0), (15.0, -30.0, 10.0))),
        "staff": ((30.0, 0.0, (80.0, -10.0, -20.0), (60.0, -40.0, -45.0)),
                  (25.0, 6.0, (95.0, 0.0, -15.0), (70.0, -30.0, -45.0))),
        "throw": ((40.0, -4.0, (100.0, -90.0, 55.0), (60.0, -20.0, -20.0)),
                  (10.0, 10.0, (98.0, 0.0, -10.0), (-20.0, -40.0, 15.0))),
        "overhead": ((20.0, -4.0, (165.0, -10.0, -5.0), (40.0, -40.0, 15.0)),
                     (25.0, 8.0, (95.0, 0.0, -15.0), (20.0, -40.0, 15.0))),
        "rifle": ((40.0, 4.0, (30.0, -60.0, 40.0), (60.0, -25.0, -45.0)),
                  (40.0, 0.0, (26.0, -60.0, 40.0), (56.0, -25.0, -45.0))),
        "push": ((10.0, -4.0, (35.0, -115.0, 5.0), (35.0, -115.0, 5.0)),
                 (0.0, 8.0, (90.0, 0.0, -18.0), (90.0, 0.0, -18.0))),
    },
    "melee_attack": {
        "punch": ((-15.0, 0.0, (-35.0, -100.0, 10.0), (40.0, -90.0, -10.0)),
                  (15.0, 8.0, (85.0, 0.0, -10.0), (20.0, -90.0, 0.0))),
        "palm": ((-20.0, 4.0, (-20.0, -110.0, 15.0), (50.0, -70.0, -20.0)),
                 (20.0, 10.0, (78.0, 0.0, -12.0), (-10.0, -60.0, 10.0))),
        "thrust": ((-25.0, 0.0, (-10.0, -100.0, 0.0), (30.0, -70.0, -20.0)),
                   (10.0, 14.0, (85.0, -10.0, -15.0), (70.0, -30.0, -40.0))),
        "swing": ((-45.0, 0.0, (70.0, -30.0, 85.0), (60.0, -50.0, -40.0)),
                  (35.0, 8.0, (80.0, -5.0, -60.0), (55.0, -40.0, -50.0))),
        "overhead": ((-10.0, -8.0, (165.0, -20.0, 0.0), (160.0, -25.0, -10.0)),
                     (0.0, 18.0, (65.0, 0.0, -10.0), (60.0, 0.0, -15.0))),
        "claw": ((-30.0, -4.0, (135.0, -40.0, 45.0), (30.0, -60.0, 0.0)),
                 (30.0, 12.0, (40.0, -10.0, -45.0), (20.0, -60.0, 10.0))),
    },
}
DEFAULT_POSE = {"ranged_attack": "point", "melee_attack": "punch"}
STRIKE_FRAMES = 3


def _blend(a: tuple, b: tuple, p: float) -> tuple:
    p = _ease(p)
    if isinstance(a, tuple):
        return tuple(_blend(x, y, p) for x, y in zip(a, b))
    return a + (b - a) * p


def _gesture(kind: str, pose: str | None, age: int, windup: int, length: int, rest: tuple) -> tuple:
    """The (twist, lean, right arm, left arm) key pose of a skill ``age`` frames in."""
    charge, strike = POSES[kind][pose or DEFAULT_POSE[kind]]
    if age < windup:
        return _blend(rest, charge, (age + 1) / windup)
    after = age - windup
    if after < STRIKE_FRAMES:
        return _blend(charge, strike, (after + 1) / STRIKE_FRAMES)
    return _blend(strike, rest, (after - STRIKE_FRAMES + 1) / max(1, length - windup - STRIKE_FRAMES))


def mannequin(player: dict, locomotion: dict | None = None, poses: dict | None = None) -> list[Prim]:
    """The protagonist: capsule limbs, a sphere head, animated from the player's state. ``poses``
    maps a mechanic to its gesture (``POSES``); legs stride along the direction the player moves."""
    up = np.array([0.0, 1.0, 0.0])
    yaw = player["yaw"]
    action = player["action"] or {}
    kind, age = action.get("kind"), action.get("age", 0)
    windup, length = max(1, action.get("windup", 1)), max(1, action.get("length", 1))
    stride = min(1.0, player["speed"] / 2.2)
    if kind == "dash":
        stride = 0.0
    phase = player["phase"]
    swing = math.radians(30.0) * stride * math.sin(phase)
    walk_elbow = -15.0 - 30.0 * stride
    rest = (0.0, 5.0 * stride, (0.0, walk_elbow, 8.0), (0.0, walk_elbow, 8.0))
    twist, lean_deg, arm_r, arm_l = rest
    if kind in POSES:
        twist, lean_deg, arm_r, arm_l = _gesture(kind, (poses or {}).get(kind), age, windup, length, rest)
    fx, fz = _forward(yaw + math.radians(twist))
    rx, rz = _right(yaw + math.radians(twist))
    f, r = np.array([fx, 0.0, fz]), np.array([rx, 0.0, rz])
    loco = locomotion or {}
    move_f, move_s = float(loco.get("forward", 0.0)), float(loco.get("strafe", 0.0))
    yf, yr = _forward(yaw), _right(yaw)
    m = np.array([yf[0] * move_f + yr[0] * move_s, 0.0, yf[1] * move_f + yr[1] * move_s])
    m = m / np.linalg.norm(m) if np.linalg.norm(m) > 1e-6 else f
    base = np.array([player["x"], 0.0, player["z"]])
    lean = math.radians(22.0 if kind == "dash" else lean_deg)
    lean_dir = f if kind else m
    body_up = up * math.cos(lean) + lean_dir * math.sin(lean)
    hip = base + up * (0.95 + 0.035 * stride * math.cos(2.0 * phase))
    chest = hip + body_up * 0.5
    head = hip + body_up * 0.78

    knee_l = math.radians(40.0) * stride * max(0.0, math.sin(phase + math.pi / 2))
    knee_r = math.radians(40.0) * stride * max(0.0, math.sin(phase - math.pi / 2))
    prims: list[Prim] = []

    def cap(a, b, radius, albedo=0.78):
        prims.append(Prim("capsule", tuple(a), tuple(b), radius, albedo, HUMAN))

    # legs, swinging along the direction of travel
    for sign, leg_swing, knee in ((-1.0, swing, knee_l), (1.0, -swing, knee_r)):
        stride_dir = m
        if kind == "dash":
            leg_swing, knee = (math.radians(30.0), math.radians(50.0)) if sign < 0 else (math.radians(-25.0), 0.3)
            stride_dir = f
        hip_j = hip + r * 0.11 * sign
        knee_p, ankle = _limb(hip_j, leg_swing, knee, 0.46, 0.44, stride_dir, up)
        cap(hip_j, knee_p, 0.08)
        cap(knee_p, ankle, 0.065)
        cap(ankle, ankle + f * 0.14 - up * 0.02, 0.05)
    # torso and head
    cap(hip + body_up * 0.02, chest - body_up * 0.04, 0.17)
    cap(chest - body_up * 0.04 - r * 0.12, chest - body_up * 0.04 + r * 0.12, 0.13)
    prims.append(Prim("sphere", tuple(head), (0, 0, 0), 0.125, 0.8, HUMAN))
    # arms: the walk's counter-swing, or the skill's gesture
    for sign, (a_swing, a_elbow, a_out) in ((-1.0, arm_l), (1.0, arm_r)):
        shoulder = chest - body_up * 0.04 + r * 0.24 * sign
        arm_swing = math.radians(a_swing) + (-swing if sign < 0 else swing) * 0.8 * (0.0 if kind in POSES else 1.0)
        elbow, out = math.radians(a_elbow), math.radians(a_out)
        if kind == "dash":
            arm_swing, elbow = math.radians(-40.0), math.radians(-30.0)
        hand_elbow, hand = _limb(shoulder, arm_swing, elbow, 0.29, 0.27, f, up, out * sign, r)
        cap(shoulder, hand_elbow, 0.06)
        cap(hand_elbow, hand, 0.055)
    return prims


def _ease(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3.0 - 2.0 * p)


def dummy(d: dict) -> list[Prim]:
    """A wooden training dummy on a round base; it rocks when hit and falls when out of hit points."""
    base = np.array([d["x"], 0.0, d["z"]])
    pivot = base + np.array([0.0, 0.12, 0.0])
    tdx, tdz = d["tilt_dir"]
    axis = np.cross(np.array([0.0, 1.0, 0.0]), np.array([tdx, 0.0, tdz]))
    angle = d["tilt"]

    def at(offset) -> np.ndarray:
        v = np.asarray(offset, dtype=float)
        return pivot + (_rot(v, axis, angle) if angle > 1e-6 else v)

    prims = [Prim("capsule", tuple(base + [0, 0.05, 0]), tuple(base + [0, 0.07, 0]), 0.38, 0.62, PROP)]
    prims.append(Prim("capsule", tuple(at([0, 0.0, 0])), tuple(at([0, 0.78, 0])), 0.06, 0.66, PROP))
    prims.append(Prim("capsule", tuple(at([0, 0.88, 0])), tuple(at([0, 1.33, 0])), 0.24, 0.72, PROP))
    prims.append(Prim("sphere", tuple(at([0, 1.55, 0])), (0, 0, 0), 0.15, 0.72, PROP))
    for side in (-1.0, 1.0):
        prims.append(Prim("capsule", tuple(at([0.2 * side, 1.18, 0])), tuple(at([0.42 * side, 1.25, 0.05])), 0.05,
                          0.66, PROP))
    return prims


def static_scene(layout: str = "courtyard") -> list[Prim]:
    prims = []
    for cx, cz, hx, hz, h, entity in LAYOUTS[layout].blockers():
        albedo = 0.70 if entity == "wall" else 0.66
        cls = WALL if entity == "wall" else PROP
        prims.append(Prim("box", (cx - hx, 0.0, cz - hz), (cx + hx, h, cz + hz), 0.0, albedo, cls))
    return prims


def scene(snap: Snapshot, player: bool = True, poses: dict | None = None, projectiles: bool = True) -> list[Prim]:
    prims = static_scene(snap.layout) + (mannequin(snap.player, snap.locomotion, poses) if player else [])
    for d in snap.dummies:
        prims += dummy(d)
    for x, y, z in snap.projectiles if projectiles else ():
        prims.append(Prim("sphere", (x, y, z), (0, 0, 0), 0.13, 1.0, PROP, emissive=True))
    return prims


# -- ray casting ----------------------------------------------------------------------------------


def camera_rays(cam: dict, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(cam["eye"], dtype=np.float64)
    fwd = np.asarray(cam["target"], dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    focal = (width / 2.0) / math.tan(math.radians(cam["hfov_deg"]) / 2.0)
    xs = (np.arange(width) + 0.5 - width / 2.0) / focal
    ys = -(np.arange(height) + 0.5 - height / 2.0) / focal
    gx, gy = np.meshgrid(xs, ys)
    dirs = gx[..., None] * right + gy[..., None] * up + fwd
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    return eye, dirs.reshape(-1, 3), fwd


def c2w(cam: dict) -> list[list[float]]:
    """Camera-to-world (x right, y up, z forward) for logs and camera-aware consumers."""
    eye = np.asarray(cam["eye"], dtype=np.float64)
    fwd = np.asarray(cam["target"], dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, fwd, eye
    return m.round(6).tolist()


def _sphere(o, d, c, r):
    oc = o - c
    b = d @ oc
    h = b * b - (oc @ oc - r * r)
    t = np.full(d.shape[0], np.inf)
    ok = h > 0
    t[ok] = -b[ok] - np.sqrt(h[ok])
    t[t <= 1e-4] = np.inf
    return t


def _capsule(o, d, a, b, r):
    """Ray-capsule distance (Inigo Quilez's formulation), with ``o`` shared by every ray."""
    ba, oa = b - a, o - a
    baba, bard, baoa = ba @ ba, d @ ba, ba @ oa
    rdoa, oaoa = d @ oa, oa @ oa
    aa = baba - bard * bard
    bb = baba * rdoa - baoa * bard
    cc = baba * oaoa - baoa * baoa - r * r * baba
    h = bb * bb - aa * cc
    t = np.full(d.shape[0], np.inf)
    ok = (h >= 0) & (np.abs(aa) > 1e-12)
    tc = np.where(ok, (-bb - np.sqrt(np.maximum(h, 0))) / np.where(np.abs(aa) > 1e-12, aa, 1.0), np.inf)
    y = baoa + tc * bard
    body = ok & (y > 0) & (y < baba)
    t[body] = tc[body]
    for end, sign in ((a, 0), (b, 1)):
        caps = ok & ~body & ((y <= 0) if sign == 0 else (y >= baba))
        if caps.any():
            ts = _sphere(o, d[caps], end, r)
            t[caps] = ts
    t[t <= 1e-4] = np.inf
    return t


def _box(o, d, lo, hi):
    inv = 1.0 / np.where(np.abs(d) > 1e-12, d, 1e-12)
    t0, t1 = (lo - o) * inv, (hi - o) * inv
    tmin, tmax = np.minimum(t0, t1).max(axis=1), np.maximum(t0, t1).min(axis=1)
    t = np.where((tmax >= tmin) & (tmax > 1e-4), np.where(tmin > 1e-4, tmin, np.inf), np.inf)
    return t


def _normal(prim: Prim, p: np.ndarray) -> np.ndarray:
    if prim.kind == "sphere":
        n = p - np.asarray(prim.a)
    elif prim.kind == "capsule":
        a, b = np.asarray(prim.a), np.asarray(prim.b)
        ba = b - a
        h = np.clip(((p - a) @ ba) / max(ba @ ba, 1e-12), 0.0, 1.0)
        n = p - (a + h[:, None] * ba)
    else:
        lo, hi = np.asarray(prim.a), np.asarray(prim.b)
        c, half = (lo + hi) / 2, (hi - lo) / 2
        q = (p - c) / np.maximum(half, 1e-9)
        axis = np.abs(q).argmax(axis=1)
        n = np.zeros_like(p)
        n[np.arange(len(p)), axis] = np.sign(q[np.arange(len(p)), axis])
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def _bound(prim: Prim) -> tuple[np.ndarray, float]:
    if prim.kind == "sphere":
        return np.asarray(prim.a), prim.r
    a, b = np.asarray(prim.a), np.asarray(prim.b)
    if prim.kind == "capsule":
        return (a + b) / 2, float(np.linalg.norm(b - a)) / 2 + prim.r
    return (a + b) / 2, float(np.linalg.norm(b - a)) / 2


@dataclass(frozen=True)
class Buffers:
    image: np.ndarray  # [H, W, 3] uint8 shaded grey-box render
    depth: np.ndarray  # [H, W] float32 metric z-depth, 0 where there is sky
    ids: np.ndarray  # [H, W] uint8 class ids


def cast(snap: Snapshot, width: int, height: int, player: bool = True, glow=None, poses: dict | None = None,
         projectiles: bool = True) -> Buffers:
    glow = GLOW if glow is None else np.asarray(glow, dtype=np.float64)
    eye, d, fwd = camera_rays(snap.camera, width, height)
    n_rays = d.shape[0]
    t_best = np.full(n_rays, np.inf)
    owner = np.full(n_rays, -1, dtype=np.int64)  # -2 ground, >= 0 primitive index
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(d[:, 1] < -1e-6, -eye[1] / d[:, 1], np.inf)
    t_best, owner = np.where(tg < t_best, tg, t_best), np.where(tg < t_best, -2, owner)
    prims = scene(snap, player, poses, projectiles)
    for index, prim in enumerate(prims):
        centre, radius = _bound(prim)
        to = centre - eye
        dist = float(np.linalg.norm(to))
        if dist > radius:
            cos_cone = math.cos(math.asin(min(1.0, radius / dist)))
            sel = np.nonzero(d @ (to / dist) >= cos_cone - 1e-6)[0]
        else:
            sel = np.arange(n_rays)
        if sel.size == 0:
            continue
        ds = d[sel]
        if prim.kind == "sphere":
            t = _sphere(eye, ds, np.asarray(prim.a), prim.r)
        elif prim.kind == "capsule":
            t = _capsule(eye, ds, np.asarray(prim.a), np.asarray(prim.b), prim.r)
        else:
            t = _box(eye, ds, np.asarray(prim.a), np.asarray(prim.b))
        closer = t < t_best[sel]
        t_best[sel[closer]] = t[closer]
        owner[sel[closer]] = index

    hit = np.isfinite(t_best)
    p = eye + d * np.where(hit, t_best, 0.0)[:, None]
    colour = np.empty((n_rays, 3))
    sky_w = np.clip(d[:, 1] * 2.5, 0.0, 1.0)[:, None]
    colour[:] = SKY_HORIZON * (1 - sky_w) + SKY_TOP * sky_w
    ids = np.full(n_rays, SKY, dtype=np.uint8)

    ground = owner == -2
    if ground.any():
        shade = np.full(ground.sum(), 0.62)
        gp = p[ground]
        feet = [(snap.player["x"], snap.player["z"])] if player else []
        for x, z in feet + [(dm["x"], dm["z"]) for dm in snap.dummies]:
            r2 = (gp[:, 0] - x) ** 2 + (gp[:, 2] - z) ** 2
            shade *= 1.0 - 0.38 * np.exp(-r2 / 0.18)
        colour[ground] = shade[:, None]
        ids[ground] = GROUND
    glow_points = []
    for index, prim in enumerate(prims):
        mine = owner == index
        if prim.emissive:
            glow_points.append(np.asarray(prim.a))
        if not mine.any():
            continue
        if prim.emissive:
            colour[mine] = glow
        else:
            n = _normal(prim, p[mine])
            lambert = np.clip(n @ LIGHT, 0.0, 1.0)
            sky = 0.5 + 0.5 * n[:, 1]
            colour[mine] = (prim.albedo * (0.42 + 0.38 * lambert + 0.22 * sky))[:, None]
        ids[mine] = prim.cls
    fog = 1.0 - np.exp(-np.where(hit, t_best, 0.0) / 70.0)
    colour[hit] = colour[hit] * (1 - fog[hit, None]) + SKY_HORIZON * fog[hit, None]
    for centre in glow_points:  # a soft halo, as the trained capsule proxies draw their projectile
        to = centre - eye
        dist = np.linalg.norm(to)
        cosang = d @ (to / dist)
        ang = np.arccos(np.clip(cosang, -1.0, 1.0))
        visible = (t_best >= dist - 0.3)
        halo = np.exp(-(ang * dist / 0.32) ** 2) * visible
        colour += halo[:, None] * (glow - colour) * 0.85
    depth = np.where(hit, t_best * (d @ fwd), 0.0).astype(np.float32)
    image = np.clip(colour * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return Buffers(image.reshape(height, width, 3), depth.reshape(height, width), ids.reshape(height, width))


def render(snap: Snapshot, width: int, height: int, modalities=("style", ), supersample: int = 2, glow=None,
           poses: dict | None = None, projectiles: bool = True) -> dict:
    """``{modality: [H, W, 3] uint8}``. ``style`` is supersampled; codes are point-sampled.
    ``projectiles=False`` hides the flying shot (a skill that strikes at the target, not across)."""
    s = max(1, supersample)
    buf = cast(snap, width * s, height * s, glow=glow, poses=poses, projectiles=projectiles)
    out = {}
    if "style" in modalities:
        out["style"] = buf.image.reshape(height, s, width, s, 3).mean(axis=(1, 3)).round().astype(np.uint8)
    depth, ids = buf.depth[s // 2::s, s // 2::s], buf.ids[s // 2::s, s // 2::s]
    if "depth" in modalities:
        out["depth"] = proxy.depth_frame(depth)
    if "semantic" in modalities:
        out["semantic"] = proxy.semantic_frame(ids)
    if "duv" in modalities:
        out["duv"] = proxy.duv_frame(depth, ids)
    return out
