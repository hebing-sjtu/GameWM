import json
import math
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from gamewm_engine import load, plan  # noqa: E402
from gamewm_engine import proxy
from gamewm_engine.games.training_ground import DEMO, Bot, Face, Hold, Input, Press, Script, TrainingGround, World
from gamewm_engine.games.training_ground import rules as R
from gamewm_engine.games.training_ground.inputs import facing
from gamewm_engine.games.training_ground.render import cast, render
from gamewm_engine.stream import CHUNKING, Chunk, DirectorySink, pump, read_directory


def run(script, seconds, fps=24):
    world = World(fps)
    events = []
    for _ in range(round(seconds * fps)):
        events += world.step(script(world))
    return world, events


def test_melee_lands_at_the_end_of_its_windup():
    world = World(24)
    world.x, world.z = R.DUMMIES["dummy_center"][0], R.DUMMIES["dummy_center"][1] - 1.2
    events = world.step(Input(melee=True))
    assert not events
    for _ in range(R.ticks(R.MELEE.windup, 24) - 1):
        assert not world.step(Input())
    (hit, ) = world.step(Input())
    assert (hit.mechanic, hit.target, hit.outcome) == ("melee_attack", "dummy_center", "hit")
    assert hit.detail == {"frame": 6, "damage": 34, "hp": 66}


def test_melee_out_of_reach_or_behind_misses():
    world = World(24)
    world.x, world.z = 0.0, R.DUMMIES["dummy_center"][1] - 2.5
    world.step(Input(melee=True))
    events = [e for _ in range(10) for e in world.step(Input())]
    assert [(e.mechanic, e.outcome) for e in events] == [("melee_attack", "miss")]


def test_cooldowns_gate_presses():
    world = World(24)
    world.x, world.z = 0.0, R.DUMMIES["dummy_center"][1] - 1.1
    hits = []
    for tick in range(48):
        hits += [e for e in world.step(Input(melee=True)) if e.mechanic == "melee_attack"]
    assert [e.detail["frame"] for e in hits] == [6, 6 + 19, 6 + 38]  # cooldown 0.8 s = 19 ticks


def test_three_flame_fists_knock_a_dummy_down_and_it_gets_back_up():
    world = World(24)
    world.x, world.z = 0.0, R.DUMMIES["dummy_center"][1] - 1.1
    events = []
    for tick in range(24 * 7):
        events += world.step(Input(melee=tick in (0, 20, 40)))
    kinds = [(e.mechanic, e.actor, e.outcome) for e in events]
    assert kinds.count(("melee_attack", "player", "hit")) == 3
    down = next(e for e in events if e.mechanic == "knockdown")
    rise = next(e for e in events if e.mechanic == "respawn")
    assert down.actor == rise.actor == "dummy_center"
    assert rise.detail["frame"] - down.detail["frame"] == R.ticks(0.5, 24) + R.ticks(3.0, 24)
    assert world.dummies[1].state == "up" and world.dummies[1].hp == 100


def test_fireball_flies_to_the_dummy_ahead():
    world, events = run(Script([Press(0.0, "ranged")]), 2.0)
    cast_, hit = [e for e in events if e.mechanic == "ranged_attack"]
    assert cast_.outcome == "cast" and cast_.detail["frame"] == R.ticks(R.RANGED.windup, 24)
    assert (hit.target, hit.outcome) == ("dummy_center", "hit")
    flight = (hit.detail["frame"] - cast_.detail["frame"]) / 24
    distance = R.DUMMIES["dummy_center"][1] - R.RANGED.release_forward - R.DUMMY_RADIUS - R.RANGED.radius
    assert flight == pytest.approx(distance / R.RANGED.speed, abs=1.5 / 24)


def test_fireball_turned_away_hits_the_wall():
    world, events = run(Script([Hold(0.0, 1.2, turn=1.0), Press(1.25, "ranged")]), 4.0)
    miss = [e for e in events if e.mechanic == "ranged_attack" and e.outcome != "cast"]
    assert [(e.target, e.outcome) for e in miss] == [("wall", "miss")]


def test_dash_covers_its_distance_unless_blocked():
    world, events = run(Script([Press(0.0, "dash")]), 0.5)
    assert world.z == pytest.approx(R.DASH.distance)
    assert [e.outcome for e in events if e.mechanic == "dash"] == ["start", "end"]
    world = World(24)
    world.z = R.DUMMIES["dummy_center"][1] - 2.0
    world.step(Input(dash=True))
    for _ in range(10):
        world.step(Input())
    assert world.z == pytest.approx(R.DUMMIES["dummy_center"][1] - R.DUMMY_RADIUS - R.PLAYER_RADIUS)


def test_face_turns_onto_the_target():
    world, _ = run(Script([Face(0.0, 1.0, "dummy_right")]), 1.0)
    assert facing(world, "dummy_right") < 1.0


def test_the_demo_hits_every_beat_and_is_deterministic():
    _, events = run(DEMO, DEMO.seconds)
    hits = [(e.mechanic, e.target) for e in events if e.outcome == "hit"]
    assert hits.count(("ranged_attack", "dummy_center")) == 4
    assert hits.count(("melee_attack", "dummy_right")) == 3
    assert {e.actor for e in events if e.mechanic == "knockdown"} == {"dummy_center", "dummy_right"}
    _, again = run(DEMO, DEMO.seconds)
    assert [(e.t, e.mechanic, e.target, e.outcome) for e in events] == [
        (e.t, e.mechanic, e.target, e.outcome) for e in again]


def test_bot_scores_hits():
    _, events = run(Bot(seed=1), 12.0)
    assert sum(e.outcome == "hit" for e in events) >= 4


def test_renderer_writes_the_trained_codes():
    world = World(24)
    world.step(Input())
    snap = world.snapshot()
    buf = cast(snap, 160, 88)
    present = set(np.unique(buf.ids).tolist())
    assert {proxy.CLASS_ID[c] for c in ("sky", "terrain", "human", "prop", "building_structure")} <= present
    out = render(snap, 160, 88, ("style", "depth", "semantic", "duv"))
    assert all(frame.shape == (88, 160, 3) and frame.dtype == np.uint8 for frame in out.values())
    palette = {tuple(c) for c in proxy.semantic_palette().tolist()}
    assert {tuple(c) for c in out["semantic"].reshape(-1, 3).tolist()} <= palette
    assert set(np.unique(out["duv"][..., 1]).tolist()) <= set(proxy.DUV_U)
    assert set(np.unique(out["duv"][..., 2]).tolist()) <= set(proxy.DUV_V)
    sky = (out["semantic"] == proxy.semantic_palette()[proxy.CLASS_ID["sky"]]).all(axis=-1)
    assert sky.any() and (out["depth"][..., 0][sky] == 0).all() and (out["depth"][..., 0][~sky] > 0).all()
    assert (out["depth"][..., 0] == out["duv"][..., 0]).all()


@pytest.mark.parametrize("model", ["h3-33b", "wan2.2-5b", "wan2.2-14b"])
def test_chunks_follow_the_vae(model):
    c = CHUNKING[model]
    game = TrainingGround(model, ("style", ), size=(64, 36), supersample=1)
    frames = c.prefix + 3 * c.size + 2
    messages = list(game.stream(Script([Hold(0.0, 99.0, forward=1.0)]), frames=frames))
    session, *chunks, end = messages
    assert session["type"] == "session" and end["type"] == "end" and end["frames"] == frames
    assert all(isinstance(m, Chunk) for m in chunks)
    assert [m.span for m in chunks] == [(0, c.prefix)] + [
        (c.prefix + i * c.size, c.prefix + (i + 1) * c.size) for i in range(3)] + [(frames - 2, frames)]
    assert [m.frames["style"].shape[0] for m in chunks] == [b - a for a, b in (m.span for m in chunks)]
    last = chunks[3]
    assert last.meta["text"]["window"] == [0, 3]
    assert last.meta["text"]["document"].startswith("Reference modality: style video.\n[0.00s-5.17s] ")
    assert len(last.meta["states"]) == c.size and len(last.meta["states"][0]["camera"]["c2w"]) == 4


def test_h3_window_spans_six_chunks():
    c = CHUNKING["h3-33b"]
    assert [c.bounds(k) for k in range(3)] == [(0, 5), (5, 22), (22, 39)]
    assert c.bounds(7) == (107, 124)  # 124 frames = 37 latents = 2 + 7 * 5
    assert c.window_start(7) == 2 and c.bounds(c.window_start(7))[0] == 22


def test_stream_round_trips_through_a_directory(tmp_path):
    game = TrainingGround("h3-33b", ("style", "duv"), size=(64, 36), supersample=1)
    pump(game.stream(DEMO, frames=39), DirectorySink(tmp_path))
    session, *chunks, end = list(read_directory(tmp_path))
    assert session["modalities"] == ["style", "duv"] and end["chunks"] == len(chunks) == 3
    assert chunks[1].frames["duv"].shape == (17, 36, 64, 3)
    rows = [json.loads(line) for line in (tmp_path / "chunks.jsonl").read_text().splitlines()]
    assert rows[1]["proxy"] == {"style": "frames/style/000001.npy", "duv": "frames/duv/000001.npy"}


def test_take_sink_writes_a_request_the_vm_path_accepts(tmp_path):
    pytest.importorskip("av")
    pytest.importorskip("PIL")
    game = TrainingGround("h3-33b", ("style", ), size=(64, 36), supersample=1)
    take = game.take_sink(tmp_path)
    pump(game.stream(DEMO, DEMO.seconds), take)
    request = load(tmp_path / "request.json")
    layout = plan(request)
    assert layout.output_frames == 294 and [s.role for s in layout.segments] == ["w0_omni", "wn_omni", "wn_omni"]
    text = " ".join(s.caption for s in layout.segments)
    first, second, _ = (s.caption for s in layout.segments)
    assert "raises his right hand to shoulder height at the center training dummy" in first
    assert "hurls another fireball at the center training dummy" in first
    assert first.index("walks toward the center training dummy with an easy stride") < first.index("raises his")
    assert "dashes forward low in a streak of embers" in second
    assert "drives a burning punch into the right training dummy" in second
    assert second.count("throws another flaming punch into the right training dummy") == 2
    assert "knocking it to the ground" in text and "backs away with light, careful steps" in text
    assert "hood, stands loose and ready, shifting his weight, then walks" in first
    assert all(not s.caption.count("{") for s in layout.segments)
    import av

    with av.open(str(tmp_path / "proxy" / "style.mp4")) as container:
        assert sum(1 for _ in container.decode(video=0)) == 294
    assert math.isclose(request.seconds, 294 / 24, abs_tol=1e-6)


def _showcase(tmp_path, **kwargs):
    from gamewm_engine.games.training_ground.inputs import SHOWCASE

    pytest.importorskip("av")
    game = TrainingGround("h3-33b", ("style", ), size=(64, 36), supersample=1, **kwargs)
    pump(game.stream(SHOWCASE, SHOWCASE.seconds), game.take_sink(tmp_path))
    (segment, ) = plan(load(tmp_path / "request.json")).segments
    return game, segment.caption


def test_push_camera_closes_in_for_each_attack_and_eases_out_for_the_dash(tmp_path):
    game, _ = _showcase(tmp_path, camera="push")
    close = {s.tick: s.camera["close"] for s in game.snapshots}
    hfov = {s.tick: s.camera["hfov_deg"] for s in game.snapshots}
    assert close[40] < 0.01 and hfov[40] == R.CAMERA_HFOV_DEG
    assert close[70] > 0.95 and abs(hfov[70] - R.CAMERA_SHOTS["push"].hfov_deg) < 0.5
    assert close[90] < 0.35 and close[110] > 0.95
    plain, _ = _showcase(tmp_path / "follow")
    assert all(s.camera["hfov_deg"] == R.CAMERA_HFOV_DEG for s in plain.snapshots)


@pytest.mark.parametrize("style", ["chain", "prose", "phased", "detailed", "timed", "spans", "coarse"])
def test_every_text_style_fills_its_placeholders(tmp_path, style):
    _, caption = _showcase(tmp_path, camera="push", text_style=style)
    assert "{" not in caption and "  " not in caption and " ," not in caption
    assert "the camera pushes in close behind him" in caption
    if style in ("phased", "detailed", "timed", "spans"):
        assert "raises his right hand to shoulder height toward the center training dummy, a flame" in caption
        assert "t explodes in a ball of flame against the center training dummy, rocking it back" in caption
    if style == "timed":
        assert "At 2.6 seconds, he thrusts his palm forward" in caption
    if style == "spans":
        assert "[2.62s-2.93s] He thrusts his palm forward" in caption
    if style == "coarse":
        assert "he uses Fireball on the center training dummy" in caption


EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "training_ground"


def _fx_character(cid: str) -> tuple[dict, dict]:
    roster = json.loads((EXAMPLES / "roster_fx.json").read_text())
    entry = next(c for c in roster["characters"] if c["id"] == cid)
    character = {"subject": "protagonist", "look": entry["look"], "pronoun": entry["pronoun"],
                 "skills": entry["skills"], "moves": entry["moves"]}
    return character, {"medium": "A 3D video game", "setting": entry["setting"], "lighting": entry["lighting"]}


def test_example_effect_specs_pass_and_the_old_roster_does_not():
    from gamewm_engine.games.training_ground import effects

    for path in sorted((EXAMPLES / "effects").glob("*.json")):
        roster = json.loads((EXAMPLES / "roster.json").read_text())
        pronoun = next(c["pronoun"] for c in roster["characters"] if c["id"] == path.stem)
        assert effects.validate(json.loads(path.read_text()), pronoun) == [], path.stem
    old = json.loads((EXAMPLES / "roster.json").read_text())["characters"][1]
    flat = {k: {**v, **v.get("phases", {}), "pose": "staff" if k == "ranged_attack" else "thrust",
                "kind": "projectile"} for k, v in old["skills"].items()}
    problems = effects.validate(flat, old["pronoun"])
    assert "ranged_attack.decay: missing" in problems
    assert any(p.startswith("ranged_attack.impact: describe the effect as light") for p in problems)


def test_long_bout_tells_a_skill_and_its_decay_in_every_segment(tmp_path):
    from gamewm_engine.games.training_ground.inputs import LONG

    pytest.importorskip("av")
    character, scene = _fx_character("01_fire_mage")
    game = TrainingGround("h3-33b", ("style", ), character=character, scene=scene, size=(64, 36), supersample=1,
                          camera="push", text_style="phased")
    pump(game.stream(LONG, frames=719), game.take_sink(tmp_path))
    layout = plan(load(tmp_path / "request.json"))
    assert layout.output_frames == 719 and len(layout.segments) == 8
    downs = [e for e in game.events if e.mechanic == "knockdown"]
    assert len(downs) == 3 and sum(e.mechanic == "respawn" for e in game.events) == 3
    for s in layout.segments:
        assert any(e.actor == "player" and e.outcome == "hit" for e in s.events), s.index
        assert "vanish in a split second" in s.caption or "die out at once" in s.caption, s.index
        assert "{" not in s.caption and "eases back out" not in s.caption
        assert s.caption.count("the camera pushes in") <= 1 and len(s.caption.split()) < 360


def test_detailed_style_tells_every_use_and_a_steady_camera_never_pushes_in(tmp_path):
    from gamewm_engine.games.training_ground.inputs import LONG

    pytest.importorskip("av")
    character, scene = _fx_character("08_arcane_sorceress")
    game = TrainingGround("h3-33b", ("style", ), character=character, scene=scene, size=(64, 36), supersample=1,
                          camera="steady", text_style="detailed")
    pump(game.stream(LONG, frames=719), game.take_sink(tmp_path))
    layout = plan(load(tmp_path / "request.json"))
    windup = "plants her feet and draws both hands to her chest"
    for s in layout.segments:
        assert "{" not in s.caption and "pushes in" not in s.caption, s.index
        casts = sum(e.mechanic == "ranged_attack" and e.outcome == "hit" and e.detail["frame"] >= s.start
                    for e in s.events)
        assert s.caption.count(windup) >= min(casts, 1), s.index
    text = " ".join(s.caption for s in layout.segments)
    assert "once more" in text and "The glowing violet runes she left behind flicker and vanish" in text
    assert max(len(s.caption.split()) for s in layout.segments) < 440
    assert "white marble" in layout.segments[0].caption


def test_a_shot_that_lands_after_the_window_is_told_without_its_impact(tmp_path):
    character, scene = _fx_character("01_fire_mage")
    game = TrainingGround("h3-33b", ("style", ), character=character, scene=scene, size=(64, 36), supersample=1,
                          text_style="phased")
    script = Script([Face(0.0, 0.6, "dummy_center"), Press(4.62, "ranged"), Hold(4.7, 124 / 24)])
    for _ in range(124):
        game.feed(script(game.world))
    from gamewm_engine import compose

    events = tuple(e for e in game.events if e.detail["frame"] < 124)
    assert [(e.outcome, e.target) for e in events if e.mechanic == "ranged_attack"] == [("cast", "dummy_center")]
    caption = compose.caption(game._request, game.action_for(0, 124), events)
    assert "raises his wand at the center training dummy" in caption and "flicks the wand forward" in caption
    assert "explodes" not in caption


def test_skill_poses_move_the_proxy_body_and_at_target_shots_do_not_fly():
    from gamewm_engine.games.training_ground.render import POSES, mannequin

    player = {"x": 0.0, "z": 0.0, "yaw": 0.0, "phase": 0.0, "speed": 0.0,
              "action": {"kind": "ranged_attack", "age": 8, "windup": 10, "length": 17}}
    hands = {name: mannequin(player, None, {"ranged_attack": name})[-1].b for name in POSES["ranged_attack"]}
    assert hands["overhead"][1] > 1.9 > 1.6 > hands["point"][1] > 1.2  # the hammer goes up to the sky
    assert len({tuple(round(v, 2) for v in h) for h in hands.values()}) == len(hands)
    world = World(24)
    world.x, world.z = 0.0, 4.0
    for _ in range(R.ticks(R.RANGED.windup, 24) + 3):
        world.step(Input(ranged=True))
    snap = world.snapshot()
    assert snap.projectiles
    shown, hidden = render(snap, 96, 54), render(snap, 96, 54, projectiles=False)
    glow = lambda img: int((np.abs(img.astype(int) - (255, 199, 107)).sum(-1) < 40).sum())  # noqa: E731
    assert glow(shown["style"]) > 0 and glow(hidden["style"]) == 0


@pytest.mark.parametrize("name", ["CORRIDOR_LONG", "ROOMS_LONG"])
def test_layout_bouts_never_miss_and_keep_the_camera_inside(name):
    from gamewm_engine.games.training_ground import inputs

    script = getattr(inputs, name)
    world = World(24, "push", script.layout)
    events = []
    tall = [b for b in world.layout.blockers() if b[4] >= 2.0]
    for _ in range(719):
        events += world.step(script(world))
        gap = min(max(abs(world.cam_x - bx) - hx, abs(world.cam_z - bz) - hz) for bx, bz, hx, hz, _, _ in tall)
        assert gap > 0.15, (world.cam_x, world.cam_z)
    assert not [e for e in events if e.outcome == "miss"]
    assert sum(e.mechanic == "knockdown" for e in events) >= 3
    assert len({e.target for e in events if e.outcome == "hit"}) >= len(world.layout.dummies) - 1


@pytest.mark.parametrize("name", ["LONG", "CORRIDOR_LONG", "ROOMS_LONG"])
def test_steady_camera_pans_slowly_keeps_its_lens_and_frames_every_hit(name):
    from gamewm_engine.games.training_ground import inputs

    script = getattr(inputs, name)
    world = World(24, "steady", script.layout)
    tall = [b for b in world.layout.blockers() if b[4] >= 2.0]
    views, cap = [], R.CAMERA_SHOTS["steady"].yaw_rate_deg
    for _ in range(719):
        events = world.step(script(world))
        cam = world.snapshot().camera
        (ex, _, ez), (tx, _, tz) = cam["eye"], cam["target"]
        views.append(math.atan2(tx - ex, tz - ez))
        assert cam["hfov_deg"] == R.CAMERA_HFOV_DEG
        for e in events:
            if e.outcome == "hit":
                dx, dz, _ = world.layout.dummies[e.target]
                off = (math.atan2(dx - ex, dz - ez) - views[-1] + math.pi) % (2 * math.pi) - math.pi
                assert abs(math.degrees(off)) < R.CAMERA_HFOV_DEG / 2, (e.t, e.target)
        if tall:
            gap = min(max(abs(world.cam_x - bx) - hx, abs(world.cam_z - bz) - hz) for bx, bz, hx, hz, _, _ in tall)
            assert gap > 0.15, (world.cam_x, world.cam_z)
    rates = sorted(abs(math.degrees((b - a + math.pi) % (2 * math.pi) - math.pi)) * 24 for a, b in zip(views, views[1:]))
    assert rates[int(0.95 * len(rates))] < cap + 5 and rates[-1] < 2 * cap


def test_courtyard_is_the_default_layout_and_others_move_the_start():
    from gamewm_engine.games.training_ground.layout import LAYOUTS

    assert World(24).layout is LAYOUTS["courtyard"] and World(24).snapshot().layout == "courtyard"
    rooms = World(24, layout="rooms")
    assert (rooms.x, rooms.z) == LAYOUTS["rooms"].start[:2]
    assert {d.id for d in rooms.dummies} == set(LAYOUTS["rooms"].dummies)
    with pytest.raises(ValueError):
        World(24, layout="moon")
    with pytest.raises(ValueError):
        Script([Face(0.0, 1.0, "dummy_center")], layout="rooms")


def test_walk_reaches_a_mark_through_the_gateway():
    from gamewm_engine.games.training_ground.inputs import Walk

    script = Script([Walk(0.0, 6.0, "gate_in", stop=0.3), Walk(6.0, 12.0, "gate_out")], layout="rooms")
    world = World(24, layout="rooms")
    for _ in range(12 * 24):
        world.step(script(world))
    assert math.dist((world.x, world.z), world.layout.point("gate_out")) < 0.5


def test_rooms_camera_is_pulled_in_front_of_the_divider():
    from gamewm_engine.games.training_ground.layout import LAYOUTS

    rooms = LAYOUTS["rooms"]
    assert rooms.sight(0.0, 12.5, 3.5, 7.0) < 1.0  # behind the right half of the divider
    assert rooms.sight(0.0, 12.5, 0.0, 7.0) == 1.0  # straight back through the gateway
    assert LAYOUTS["courtyard"].camera_collide is False


def test_layout_retext_rewrites_the_arena_and_keeps_the_rest():
    from gamewm_engine.games.training_ground.layout import LAYOUTS

    entry = json.loads((EXAMPLES / "roster_fx.json").read_text())["characters"][0]
    assert LAYOUTS["courtyard"].retext(entry) == entry
    corridor, rooms = LAYOUTS["corridor"].retext(entry), LAYOUTS["rooms"].retext(entry)
    assert "corridor" in corridor["setting"] and "six" in corridor["setting"]
    assert "gateway" in rooms["setting"] and "in each" in rooms["setting"]
    assert corridor["look"] == entry["look"] and "three grey dummies" not in corridor["props"]
    druid = json.loads((EXAMPLES / "roster_fx.json").read_text())["characters"][6]
    assert LAYOUTS["corridor"].retext(druid)["lighting"].endswith("lights the corridor")
    assert LAYOUTS["rooms"].retext(druid)["lighting"].endswith("lights both courtyards")


def test_card_entries_pass_and_text_that_breaks_a_layout_does_not():
    from gamewm_engine.games.training_ground import oc

    entries = json.loads((EXAMPLES / "roster_fx.json").read_text())["characters"]
    entries.append(json.loads((EXAMPLES / "oc_example.json").read_text())["entry"])
    for entry in entries:
        assert oc.validate(entry, entry["id"]) == [], entry["id"]
    assert all(oc.validate(c) for c in json.loads((EXAMPLES / "roster.json").read_text())["characters"])
    bad = json.loads(json.dumps(entries[-1]))
    bad["setting"] = bad["setting"].replace("three wooden training dummies", "wooden training dummies")
    bad["moves"]["idle"] = "stands in the middle of the courtyard"
    del bad["moves"]["turn_left"]
    del bad["skills"]["dash"]["phases"]["recover"]
    bad["skills"]["ranged_attack"]["phases"]["travel"] = "the lance streaks over the yard as a bright golden line"
    problems = oc.validate(bad, bad["id"])
    assert "setting: say 'three ... training dummies on round stands'" in problems
    assert any(p.startswith("setting: the corridor layout cannot rewrite it") for p in problems)
    assert any(p.startswith("moves.idle: do not name the courtyard") for p in problems)
    assert "moves.turn_left: missing" in problems and "dash.phases.recover: missing" in problems
    assert any(p.startswith("ranged_attack: name the arena only as") for p in problems)
