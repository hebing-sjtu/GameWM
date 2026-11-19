import copy
import json
from pathlib import Path

import pytest

from gamewm_engine import ProtocolError, parse, plan
from gamewm_engine.compose import SPAN, segment_count
from gamewm_engine.protocol import SHEET_PROMPTS

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples" / "seg0162_fire_mage.json").read_text(encoding="utf-8"))


def request(**take):
    payload = copy.deepcopy(EXAMPLE)
    payload["take"].update(take)
    return parse(payload, check_files=False)


def test_example_plans_one_segment_with_the_skill_in_its_caption():
    layout = plan(request())
    assert layout.output_frames == 124
    (segment, ) = layout.segments
    assert segment.role == "w0_omni" and (segment.start, segment.end) == (0, 124)
    assert segment.caption.startswith(f"{SPAN} A 3D video game captures a curving sandstone-paved road")
    assert "The protagonist from the identity sheet, a young dark-haired man" in segment.caption
    assert "hurls a fireball that bursts into flames on the hooded cultist in a black robe;" in segment.caption
    assert segment.caption.endswith("stays down. The camera follows from behind as the road bends to the right.")
    assert segment.document.split("\n") == [
        "Reference modality: style video.", SHEET_PROMPTS["protagonist"], segment.caption]


def test_without_a_sheet_the_identity_line_is_dropped():
    payload = copy.deepcopy(EXAMPLE)
    del payload["assets"]["character"]["sheet"]
    (segment, ) = plan(parse(payload, check_files=False)).segments
    assert "<Picture 2>" not in segment.document
    assert ", The protagonist, a young" not in segment.caption and "The protagonist, a young" in segment.caption


def test_long_takes_overlap_by_39_frames():
    r = request(seconds=None)
    assert segment_count(124, r) == (1, 124)
    assert segment_count(125, r) == (2, 125)
    assert segment_count(294, r) == (3, 294)
    assert segment_count(295, r) == (4, 295)
    layout = plan(request(seconds=None, action=["{skill:ranged_attack}", "walks on", "stops"]), 294)
    assert [(s.start, s.end, s.role) for s in layout.segments] == [
        (0, 124, "w0_omni"), (85, 209, "wn_omni"), (170, 294, "wn_omni")]


def test_a_skill_needs_its_event_inside_the_segment():
    with pytest.raises(ProtocolError, match="no ranged_attack event"):
        plan(request(seconds=None, action=["walks", "{skill:ranged_attack}"]), 200)


def test_unknown_references_are_rejected():
    with pytest.raises(ProtocolError, match="no such skill"):
        request(action="then {skill:teleport}")
    with pytest.raises(ProtocolError, match="not in take.entities"):
        request(events=[{"t": 1.0, "mechanic": "ranged_attack", "actor": "player", "target": "ghost"}])
    with pytest.raises(ProtocolError, match="modality"):
        request(proxy={"video": "x.mp4", "modality": "lowpoly"})


def test_an_aimed_skill_takes_its_target_from_the_matching_event():
    entities = {"cultist": "the hooded cultist", "wall": "the courtyard wall"}
    events = [{"t": 1.0, "mechanic": "ranged_attack", "actor": "player", "target": "wall", "outcome": "miss",
               "detail": {"frame": 24}},
              {"t": 2.0, "mechanic": "ranged_attack", "actor": "player", "target": "cultist", "outcome": "hit"}]
    r = request(entities=entities, events=events,
                action="{skill:ranged_attack>wall}, then {skill:ranged_attack>cultist}")
    assert r.events[0].detail == {"frame": 24}
    (segment, ) = plan(r).segments
    assert "bursts into flames on the courtyard wall, then raises" in segment.caption
    assert "bursts into flames on the hooded cultist again." in segment.caption
    with pytest.raises(ProtocolError, match="not in take.entities"):
        request(entities=entities, events=events, action="{skill:ranged_attack>ghost}")
    with pytest.raises(ProtocolError, match="no ranged_attack event at crate"):
        plan(request(entities={**entities, "crate": "a crate"}, events=events,
                     action="{skill:ranged_attack>crate}"))


def test_moves_and_repeats_read_in_the_characters_own_words():
    payload = copy.deepcopy(EXAMPLE)
    char = payload["assets"]["character"]
    char["skills"]["ranged_attack"]["again"] = "hurls another fireball at {target}"
    char["moves"] = {"walk_to": "strides toward {target}, coat swinging"}
    events = [{"t": 1.0, "mechanic": "ranged_attack", "actor": "player", "target": "cultist", "outcome": "hit"},
              {"t": 2.0, "mechanic": "ranged_attack", "actor": "player", "target": "cultist", "outcome": "hit"}]
    payload["take"].update(events=events, action=("{move:idle}, then {move:walk_to>cultist}, then "
                                                  "{skill:ranged_attack}, then {skill:ranged_attack}"))
    (segment, ) = plan(parse(payload, check_files=False)).segments
    cultist = payload["take"]["entities"]["cultist"]
    assert f"stands still, then strides toward {cultist}, coat swinging, then raises" in segment.caption
    assert f"then hurls another fireball at {cultist}." in segment.caption
    payload["take"]["action"] = "{move:walk_to}"
    with pytest.raises(ProtocolError, match="needs a target"):
        parse(payload, check_files=False)
    char["moves"] = {"swim": "swims"}
    with pytest.raises(ProtocolError, match="unknown kinds"):
        parse(payload, check_files=False)


def test_skill_phases_and_pronouns_fill_sentence_by_sentence():
    payload = copy.deepcopy(EXAMPLE)
    char = payload["assets"]["character"]
    char["pronoun"] = "she"
    char["skills"]["ranged_attack"]["phases"] = {"windup": "raises her hand toward {target}",
                                                 "impact": "it bursts on {target}"}
    events = [{"t": 1.0, "mechanic": "ranged_attack", "actor": "player", "target": "cultist", "outcome": "hit"}]
    payload["take"].update(events=events, action=(
        "{move:walk}. As the camera pushes in close behind {him}, {he} {skill:ranged_attack.windup>cultist}. "
        "{skill:ranged_attack.impact>cultist}. {His} {skill:ranged_attack.name} lands."))
    (segment, ) = plan(parse(payload, check_files=False)).segments
    cultist = payload["take"]["entities"]["cultist"]
    name = char["skills"]["ranged_attack"]["name"]
    assert f"behind her, she raises her hand toward {cultist}. It bursts on {cultist}. Her {name} lands." \
        in segment.caption
    char["skills"]["ranged_attack"]["phases"] = {"charge": "x"}
    with pytest.raises(ProtocolError, match="phase"):
        parse(payload, check_files=False)
    char["skills"]["ranged_attack"]["phases"] = {}
    char["pronoun"] = "xe"
    with pytest.raises(ProtocolError, match="pronoun"):
        parse(payload, check_files=False)
    char["pronoun"] = "she"
    payload["take"]["action"] = "{move:walk.windup}"
    with pytest.raises(ProtocolError):
        parse(payload, check_files=False)


def test_an_event_on_a_segment_boundary_lands_in_the_later_segment():
    events = [{"t": round(170 / 24, 6), "mechanic": "ranged_attack", "actor": "player", "target": "cultist",
               "outcome": "hit"}]
    layout = plan(request(seconds=None, events=events, action=["walks", "walks", "{skill:ranged_attack}"]), 294)
    assert [len(s.events) for s in layout.segments] == [0, 1, 1]


def test_wan_models_take_one_clip():
    payload = copy.deepcopy(EXAMPLE)
    payload["model"] = "wan2.2-14b"
    r = parse(payload, check_files=False)
    assert segment_count(200, r) == (1, 81)
    assert plan(r, 200).output_frames == 81


def test_schema_accepts_the_example():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "protocol" / "request.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(EXAMPLE, schema)
