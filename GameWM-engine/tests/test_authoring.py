import json

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from gamewm_engine import authoring  # noqa: E402
from gamewm_engine.games.training_ground import TrainingGround  # noqa: E402


def figure(height=200, width=60, staff=True):
    """A blocky RGBA person: a body, a narrower head, and a thin staff above the head."""
    a = np.zeros((height + 40, width + 20, 4), np.uint8)
    a[40:height + 40, 10:width + 10] = (200, 30, 30, 255)
    a[40:70, 25:width - 5] = (230, 200, 180, 255)
    if staff:
        a[0:height + 40, width + 14:width + 16] = (120, 80, 40, 255)
    return Image.fromarray(a)


def test_body_extent_ignores_a_staff_above_the_head():
    top, bottom, _ = authoring.body_extent(np.asarray(figure())[..., 3])
    assert (top, bottom) == (40, 239)


def test_paste_stands_the_body_on_the_box():
    out = authoring.paste(Image.new("RGB", (400, 300), (128, 128, 128)), figure(staff=False), (150, 100, 200, 250))
    ys, xs = np.nonzero(np.abs(np.asarray(out).astype(int) - 128).max(axis=-1) > 40)
    assert abs(ys.max() - 249) <= 1 and abs(ys.min() - 100) <= 1
    assert abs((xs.min() + xs.max()) / 2 - 175) <= 1.5


def test_cutout_takes_the_back_view_off_a_plain_backdrop():
    pytest.importorskip("cv2")
    sheet = np.full((300, 600, 3), 225, np.uint8)
    for k in range(3):
        sheet[60:280, k * 200 + 80:k * 200 + 120] = (40 * k + 20, 60, 160)
    cut = authoring.cutout(Image.fromarray(sheet))
    alpha = np.asarray(cut)[..., 3]
    assert cut.size == (40, 220) and (alpha > 127).mean() > 0.95
    assert tuple(np.asarray(cut)[110, 20, :3]) == (100, 60, 160)


def test_opening_view_and_pack_round_trip(tmp_path):
    game = TrainingGround("h3-33b")
    view = game.opening(224, 128, supersample=1)
    x0, y0, x1, y1 = view["player_box"]
    assert 0 < x0 < x1 < 224 and 0 < y0 < y1 <= 128
    assert (view["style"] != view["empty"]).any(axis=-1)[y0:y1, x0:x1].mean() > 0.3
    sheet = tmp_path / "sheet.jpg"
    Image.new("RGB", (30, 10)).save(sheet)
    character = {"subject": "protagonist", "look": "a knight", "skills": {"dash": {"name": "D", "text": "dashes"}}}
    path = authoring.write_pack(tmp_path / "knight", "knight", {"setting": "s", "lighting": "l"}, character, sheet,
                                {"player_box": [x0, y0, x1, y1]}, "training_ground")
    assert json.loads(path.read_text())["character"]["sheet"] == "sheet.jpg"
    with pytest.raises(FileNotFoundError):
        authoring.load_pack(path)
    Image.new("RGB", (16, 9)).save(tmp_path / "knight" / "first_frame.png")
    pack = authoring.load_pack(tmp_path / "knight")
    assert pack.character["sheet"] == str((tmp_path / "knight" / "sheet.jpg").resolve())
    assert pack.first_frame.name == "first_frame.png"
