"""Authored-once assets: from a character sheet and the engine's opening view to an asset pack.

The VM needs three things that are made once per character and scene, not per take:
``<Picture 1>`` (the first frame, painted over the engine's opening camera view, objects exactly
where proxy frame 0 has them), ``<Picture 2>`` (the identity sheet) and the text of the skill
tree. Painting a first frame straight from a sheet makes image models enlarge the character, so
the pipeline is:

1. render the opening view twice, with and without the player (the engine knows where the player is);
2. cut the sheet's back view out and paste it over the player's box, feet on the player's feet;
3. restyle the collage with an image model, keeping the composition (``restyle_prompt``);
4. review the painting against the proxy's class boundaries (``review_panel``) and redo if it drifts;
5. write the asset pack (``pack.json``) that the engine loads at run time.

Steps 1, 2, 4 and 5 are here; step 3 is whatever image model the studio uses.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PACK_SCHEMA = "gamewm.asset-pack.v1"


# -- sheet cut-out ----------------------------------------------------------------------------------


def cutout(sheet: Image.Image, view: int = 2, views: int = 3) -> Image.Image:
    """The ``view``-th of ``views`` side-by-side figures on a plain backdrop, as a tight RGBA crop.

    Turnaround sheets are front / side / back, so ``view=2`` is the back, which is what a follow
    camera sees in the opening frame.
    """
    import cv2

    rgb = np.asarray(sheet.convert("RGB"))
    w = rgb.shape[1]
    panel = np.ascontiguousarray(rgb[:, w * view // views:w * (view + 1) // views])
    h, pw = panel.shape[:2]
    border = np.concatenate([panel[:6].reshape(-1, 3), panel[-6:].reshape(-1, 3), panel[:, :6].reshape(-1, 3),
                             panel[:, -6:].reshape(-1, 3)])
    backdrop = np.median(border, axis=0)
    dist = np.abs(panel.astype(np.float32) - backdrop).max(axis=-1)
    mask = np.where(dist > 22, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    mask[dist > 60] = cv2.GC_FGD
    mask[:4], mask[-4:], mask[:, :4], mask[:, -4:] = (cv2.GC_BGD, ) * 4
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    alpha = np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.nonzero(areas >= 0.01 * areas.max())[0]
        alpha = np.isin(labels, keep).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    soft = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)
    ys, xs = np.nonzero(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([panel, (soft * 255).astype(np.uint8)])[y0:y1, x0:x1]
    return Image.fromarray(np.ascontiguousarray(rgba))


def body_extent(alpha: np.ndarray, min_width: float = 0.12) -> tuple[int, int, float]:
    """``(top, bottom, centre x)`` of the body, ignoring thin things that stick out (a staff tip
    above the head, a rifle barrel): rows narrower than ``min_width`` of the widest row don't count."""
    solid = alpha > 127
    widths = solid.sum(axis=1)
    body = np.nonzero(widths >= min_width * widths.max())[0]
    top, bottom = int(body.min()), int(np.nonzero(widths)[0].max())
    rows = solid[body]
    xs = np.nonzero(rows)[1]
    return top, bottom, float(np.median(xs))


def paste(background: Image.Image, figure: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Scale ``figure`` (RGBA) so its body spans the height of ``box`` and stand it on the box's
    bottom edge, centred on the box."""
    x0, y0, x1, y1 = box
    alpha = np.asarray(figure)[..., 3]
    top, bottom, cx = body_extent(alpha)
    scale = (y1 - y0) / max(1, bottom - top)
    size = (max(1, round(figure.width * scale)), max(1, round(figure.height * scale)))
    scaled = figure.resize(size, Image.LANCZOS)
    left = round((x0 + x1) / 2 - cx * scale)
    up = round(y1 - bottom * scale)
    out = background.convert("RGB").copy()
    out.paste(scaled, (left, up), scaled)
    return out


# -- review ---------------------------------------------------------------------------------------


def boundaries(ids: np.ndarray) -> np.ndarray:
    """Pixels where the class changes: the proxy's layout as a line drawing."""
    edge = np.zeros(ids.shape, bool)
    edge[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edge[1:] |= ids[1:] != ids[:-1]
    return edge


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Scale to cover ``size`` and centre-crop, the transform the VM applies to ``<Picture 1>``."""
    w, h = size
    scale = max(w / image.width, h / image.height)
    resized = image.convert("RGB").resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    left, top = (resized.width - w) // 2, (resized.height - h) // 2
    return resized.crop((left, top, left + w, top + h))


def review_panel(proxy: Image.Image, painted: Image.Image, ids: np.ndarray, box: tuple[int, int, int, int],
                 sheet: Image.Image | None = None, width: int = 896) -> Image.Image:
    """A 2x2 grid: proxy, painting / painting with the proxy's class boundaries (red) and player
    box (blue), sheet."""
    size = proxy.size
    painted = cover(painted, size)
    overlay = np.asarray(painted).copy()
    edge = boundaries(ids)
    edge[1:] |= edge[:-1].copy()
    edge[:, 1:] |= edge[:, :-1].copy()
    overlay[edge] = (255, 40, 40)
    over = Image.fromarray(overlay)
    ImageDraw.Draw(over).rectangle(box, outline=(40, 160, 255), width=3)
    tiles = [proxy.convert("RGB"), painted, over, sheet.convert("RGB") if sheet is not None else proxy.convert("RGB")]
    height = round(width * size[1] / size[0])
    tiles = [cover(t, (width, height)) if i < 3 else t.resize((width, round(t.height * width / t.width)))
             for i, t in enumerate(tiles)]
    out = Image.new("RGB", (2 * width + 8, 2 * height + 8), (20, 20, 20))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % 2) * (width + 8), (i // 2) * (height + 8)))
    return out


# -- restyle prompt -------------------------------------------------------------------------------


def restyle_prompt(setting: str, lighting: str, look: str, props: str) -> str:
    """The image-model instruction for step 3; ``props`` says what the grey boxes become."""
    return (
        "Restyle this grey-box 3D game frame into a finished, high-quality third-person video game screenshot. "
        "Keep the composition exactly: the same camera, horizon, perspective, and every object at the same "
        f"position and size. The scene is {setting}. {lighting}. {props} "
        f"The figure in the lower middle is the player character seen from behind, {look}: keep it at exactly "
        "the same position, size and pose, standing on the ground, only blend its lighting into the scene. "
        "Do not add people, creatures, text, logos or UI. Keep the open sky band above the walls.")


# -- asset pack -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pack:
    """A loaded asset pack: what ``TrainingGround(character=..., scene=..., first_frame=...)`` takes."""

    id: str
    root: Path
    scene: dict[str, str]
    character: dict[str, Any]
    first_frame: Path | None
    meta: dict[str, Any]


def write_pack(root: str | Path, entry_id: str, scene: dict[str, str], character: dict[str, Any], sheet: Path,
               authoring: dict[str, Any], game: str) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sheet, root / f"sheet{sheet.suffix.lower()}")
    payload = {
        "schema": PACK_SCHEMA,
        "id": entry_id,
        "game": game,
        "first_frame": "first_frame.png",
        "scene": scene,
        "character": {**character, "sheet": f"sheet{sheet.suffix.lower()}"},
        "authoring": authoring,
    }
    path = root / "pack.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_pack(path: str | Path, require_first_frame: bool = True) -> Pack:
    path = Path(path)
    if path.is_dir():
        path = path / "pack.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("schema") != PACK_SCHEMA:
        raise ValueError(f"{path}: schema must be {PACK_SCHEMA!r}")
    root = path.parent.resolve()
    character = dict(row["character"])
    if character.get("sheet"):
        character["sheet"] = str(root / character["sheet"])
    first = root / row["first_frame"]
    if require_first_frame and not first.is_file():
        raise FileNotFoundError(f"{row['id']}: no painted first frame at {first}")
    return Pack(row["id"], root, dict(row["scene"]), character, first if first.is_file() else None, row)
