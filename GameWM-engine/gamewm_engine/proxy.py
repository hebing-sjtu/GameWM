"""Turn an engine's depth and class-id buffers into the VM's proxy modalities.

The constants are FastVideo's (``fastvideo/pipelines/basic/minimax_h3/proxy.py``), which encoded
every training proxy; an engine that writes these exact codes needs no estimation step.
"""

from __future__ import annotations

import math

import numpy as np

DEPTH_NEAR_METRES = 0.3
DEPTH_FAR_METRES = 256.0
DEPTH_VALID_EPSILON_METRES = 1.0e-3

CLASSES = (
    "void_unknown", "sky", "water", "terrain", "road_paved", "vegetation", "building_structure", "infrastructure",
    "human", "animal", "vehicle", "prop",
)
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}

# DUV: classes on a 4x3 grid of the (G, B) square; separate semantic video: a 3x2x2 RGB lattice.
DUV_U = (32, 96, 160, 224)
DUV_V = (43, 128, 213)
SEMANTIC_R = (43, 128, 213)
SEMANTIC_G = (64, 192)
SEMANTIC_B = (64, 192)


def depth_unit(metric: np.ndarray) -> np.ndarray:
    """Log-normalized depth in ``[0, 1]``, near bright; zero where there is no surface."""

    metric = np.asarray(metric, dtype=np.float64)
    out = np.zeros(metric.shape, dtype=np.float32)
    valid = np.isfinite(metric) & (metric > DEPTH_VALID_EPSILON_METRES)
    if valid.any():
        clipped = np.clip(metric[valid], DEPTH_NEAR_METRES, DEPTH_FAR_METRES)
        span = math.log(DEPTH_FAR_METRES) - math.log(DEPTH_NEAR_METRES)
        out[valid] = np.clip((math.log(DEPTH_FAR_METRES) - np.log(clipped)) / span, 0.0, 1.0)
    return out


def semantic_palette() -> np.ndarray:
    ids = np.arange(len(CLASSES))
    return np.stack((np.asarray(SEMANTIC_R)[ids % 3], np.asarray(SEMANTIC_G)[(ids // 3) % 2],
                     np.asarray(SEMANTIC_B)[ids // 6]), axis=-1).astype(np.uint8)


def _byte(unit: np.ndarray) -> np.ndarray:
    return np.clip(np.round(unit * 255.0), 0, 255).astype(np.uint8)


def depth_frame(metric: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` uint8 grey depth."""
    return np.repeat(_byte(depth_unit(metric))[..., None], 3, axis=-1)


def semantic_frame(ids: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` uint8 flat class colours."""
    return semantic_palette()[np.asarray(ids, dtype=np.int64)]


def duv_frame(metric: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """``[H, W, 3]`` uint8 packed depth (R) and class codes (G, B)."""
    ids = np.asarray(ids, dtype=np.int64)
    u = np.asarray(DUV_U, dtype=np.uint8)[ids % len(DUV_U)]
    v = np.asarray(DUV_V, dtype=np.uint8)[ids // len(DUV_U)]
    return np.stack((_byte(depth_unit(metric)), u, v), axis=-1)
