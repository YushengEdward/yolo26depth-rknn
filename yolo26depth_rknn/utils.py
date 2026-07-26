#!/usr/bin/env python3
"""Shared utilities for YOLO26-Depth RKNN inference."""
from __future__ import annotations

import re

import cv2
import numpy as np


def detect_imgsz_from_path(model_path: str) -> int:
    """Auto-detect input size from model filename.

    Examples:
        yolo26n-depth-float.rknn       -> 640
        yolo26n-depth_768-float.rknn   -> 768
        yolo26x-depth_1280.onnx        -> 1280
    """
    m = re.search(r'_(\d+)-', model_path)
    return int(m.group(1)) if m else 640


def detect_imgsz_from_onnx(onnx_path: str) -> int:
    """Read input size from ONNX model metadata."""
    import onnx
    m = onnx.load(onnx_path)
    shape = m.graph.input[0].type.tensor_type.shape.dim
    return int(shape[2].dim_value)  # H in N,C,H,W


def colorize_depth(depth: np.ndarray, mode: str = "disparity") -> np.ndarray:
    """Convert metric depth map to a BGR heatmap image.

    Args:
        depth: (H, W) float32 depth map in meters.
        mode: "disparity" (1/d with percentile clipping) or "metric" (linear).

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)

    if mode == "disparity":
        v = np.zeros_like(depth)
        v[valid] = 1.0 / np.maximum(depth[valid], 1e-6)
        lo, hi = np.percentile(v[valid], (2, 98))
    else:
        v = depth
        lo = float(depth[valid].min())
        hi = float(depth[valid].max())

    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((v - lo) / (hi - lo), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    heat = cv2.applyColorMap(idx, cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def save_outputs(depth: np.ndarray, image: np.ndarray, save_path: str | None = None,
                 save_depth: str | None = None, save_heat: str | None = None,
                 mode: str = "disparity", overlay_alpha: float = 0.5):
    """Save inference results (overlay, heatmap, raw depth).

    Args:
        depth: (H, W) float32 depth map.
        image: Original BGR image (H, W, 3) uint8.
        save_path: If set, save overlay (image + heatmap blended).
        save_depth: If set, save raw depth as .npy.
        save_heat:  If set, save standalone heatmap image.
        mode:       "disparity" or "metric" for colorization.
        overlay_alpha: Blend weight for heatmap (0-1).
    """
    if save_path or save_heat:
        heat = colorize_depth(depth, mode=mode)

    if save_heat:
        cv2.imwrite(save_heat, heat)
        print(f"  Heatmap saved: {save_heat}")

    if save_path:
        overlay = cv2.addWeighted(image, 1 - overlay_alpha, heat, overlay_alpha, 0)
        cv2.imwrite(save_path, overlay)
        print(f"  Overlay saved: {save_path}")

    if save_depth:
        np.save(save_depth, depth.astype(np.float32))
        print(f"  Depth saved: {save_depth}")
