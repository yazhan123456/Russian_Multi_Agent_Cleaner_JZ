from __future__ import annotations

from typing import Iterable

import numpy as np


def build_sanitized_page(
    image: np.ndarray,
    routed_blocks: Iterable[dict],
    *,
    mask_fill: int = 255,
) -> np.ndarray:
    sanitized = np.array(image, copy=True)
    blocks = list(routed_blocks)
    keep_rects = [
        _clip_bbox(sanitized, block.get("bbox") or [0, 0, 0, 0])
        for block in blocks
        if str(block.get("action") or "") == "keep"
    ]
    for block in blocks:
        if str(block.get("action") or "") != "mask":
            continue
        mask_rects = [_clip_bbox(sanitized, block.get("bbox") or [0, 0, 0, 0])]
        for keep_rect in keep_rects:
            next_rects: list[tuple[int, int, int, int]] = []
            for mask_rect in mask_rects:
                next_rects.extend(_subtract_rect(mask_rect, keep_rect))
            mask_rects = next_rects
            if not mask_rects:
                break
        for x1, y1, x2, y2 in mask_rects:
            sanitized[y1:y2, x1:x2] = mask_fill
    return sanitized


def _clip_bbox(image: np.ndarray, bbox: list[int]) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _subtract_rect(
    base: tuple[int, int, int, int],
    cut: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    bx1, by1, bx2, by2 = base
    cx1, cy1, cx2, cy2 = cut

    ix1 = max(bx1, cx1)
    iy1 = max(by1, cy1)
    ix2 = min(bx2, cx2)
    iy2 = min(by2, cy2)

    if ix1 >= ix2 or iy1 >= iy2:
        return [base]

    fragments: list[tuple[int, int, int, int]] = []
    if by1 < iy1:
        fragments.append((bx1, by1, bx2, iy1))
    if iy2 < by2:
        fragments.append((bx1, iy2, bx2, by2))
    if bx1 < ix1:
        fragments.append((bx1, iy1, ix1, iy2))
    if ix2 < bx2:
        fragments.append((ix2, iy1, bx2, iy2))
    return [rect for rect in fragments if rect[0] < rect[2] and rect[1] < rect[3]]
