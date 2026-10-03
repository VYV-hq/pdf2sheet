"""Table-grid detection: find bordered cells and group them into row clusters."""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RowCluster:
    y0: int
    y1: int
    cells: list = field(default_factory=list)  # [(x0, x1), ...] sorted by x


def find_row_clusters(gray: np.ndarray):
    """Detect bordered table cells; return list[RowCluster] sorted by y."""
    h, w = gray.shape
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 15, 10)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 70))))
    mask = cv2.dilate(horiz | vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    # cell interiors = holes enclosed by the line mask
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    boxes = []
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] == -1:      # outer contour -> not a cell interior
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 25 or ch < 18 or cw > w * 0.98 or ch > h * 0.5:
            continue
        boxes.append((x, y, cw, ch))

    if not boxes:
        return []

    # cluster boxes into rows by y-center
    boxes.sort(key=lambda b: (b[1], b[0]))
    clusters: list[RowCluster] = []
    for (x, y, cw, ch) in boxes:
        yc = y + ch / 2
        placed = False
        for rc in clusters:
            if rc.y0 - 5 <= yc <= rc.y1 + 5:
                rc.y0 = min(rc.y0, y)
                rc.y1 = max(rc.y1, y + ch)
                rc.cells.append((x, x + cw))
                placed = True
                break
        if not placed:
            clusters.append(RowCluster(y0=y, y1=y + ch, cells=[(x, x + cw)]))

    for rc in clusters:
        rc.cells.sort()
        rc.cells = _merge_overlaps(rc.cells)
    clusters.sort(key=lambda rc: rc.y0)
    return clusters


def _merge_overlaps(spans):
    """Merge x-spans that overlap by more than half (duplicate contours)."""
    out = []
    for (x0, x1) in spans:
        if out and x0 < out[-1][1] - (x1 - x0) * 0.5:
            out[-1] = (out[-1][0], max(out[-1][1], x1))
        else:
            out.append((x0, x1))
    return out


def match_ratios(cells, ratios, tol):
    """True when a row's cell boundaries match the expected layout ratios."""
    if len(cells) != len(ratios) - 1:
        return False
    x_start, x_end = cells[0][0], cells[-1][1]
    width = x_end - x_start
    if width <= 0:
        return False
    for i, (x0, x1) in enumerate(cells):
        r0 = (x0 - x_start) / width
        r1 = (x1 - x_start) / width
        if abs(r0 - ratios[i]) > tol or abs(r1 - ratios[i + 1]) > tol:
            return False
    return True
