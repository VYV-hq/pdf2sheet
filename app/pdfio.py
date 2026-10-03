"""PDF page -> grayscale image extraction with content-orientation handling.

Prefers the raw embedded scan image (full native resolution, skips watermark
overlays drawn as separate objects). Falls back to rendering the page at
RENDER_DPI. Some banks scan landscape statements into portrait pages, so the
content orientation is detected (fast ruled-line heuristic + tesseract OSD)
and the image is normalized upright.
"""
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np
import pymupdf

from .config import RENDER_DPI, TESSDATA_DIR


def page_gray(doc: "pymupdf.Document", pno: int, prefer=None, force=None):
    """Return (grayscale ndarray, applied np.rot90 k) for 0-based page `pno`.

    The image is orientation-normalized (rotated upright) and deskewed, so all
    downstream geometry is computed on a stable canonical image.

    prefer: rotation k to try first for rotated pages (worker-level cache).
    force:  apply exactly this k without any detection (used for re-OCR).
    """
    gray = _raw_gray(doc, pno)
    if force is not None:
        k = force % 4
    else:
        k = _detect_rotation(gray, prefer)
    if k:
        gray = np.rot90(gray, k).copy()
    gray = _deskew(gray)
    return gray, k


def _deskew(gray) -> np.ndarray:
    """Level sub-degree scan skew using the long ruled lines."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    mask = ((small < 200).astype(np.uint8)) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    lines = cv2.HoughLinesP(mask, 1, np.pi / 1800, threshold=300,
                            minLineLength=small.shape[1] // 3, maxLineGap=12)
    if lines is None:
        return gray
    angles = []
    for seg in lines.reshape(-1, 4):
        x1, y1, x2, y2 = (int(v) for v in seg)
        if x2 == x1:
            continue
        deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(deg) <= 3:
            angles.append(deg)
    if len(angles) < 2:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.08:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _raw_gray(doc, pno) -> np.ndarray:
    page = doc[pno]
    try:
        img = _largest_embedded(doc, page)
        if img is not None:
            return img
    except Exception:
        pass
    return _render(page)


def _largest_embedded(doc, page):
    infos = page.get_images(full=True)
    if not infos:
        return None
    xref, w, h = max(((i[0], i[2], i[3]) for i in infos), key=lambda t: t[1] * t[2])
    prect = page.rect
    page_px_area = (prect.width / 72 * RENDER_DPI) * (prect.height / 72 * RENDER_DPI)
    if w * h < page_px_area * 0.30:
        return None
    pix = pymupdf.Pixmap(doc, xref)
    if pix.colorspace is None or pix.n - pix.alpha not in (1, 3):
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
    if arr.shape[2] == 3:
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).astype(np.uint8)
    else:
        gray = arr[:, :, 0].copy()
    return gray


def _render(page) -> np.ndarray:
    mat = pymupdf.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csGRAY, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


# ------------------------------------------------------------- orientation
def _line_ink(gray):
    """Ink pixel counts of long horizontal / vertical ruled lines.

    Runs at full resolution with a lenient threshold (faint gray rules would
    vanish when average-downscaled) and closes across the line's thin axis
    first so sub-degree skew does not break the long runs.
    """
    bw = ((gray < 210).astype(np.uint8)) * 255
    h, w = bw.shape
    # close along both axes: bridges sub-degree skew drift AND dashed/faded
    # line segments; text cannot form such long continuous runs regardless
    horiz = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (13, 9)))
    horiz = cv2.morphologyEx(horiz, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (max(60, w // 12), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 13)))
    vert = cv2.morphologyEx(vert, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(60, h // 12))))
    return int(horiz.sum() // 255), int(vert.sum() // 255)


def _detect_rotation(gray, prefer=None) -> int:
    hh, vv = _line_ink(gray)
    if hh == 0 and vv == 0:
        # no ruled lines at all: fall back to text-orientation detection
        small = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
        deg = _osd_rotate_deg(small)
        return {0: 0, 90: 3, 180: 2, 270: 1}.get(deg, 0)
    if hh >= vv:
        return 0                     # upright ruled lines
    # content is rotated 90 degrees; decide direction
    if prefer in (1, 3):
        return prefer
    small = cv2.resize(gray, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    deg = _osd_rotate_deg(small)
    return {0: 0, 90: 3, 180: 2, 270: 1}.get(deg, 1)


def _osd_rotate_deg(img) -> int:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        env = dict(os.environ, TESSDATA_PREFIX=TESSDATA_DIR, OMP_THREAD_LIMIT="1")
        out = subprocess.run(["tesseract", path, "stdout", "--psm", "0", "-l", "osd"],
                             capture_output=True, text=True, env=env, timeout=60)
        m = re.search(r"Rotate:\s*(\d+)", out.stdout)
        return int(m.group(1)) if m else 0
    except Exception:
        return 0
    finally:
        os.unlink(path)
