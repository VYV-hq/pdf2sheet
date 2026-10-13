"""Tesseract wrappers: single-image OCR and stacked multi-cell OCR.

Stacked OCR concatenates many cell crops vertically (with white gaps) and runs
tesseract ONCE per column, then maps recognized words back to their source cell
via TSV y-coordinates. This is ~20x faster than per-cell process spawning and
robust against line-splitting mistakes.
"""
import os
import re
import subprocess
import tempfile
import unicodedata

import cv2
import numpy as np

from .config import TESSDATA_DIR

_ENV = dict(os.environ, TESSDATA_PREFIX=TESSDATA_DIR, OMP_THREAD_LIMIT="1")


def _run(img: np.ndarray, lang: str, psm: int, whitelist=None, timeout=180):
    """Run tesseract, return list of words [{text, conf, x, y, w, h}]."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        cv2.imwrite(path, img)
        cmd = ["tesseract", path, "stdout", "-l", lang, "--psm", str(psm)]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        cmd.append("tsv")
        out = subprocess.run(cmd, capture_output=True, text=True, env=_ENV, timeout=timeout)
        words = []
        for line in out.stdout.splitlines()[1:]:
            p = line.split("\t")
            if len(p) != 12 or p[0] != "5":
                continue
            txt = p[11].strip()
            if not txt:
                continue
            conf = float(p[10])
            if conf < 0:
                continue
            words.append({"text": txt, "conf": conf,
                          "x": int(p[6]), "y": int(p[7]),
                          "w": int(p[8]), "h": int(p[9])})
        return words
    finally:
        os.unlink(path)


def binarize(img, th=150):
    """Drop gray watermark/stamp ink; keep the dark printed glyphs."""
    if img is None:
        return img
    return np.where(img < th, 0, 255).astype(np.uint8)


def ocr_single(img, lang, psm=7, whitelist=None, scale=1.0, binarize_img=False):
    if binarize_img:
        img = binarize(img)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    # white margin so glyphs never touch the border
    img = cv2.copyMakeBorder(img, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255)
    words = _run(img, lang, psm, whitelist)
    line_h = max(int(20 * scale), int(img.shape[0] * 0.55))
    words.sort(key=lambda wd: (wd["y"] // line_h, wd["x"]))
    text = " ".join(w["text"] for w in words)
    conf = float(np.mean([w["conf"] for w in words])) if words else 0.0
    return text, conf


MAX_STACK_H = 1300   # tesseract mis-segments very tall blocks; keep chunks short
MARGIN_X = 20        # white margin so glyphs never touch the image border


def ocr_stack(cells, lang, whitelist=None, scale=1.0, gap=24, binarize_img=False):
    """OCR a list of cell crops (grayscale ndarrays, or None for blanks).

    Returns list of (text, conf) aligned with `cells`. Cells are stacked
    vertically into chunks of limited height (one tesseract call per chunk)
    and recognized words are mapped back to cells by their y coordinate.
    Low-confidence cells are retried individually.
    """
    live = [(i, c) for i, c in enumerate(cells) if c is not None]
    results = [("", 0.0)] * len(cells)
    if not live:
        return results

    imgs = []
    for _, c in live:
        if binarize_img:
            c = binarize(c)
        if scale != 1.0:
            c = cv2.resize(c, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imgs.append(c)
    g = int(gap * scale)

    # split into height-limited chunks
    chunks, cur, cur_h = [], [], 0
    for k, im in enumerate(imgs):
        cur.append(k)
        cur_h += im.shape[0] + g
        if cur_h >= MAX_STACK_H:
            chunks.append(cur)
            cur, cur_h = [], 0
    if cur:
        chunks.append(cur)

    for chunk in chunks:
        width = max(imgs[k].shape[1] for k in chunk) + 2 * MARGIN_X
        parts, spans, y = [], [], 0
        for k in chunk:
            im = imgs[k]
            canvas = np.full((im.shape[0], width), 255, np.uint8)
            canvas[:, MARGIN_X: MARGIN_X + im.shape[1]] = im
            parts.append(canvas)
            spans.append((y, y + im.shape[0]))
            y += im.shape[0]
            parts.append(np.full((g, width), 255, np.uint8))
            y += g
        stack = np.vstack(parts)

        words = _run(stack, lang, psm=6, whitelist=whitelist)

        buckets = [[] for _ in chunk]
        bounds = [s[0] - g / 2 for s in spans] + [spans[-1][1] + g / 2]
        for wd in words:
            yc = wd["y"] + wd["h"] / 2
            idx = int(np.searchsorted(bounds, yc) - 1)
            if 0 <= idx < len(buckets):
                buckets[idx].append(wd)

        for j, k in enumerate(chunk):
            ws = buckets[j]
            if not ws:
                continue
            span_h = spans[j][1] - spans[j][0]
            line_h = max(int(24 * scale), int(span_h * 0.55))
            ws.sort(key=lambda wd: ((wd["y"] - spans[j][0]) // line_h, wd["x"]))
            text = " ".join(w["text"] for w in ws)
            conf = float(np.mean([w["conf"] for w in ws]))
            results[live[k][0]] = (text, conf)

    # fallback: individually re-OCR low-confidence / empty results
    for k, (orig_i, cell) in enumerate(live):
        txt, conf = results[orig_i]
        if conf < 40:
            t2, c2 = ocr_single(cell, lang, psm=7, whitelist=whitelist, scale=scale,
                                binarize_img=binarize_img)
            if c2 > conf:
                results[orig_i] = (t2, c2)
    return results


# ---------------------------------------------------------------- vocab utils
def _jamo(s: str) -> str:
    return unicodedata.normalize("NFD", s)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def vocab_correct(text: str, vocab: list, max_ratio: float = 0.6):
    """Match OCR text against a vocabulary using jamo-level edit distance.

    Returns (corrected_text, distance) - corrected only when the best match is
    clearly closest and within budget; otherwise returns the raw text, -1.
    """
    t = text.replace(" ", "")
    if not t:
        return text, -1
    if t in vocab:
        return t, 0
    # OCR often turns latin prefixes (CD, ATM, KB) into symbol noise: drop it
    t = re.sub(r"[^0-9A-Za-z가-힣]", "", t) or t
    tj = _jamo(t)
    scored = sorted((_edit_distance(tj, _jamo(v)), v) for v in vocab)
    best_d, best_v = scored[0]
    budget = max(1, int(len(_jamo(best_v)) * max_ratio))
    if best_d <= budget:
        return best_v, best_d
    return text, -1
