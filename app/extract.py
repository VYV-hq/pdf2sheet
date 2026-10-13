"""Per-page extraction: classify the page against bank profiles and pull
structured transaction rows into a unified 9-column schema.

Two table detectors:
  bordered   - full ruled grid (KB): contour cells matched by column ratios.
  headerline - borderless table (IBK): column bands located via the printed
               column-header words, row bands via ink projection.
"""
import re
import unicodedata

import cv2
import numpy as np

from .config import (BLANK_INK_PX, CELL_INSET, INK_THRESHOLD, PROFILES,
                     STATE_VOCAB)
from .grid import RowCluster, find_row_clusters, match_ratios
from .ocr import _run, ocr_single, ocr_stack, vocab_correct

DATE_RE = re.compile(r"^\d{4}/\d{2}/\d{2}")

UNIFIED_KEYS = ["거래일시", "상태", "거래구분", "거래금액", "표면잔액",
                "취급점", "적요(의뢰인등)", "수표번호", "이체상대계좌번호"]


def crop(gray, y0, y1, x0, x1, inset=CELL_INSET):
    return gray[y0 + inset: y1 - inset, x0 + inset: x1 - inset]


def is_blank(cell) -> bool:
    if cell is None or cell.size == 0:
        return True
    return int((cell < INK_THRESHOLD).sum()) < BLANK_INK_PX


def _valid_date(s: str) -> bool:
    m = DATE_RE.match(s or "")
    if not m:
        return False
    y, mo, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
    return 1990 <= y <= 2050 and 1 <= mo <= 12 and 1 <= d <= 31


# =============================================================== entry point
def process_page(gray: np.ndarray, pno: int, rot: int = 0) -> dict:
    """Extract structured content from one (orientation-normalized) page."""
    clusters = find_row_clusters(gray)

    # 1) bordered profiles (ruled grid)
    for name, prof in PROFILES.items():
        if prof["table"] != "bordered":
            continue
        result = _process_bordered(gray, pno, clusters, name, prof)
        if result is not None:
            result["rot"] = rot
            return result

    # 2) header-line profiles (borderless)
    for name, prof in PROFILES.items():
        if prof["table"] != "headerline":
            continue
        result = _process_headerline(gray, pno, name, prof)
        if result is not None:
            result["rot"] = rot
            return result

    result = _other_page(gray, pno)
    result["rot"] = rot
    return result


def _roles_map(prof):
    return {spec["role"]: i for i, spec in enumerate(prof["columns"])
            if spec["role"] != "skip"}


# ========================================================== bordered tables
def _process_bordered(gray, pno, clusters, name, prof):
    txn_rows = [rc for rc in clusters
                if match_ratios(rc.cells, prof["ratios"], prof["tol"])]
    if not txn_rows:
        return None

    header = _parse_labeled_header(gray, clusters, txn_rows[0].y0, prof)

    data_rows = txn_rows[1:] if len(txn_rows) > 1 else []
    if txn_rows and data_rows:
        c0 = crop(gray, txn_rows[0].y0, txn_rows[0].y1, *txn_rows[0].cells[0])
        if not is_blank(c0):
            t, _ = ocr_single(c0, "eng", psm=7, whitelist="0123456789/: ")
            if DATE_RE.match(t.replace(" ", "")[:10]):
                data_rows = txn_rows
    if not data_rows:
        return {"page": pno + 1, "kind": "other", "profile": name,
                "header": header, "rows": [], "confs": [], "geom": None,
                "text": ""}

    data_rows = _recover_and_split_rows(clusters, txn_rows[0], data_rows)
    return _ocr_table(gray, pno, name, prof, header,
                      [(rc.y0, rc.y1) for rc in data_rows],
                      [rc.cells for rc in data_rows])


# ======================================================== borderless tables
def _process_headerline(gray, pno, name, prof):
    found = _find_header_columns(gray, prof)
    if found is None:
        return None
    header_top, header_y1, pos, bands = found
    w = gray.shape[1]
    if not bands:
        return None

    # column projection over TEXT rows only: excludes the faint ruled
    # separator lines, so real column gaps project to (near) zero ink
    inkcol = np.zeros(w, dtype=np.int64)
    for (a, b) in bands:
        seg = gray[a + 4: b - 4, :]
        if seg.size:
            inkcol += (seg < INK_THRESHOLD).sum(axis=0)

    col_edges = _column_edges(pos, inkcol, w)
    if col_edges is None:
        return None

    header = _parse_strip_header(gray, header_top, prof)
    cells_per_row = [[(col_edges[c], col_edges[c + 1])
                      for c in range(len(col_edges) - 1)] for _ in bands]
    return _ocr_table(gray, pno, name, prof, header, bands, cells_per_row)


def _find_header_columns(gray, prof):
    """Locate the column-header text band (no dependence on ruled lines).

    Scans the top row-bands of the page, OCRs each and fuzzy-matches the
    expected column header words. Returns
    (header_top, header_bottom, anchor_positions, data_bands)."""
    expected = prof["header_words"]
    h, w = gray.shape
    bands = _row_bands(gray, 0, 0, w, h - 70)
    if not bands:
        return None
    for bi, (a, b) in enumerate(bands[:14]):
        if not (26 < b - a < 130):
            continue
        strip = gray[max(0, a): b, :]
        up = cv2.resize(strip, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        words = _run(up, "kor+eng", psm=6)
        for wd in words:
            wd["x"] = wd["x"] / 1.5
            wd["w"] = wd["w"] / 1.5
        anchors = _match_header_words(words, expected)
        if len(anchors) < prof["min_header_hits"]:
            continue
        pos = _interp_anchors(anchors, len(expected))
        if pos is None:
            continue
        return a, b, pos, bands[bi + 1:]
    return None


def _interp_anchors(anchors, n):
    pos = [None] * n
    for ei, cx in anchors.items():
        pos[ei] = cx
    known = [i for i, p in enumerate(pos) if p is not None]
    if not known:
        return None
    for i in range(n):
        if pos[i] is None:
            left = max((k for k in known if k < i), default=None)
            right = min((k for k in known if k > i), default=None)
            if left is None:
                pos[i] = pos[right] - (right - i) * 120
            elif right is None:
                pos[i] = pos[left] + (i - left) * 120
            else:
                pos[i] = pos[left] + (pos[right] - pos[left]) * (i - left) / (right - left)
    return pos


def _match_header_words(words, expected):
    """Fuzzy-match OCR words (plus merged neighbours) to expected headers.
    Returns {header_index: center_x}."""
    cands = [(wd["text"], wd["x"] + wd["w"] / 2) for wd in words]
    ws = sorted(words, key=lambda t: t["x"])
    for a, b in zip(ws, ws[1:]):
        if 0 < b["x"] - (a["x"] + a["w"]) < 40:
            cands.append((a["text"] + b["text"],
                          (a["x"] + b["x"] + b["w"]) / 2))
    anchors = {}
    for ei, exp in enumerate(expected):
        best = None
        for text, cx in cands:
            t = re.sub(r"[^0-9A-Za-z가-힣]", "", text)
            if not t:
                continue
            _, d = vocab_correct(t, [exp], max_ratio=0.5)
            if d >= 0 and (best is None or d < best[0]):
                best = (d, cx)
        if best is not None:
            anchors[ei] = best[1]
    # enforce monotonic x order along expected sequence
    ordered = sorted(anchors.items())
    keep, last_x = {}, -1
    for ei, cx in ordered:
        if cx > last_x:
            keep[ei] = cx
            last_x = cx
    return keep


def _column_edges(pos, inkcol, w):
    """Column boundaries between anchor positions on a text-rows-only ink
    projection.

    A wide low-ink valley (the real inter-column gap, possibly contaminated
    by faint ruled-line residue) must beat narrow zero-ink kerning gaps
    inside values, so the projection is smoothed with a wide moving average
    before taking the argmin, then refined to the local raw minimum."""
    n = len(pos)
    kernel = np.ones(25) / 25.0

    def boundary(lo, hi, fallback):
        lo, hi = int(max(0, lo)), int(min(w, hi))
        if hi - lo < 8:
            return int(fallback)
        seg = inkcol[lo:hi].astype(np.float64)
        sm = np.convolve(seg, kernel, mode="same")
        c = int(np.argmin(sm))
        lo2 = max(0, c - 12)
        hi2 = min(len(seg), c + 13)
        return lo + lo2 + int(np.argmin(seg[lo2:hi2]))

    edges = []
    gap0 = (pos[1] - pos[0]) if n > 1 else 200
    edges.append(boundary(pos[0] - gap0 * 1.6, pos[0] - gap0 * 0.45,
                          pos[0] - gap0))
    for i in range(n - 1):
        edges.append(boundary(pos[i] + 20, pos[i + 1] - 20,
                              (pos[i] + pos[i + 1]) / 2))
    gapN = (pos[-1] - pos[-2]) if n > 1 else 200
    edges.append(boundary(pos[-1] + gapN * 0.45, pos[-1] + gapN * 1.6,
                          pos[-1] + gapN))

    edges = [int(e) for e in edges]
    if any(edges[i + 1] - edges[i] < 25 for i in range(len(edges) - 1)):
        return None
    return edges


def _row_bands(gray, y_start, x0, x1, y_bottom=None):
    """Row bands via horizontal ink projection below the header line.
    Thin ruled separator lines are dropped; unusually tall bands (rows merged
    across a weak gap) are split evenly by the median row height."""
    h = gray.shape[0]
    y_end = int(y_bottom - 2) if y_bottom else h - 45
    region = gray[y_start + 3: y_end, x0:x1]
    ink = (region < INK_THRESHOLD).sum(axis=1)
    ys = np.where(ink > 3)[0]
    if len(ys) == 0:
        return []
    bands = []
    start = prev = ys[0]
    for y in ys[1:]:
        if y - prev > 10:
            bands.append((start, prev))
            start = y
        prev = y
    bands.append((start, prev))
    bands = [(a, b) for (a, b) in bands if b - a >= 14]
    if not bands:
        return []
    med = float(np.median([b - a for (a, b) in bands]))

    def split(a, b):
        """Split merged multi-row bands at the deepest ink valley."""
        if (b - a) < med * 1.55:
            return [(a, b)]
        lo = a + int((b - a) * 0.30)
        hi = a + int((b - a) * 0.70)
        if hi - lo < 3:
            return [(a, b)]
        y = lo + int(np.argmin(ink[lo:hi]))
        return split(a, y) + split(y + 1, b)

    out = []
    for (a, b) in bands:
        out.extend(split(a, b))
    return [(y_start + 3 + a - 7, y_start + 3 + b + 8) for (a, b) in out
            if b - a >= 12]


def _inset_for(prof) -> int:
    """Bordered cells need a wide inset to kill the rule lines; borderless
    bands are tight text bounds, so only a minimal inset is safe."""
    return CELL_INSET if prof["table"] == "bordered" else 2


# ============================================================ shared OCR core
def _ocr_table(gray, pno, name, prof, header, bands, cells_per_row):
    n = len(bands)
    ncols = len(prof["columns"])
    raw_txt = [[""] * ncols for _ in range(n)]
    raw_conf = [[0.0] * ncols for _ in range(n)]
    inset = _inset_for(prof)

    for ci, spec in enumerate(prof["columns"]):
        if spec["role"] == "skip":
            continue
        cells = []
        for ri in range(n):
            y0, y1 = bands[ri]
            x0, x1 = cells_per_row[ri][ci]
            c = crop(gray, y0, y1, x0, x1, inset=inset)
            cells.append(None if is_blank(c) else c)
        res = ocr_stack(cells, spec["lang"], whitelist=spec["wl"],
                        scale=spec["scale"])
        if spec["role"] in ("type", "branch", "desc"):
            for i, cell in enumerate(cells):
                if cell is None:
                    continue
                txt = res[i][0]
                if spec["role"] in ("type", "branch") and _needs_binarized_retry(spec, txt):
                    # Plain kor+eng reads KFTC and SPEED4; binarize is only the
                    # fallback for faint glyphs such as CC, and it can erase them.
                    for use_bin in (False, True):
                        t2, c2 = ocr_single(cell, "kor+eng", psm=7, whitelist=spec["wl"],
                                            scale=max(spec["scale"], 2.0),
                                            binarize_img=use_bin)
                        if t2 and not _needs_binarized_retry(spec, t2):
                            res[i] = (t2, c2)
                            txt = t2
                            break
                # Stacked OCR sometimes glues the next column onto a cell and
                # still reports high confidence. A single-cell read is clean.
                # A room number like 411호 is not noise: only symbol/jamo junk,
                # or a replacement that keeps the Hangul, may replace it.
                if spec["role"] == "branch" and (
                        _symbol_noise(txt) > 0 or _glued_digit(txt)):
                    t2, c2 = ocr_single(cell, spec["lang"], psm=7, whitelist=spec["wl"],
                                        scale=max(spec["scale"], 2.0))
                    if t2 and _cleaner(txt, t2):
                        res[i] = (t2, c2)
                        txt = t2
                if spec["role"] == "desc" and _desc_worth_retry(txt):
                    t2, c2 = ocr_single(cell, "kor+eng", psm=7, whitelist=spec["wl"],
                                        scale=2.0)
                    if _desc_better(txt, t2):
                        res[i] = (t2, c2)
        for ri, (txt, conf) in enumerate(res):
            raw_txt[ri][ci] = _postprocess(txt, spec)
            raw_conf[ri][ci] = round(conf, 1)

    rows, confs, keep = [], [], []
    for ri in range(n):
        uni = _build_unified(prof, raw_txt[ri])
        has_date = _valid_date(uni[0])
        amt_d = re.sub(r"\D", "", uni[3])
        bal_d = re.sub(r"\D", "", uni[4])
        has_nums = len(amt_d) >= 2 and len(bal_d) >= 3
        if has_date or has_nums:
            rows.append(uni)
            confs.append(raw_conf[ri])
            keep.append(ri)

    geom_rows = [{"band": [int(bands[i][0]), int(bands[i][1])],
                  "cells": [[int(a), int(b)] for (a, b) in cells_per_row[i]]}
                 for i in keep]
    return {
        "page": pno + 1, "kind": "transactions", "profile": name,
        "header": header, "rows": rows, "confs": confs,
        "geom": {"rows": geom_rows, "profile": name, "roles": _roles_map(prof)},
        "text": "",
    }


def _build_unified(prof, texts):
    by = {}
    for spec, t in zip(prof["columns"], texts):
        if spec["role"] != "skip":
            by[spec["role"]] = t
    if "datetime" in by:
        dt = by["datetime"]
        state = by.get("state", "")
        amt = by.get("amt", "")
    else:
        d = by.get("date", "").replace("-", "/")
        tm = by.get("time", "")
        dt = (d + " " + tm).strip()
        i_t, o_t = by.get("amt_in", ""), by.get("amt_out", "")
        if i_t and not o_t:
            state, amt = "입금", i_t
        elif o_t and not i_t:
            state, amt = "지급", o_t
        else:
            state, amt = "", (i_t or o_t)
    return [dt, state, by.get("type", ""), amt, by.get("bal", ""),
            by.get("branch", ""), by.get("desc", ""), by.get("check", ""),
            by.get("counter", "")]


# ============================================================== row recovery
def _recover_and_split_rows(clusters, header_row, data_rows):
    """Repair grid-detection gaps in bordered tables (see original docstring)."""
    med_h = float(np.median([rc.y1 - rc.y0 for rc in data_rows]))
    if med_h <= 0:
        return data_rows
    x_left = min(rc.cells[0][0] for rc in data_rows)
    x_right = max(rc.cells[-1][1] for rc in data_rows)
    y_top = header_row.y1 - 5
    y_bot = max(rc.y1 for rc in data_rows) + 1.5 * med_h

    matched_ids = {id(rc) for rc in data_rows} | {id(header_row)}
    rows = list(data_rows)
    for rc in clusters:
        if id(rc) in matched_ids or not rc.cells:
            continue
        cy = (rc.y0 + rc.y1) / 2
        if not (y_top < cy < y_bot):
            continue
        span = rc.cells[-1][1] - rc.cells[0][0]
        if span < (x_right - x_left) * 0.6:
            continue
        if rc.y1 - rc.y0 < med_h * 0.55:
            continue
        if any(abs(cy - (m.y0 + m.y1) / 2) < med_h * 0.4 for m in rows):
            continue
        nearest = min(data_rows, key=lambda m: abs(m.y0 - rc.y0))
        rows.append(RowCluster(y0=rc.y0, y1=rc.y1, cells=list(nearest.cells)))

    rows.sort(key=lambda rc: rc.y0)
    final = []
    for rc in rows:
        h = rc.y1 - rc.y0
        k = max(1, round(h / med_h))
        if k == 1:
            final.append(rc)
        else:
            for j in range(k):
                y0 = rc.y0 + int(j * h / k)
                y1 = rc.y0 + int((j + 1) * h / k)
                final.append(RowCluster(y0=y0, y1=y1, cells=list(rc.cells)))
    return final


def _needs_binarized_retry(spec, text) -> bool:
    """True when the first reading is not a real label.

    Transaction types are retried until they match the vocabulary (CD기,
    KFTC, ATM이체). Branch names are retried only when the Korean model
    returned a digit string such as SPEED4 -> 6506604.
    """
    s = re.sub(r"\s+", "", text or "")
    if spec["role"] == "type":
        canon = _canon_type(_tidy_text(text or ""))
        _, dist = vocab_correct(canon, spec.get("vocab", []), max_ratio=0.55)
        return dist < 0
    if spec["role"] == "branch":
        return bool(re.fullmatch(r"[0-9]{4,}", s))
    return False


_NOISE_RE = re.compile(r"[<>|=^\\\[\]_ㄱ-ㅎㅏ-ㅣ]|['\"‘’]")


def _symbol_noise(text: str) -> int:
    return len(_NOISE_RE.findall(text or ""))


def _glued_digit(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return bool(re.search(r"[가-힣]\d|\d[가-힣]", compact))


def _cleaner(old: str, new: str) -> bool:
    """True when `new` drops symbol garbage without throwing away Hangul.

    411호 must not become 4112. 시처도ㅇ may become 신정동.
    """
    if _symbol_noise(new) < _symbol_noise(old):
        return True
    if _symbol_noise(new) == 0 and _symbol_noise(old) == 0 and _glued_digit(old):
        old_h = len(re.findall(r"[가-힣]", old or ""))
        new_h = len(re.findall(r"[가-힣]", new or ""))
        return new_h >= old_h and not _glued_digit(new)
    return False


_LATIN_MEMO = re.compile(r"(LGU|KTN?|SKB|KB)\+?\d", re.I)


def _desc_worth_retry(text: str) -> bool:
    """Memos that lost their latin carrier code (LGU+, KT, SKB)."""
    t = text or ""
    if _symbol_noise(t):
        return True
    compact = re.sub(r"\s+", "", t)
    if re.fullmatch(r"[0-9]{8,}", compact):
        return True
    if re.search(r"[가-힣(<][0-9]{8,}", compact):
        return True
    return False


def _desc_better(old: str, new: str) -> bool:
    if not new or re.search(r"[!@#]", new):
        return False
    if _LATIN_MEMO.search(new) and not _LATIN_MEMO.search(old or ""):
        return True
    if (_symbol_noise(new) < _symbol_noise(old or "")
            and len(re.findall(r"[가-힣]", new)) >= len(re.findall(r"[가-힣]", old or "")) - 1):
        return True
    return False


def _canon_branch(t: str) -> str:
    """Digit soup and clipped latin for the SPEED4 ATM branch."""
    u = re.sub(r"\s+", "", t or "").upper()
    if "PEED4" in u or u in {"OPEED4", "0PEED4"}:
        return "SPEED4"
    if re.fullmatch(r"[04568]{6,8}", u):
        return "SPEED4"
    return t


def _tidy_text(t: str) -> str:
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"[\"'‘’“”＂]", "", t)
    t = re.sub(r"(일반업무\(자\()249[^)]*\)", r"\g<1>2497)", t)
    t = re.sub(r"^[아하]<", "", t)
    m = re.match(r"^([^가-힣+]{1,10})\+(.*)$", t.strip())
    if m and not m.group(1).upper().startswith("LGU") and re.search(
            r"[ㄱ-ㅎㅏ-ㅣ0-9\u1100-\u11FF]", m.group(1)):
        t = "LGU+" + m.group(2)
    t = re.sub(r"\s+(?=[가-힣])|(?<=[가-힣])\s+", "", t)
    t = t.replace("아브로드밴드", "브로드밴드").replace("하브로드밴드", "브로드밴드")
    t = re.sub(r"^[0-9<야하]+(?=브로드밴드)", "", t)
    letters = re.sub(r"[^가-힣]", "", t)
    if letters in {"통지수수료", "통지수료"}:
        return "SMS통지수수료"
    if "『" in t or "』" in t or (
            re.search(r"[씨시][카커까꺼]", t) and re.search(r"[\^&<>{}]", t)):
        return "FARHASADIA"
    if "시니" in t and re.search(r"[\^{0-9ㄱ-ㅎㅏ-ㅣ]", t):
        return "ZHAOKAILIN"
    if ("손" in letters and ("보" in letters or "ᄆ" in t or "ㅁ" in t)
            and set(letters) <= set("손보변경") and re.search(r"[0-9<ㄱ-ㅎㅁᄆ]", t)):
        return "DB손보변경" if "변경" in letters else "DB손보"
    if "통신요" in t and re.search(r"[<ㄱ-ㅎㅏ-ㅣ]", t):
        return "KT통신요금"
    if "만수" in t and "주공" in t and (
            re.search(r"[ㄱ-ㅎ\^&@]", t) or t.find("주공") < t.find("만수")):
        tail = ("단지" if "단지" in t else "") + ("아파트" if "아파트" in t else "")
        if "입주" in t:
            rest = t.replace("만수", "").replace("주공", "")
            tail += "입주자대표" if "대" in rest else "입주자"
        return "만수주공2" + tail
    t = re.sub(r"^[}{]+", "", t)
    t = re.sub(r"^<(?=\d)", "", t)
    m_fix = re.fullmatch(r"수\}?(\d+)리비", t)
    if m_fix:
        return f"{m_fix.group(1)}수리비"
    t = re.sub(r"^\^1/", "", t)
    t = t.replace("세어링", "셰어링")
    t = t.replace("(.*", "(")
    t = re.sub(r"\(\|\s*&?", "(", t)
    t = re.sub(r"(?<=\d)\s+(?=\d(?!차))", "", t)
    t = re.sub(r"\s*/\s*", "/", t)
    t = re.sub(r"^대카드(\d+)현$", r"현대카드\1", t)
    t = re.sub(r"(LGU\+)\s*[ㄱ-ㅎㅏ-ㅣ\u1100-\u11FF\s]*인터넷", r"\1인터넷", t)
    if ("LGU" not in t.upper() and "터넷" in t
            and re.search(r"[ㄱ-ㅎㅏ-ㅣ\u1100-\u11FF]", t)):
        t = "인터넷"
    t = re.sub(r"\s{2,}", " ", t).strip()
    return {
        "휴제영업점": "제휴영업점",
        "과금공납부": "공과금납부",
    }.get(t, t)


def _canon_type(t: str) -> str:
    """Map systematic misreads of short bank codes before vocab matching."""
    raw = re.sub(r"\s+", "", t)
    s = re.sub(r"[^0-9A-Za-z가-힣]", "", t)
    su = s.upper()
    if su in {"KFTC", "KF7C", "KFT0"}:
        return "KFTC"
    if re.fullmatch(r"사\d{1,4}", s):
        return "KFTC"
    if raw in {"&4미이체", "씨미이체"} or s in {"4미이체", "씨미이체"}:
        return "API이체"
    if re.search(r"[\^]", t) and re.search(r"[이의]체", s) and "타행" not in s:
        return "ATM이체"
    if re.fullmatch(r"\d{3,6}이체_?", s):
        return "ATM이체"
    if re.fullmatch(r"CD[7IL1기]{0,3}", su) or s in {"00기", "0기", "C07", "CD7", "ㅁ기"}:
        return "CD기"
    hangul = re.findall(r"[가-힣]", s)
    if hangul == ["타"] and re.search(r"[0-9CDILli]", s):
        return "타CD"
    if "펌" in s and "체" in s and "뱅킹" not in s:
        return "펌이체"
    return t


# ============================================================== postprocess
def _postprocess(txt: str, spec) -> str:
    t = txt.strip()
    kind = spec.get("kind", "text")
    if kind == "datetime":
        t = t.replace(" ", "")
        m = re.match(r"^(\d{4}/\d{2}/\d{2})(\d{2}:\d{2}:\d{2})?", t)
        if m:
            return m.group(1) + (" " + m.group(2) if m.group(2) else "")
        return txt.strip()
    if kind == "date_dash":
        t = t.replace(" ", "")
        m = re.match(r"^(\d{4})-?(\d{2})-?(\d{2})", t)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return t
    if kind == "time":
        t = t.replace(" ", "")
        m = re.match(r"^(\d{2}):?(\d{2}):?(\d{2})", t)
        if m:
            return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
        return t
    if kind == "int":
        return t.replace(" ", "")
    if kind == "zero_blank":
        t = t.replace(" ", "")
        return "" if t in ("0", "00") else t
    if kind == "vocab":
        c, _ = vocab_correct(_tidy_text(t), STATE_VOCAB, max_ratio=1.0)
        return c
    if kind == "vocab_soft":
        t = _canon_type(_tidy_text(t))
        c, d = vocab_correct(t, spec["vocab"], max_ratio=0.55)
        return c if d >= 0 else _tidy_text(t)
    if kind == "text":
        t = re.sub(r"^[\s_\'\"‘’|.·-]+|[\s_\'\"‘’|.·-]+$", "", t)
        t = t.replace("_", "").replace("\\", "")
        t = re.sub(r"\(\(+", "(", t)
        t = _tidy_text(t)
        if spec.get("role") == "branch":
            t = _canon_branch(t)
        return t
    return t


# ============================================================ header parsing
def _parse_labeled_header(gray, clusters, before_y, prof) -> dict:
    """Bordered header bar: label cell followed by value cell."""
    out = {"계좌번호": "", "고객명": "", "상품명": ""}
    labels = prof["header_labels"]
    cands = [rc for rc in clusters if rc.y1 <= before_y + 5 and len(rc.cells) >= 4]
    if not cands:
        return out
    rc = cands[-1]
    cells = [crop(gray, rc.y0, rc.y1, x0, x1) for (x0, x1) in rc.cells]
    texts = ocr_stack(cells, "kor+eng", scale=1.5)
    clean = re.compile(r"\s+(?=[가-힣0-9])|(?<=[가-힣0-9])\s+")
    vals = [clean.sub("", t) for t, _ in texts]
    for i, v in enumerate(vals):
        for label, key in labels.items():
            if label in v and i + 1 < len(vals):
                out[key] = vals[i + 1].strip()
    if out["계좌번호"]:
        acct = re.sub(r"[^\d-]", "", out["계좌번호"])
        if len(re.sub(r"\D", "", acct)) >= 8:
            out["계좌번호"] = acct
    return _canon_header(out)


def _parse_strip_header(gray, header_top, prof) -> dict:
    """Borderless header: OCR the strip above the column header and regex the
    label: value pairs out of the line texts."""
    out = {"계좌번호": "", "고객명": "", "상품명": ""}
    y0 = max(0, header_top - 620)
    strip = gray[y0:max(y0 + 1, header_top - 6), :]
    words = _run(strip, "kor+eng", psm=6)
    if not words:
        return out
    words.sort(key=lambda w: (w["y"], w["x"]))
    lines, cur, last_y = [], [], -100
    for w in words:
        if w["y"] - last_y > 22 and cur:
            lines.append(cur)
            cur = []
        cur.append(w)
        last_y = w["y"]
    if cur:
        lines.append(cur)
    texts = [" ".join(w["text"] for w in sorted(l, key=lambda t: t["x"]))
             for l in lines]
    joined = "\n".join(texts)
    m = re.search(r"계\s*좌\s*번\s*호\D*([\d][\d\- ]{7,})", joined)
    if m:
        acct = re.sub(r"[^\d-]", "", m.group(1))
        if len(re.sub(r"\D", "", acct)) >= 8:
            out["계좌번호"] = acct
    m = re.search(r"예\s*금\s*주\s*명\s*[:.]?\s*([가-힣A-Za-z()\s]{1,30}?)(?=\s*(거래점|출력|조회|$))",
                  joined, re.M)
    if m:
        out["고객명"] = re.sub(r"\s+", "", m.group(1))
    m = re.search(r"계\s*정\s*과\s*목\s*[:.]?\s*([가-힣A-Za-z\s]{1,20}?)(?=\s|$)", joined)
    if m:
        out["상품명"] = re.sub(r"\s+", "", m.group(1))
    return _canon_header(out)


def _canon_header(out: dict) -> dict:
    """Fix product names and company suffixes the header font always breaks."""
    prod = out.get("상품명") or ""
    prod = prod.replace("1<8두레통장", "KB두레통장")
    prod = re.sub(r"KB\s*[\\/]*\s*156통장", "KB Wise통장", prod)
    out["상품명"] = prod
    name = (out.get("고객명") or "").replace("세어링", "셰어링")
    if "제이케이" in name and "셰어링" in name:
        person = re.match(r"([가-힣]{2,4})", name)
        if person:
            out["고객명"] = f"{person.group(1)}(제이케이셰어링)"
    elif re.fullmatch(r"[가-힣]{2,4}[^가-힣A-Za-z]*셰어링\)?", name):
        person = re.match(r"([가-힣]{2,4})", name)
        out["고객명"] = f"{person.group(1)}(JK셰어링)"
    elif name:
        out["고객명"] = name
    return out


def _other_page(gray, pno) -> dict:
    small = cv2.resize(gray, None, fx=0.66, fy=0.66, interpolation=cv2.INTER_AREA)
    if int((small < INK_THRESHOLD).sum()) < 200:
        return {"page": pno + 1, "kind": "empty", "profile": "", "header": {},
                "rows": [], "confs": [], "geom": None, "text": ""}
    text, _ = ocr_single(small, "kor+eng", psm=3)
    return {"page": pno + 1, "kind": "other", "profile": "", "header": {},
            "rows": [], "confs": [], "geom": None, "text": text}


# ------------------------------------------------------------- re-OCR helpers
def _geom_inset(geom) -> int:
    return _inset_for(PROFILES[geom.get("profile", "KB")])


def reocr_cell(gray, geom, row_idx, col_idx, spec, scale=2.5):
    row = geom["rows"][row_idx]
    y0, y1 = row["band"]
    x0, x1 = row["cells"][col_idx]
    c = crop(gray, y0, y1, x0, x1, inset=_geom_inset(geom))
    if is_blank(c):
        return "", 0.0
    txt, conf = ocr_single(c, spec["lang"], psm=7, whitelist=spec["wl"], scale=scale)
    return _postprocess(txt, spec), conf


def reocr_cell_candidates(gray, geom, row_idx, col_idx, spec, scale=2.5):
    """Multiple re-OCR readings: raw grayscale plus binarized variants that
    suppress light watermark/stamp ink. Distinct non-empty texts, best first."""
    row = geom["rows"][row_idx]
    y0, y1 = row["band"]
    x0, x1 = row["cells"][col_idx]
    c = crop(gray, y0, y1, x0, x1, inset=_geom_inset(geom))
    if is_blank(c):
        return []
    variants = [c]
    for th in (120, 140):
        variants.append(np.where(c < th, 0, 255).astype(np.uint8))
    outs = []
    for v in variants:
        txt, conf = ocr_single(v, spec["lang"], psm=7, whitelist=spec["wl"], scale=scale)
        t = _postprocess(txt, spec)
        if t:
            outs.append((t, conf))
    outs.sort(key=lambda x: -x[1])
    seen, uniq = set(), []
    for t, _ in outs:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq
