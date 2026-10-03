"""Job orchestrator ("the agent").

Plan   : split the PDF into batches of n pages.
Act    : workers extract structured rows per page (checkpointed as JSONL).
Verify : merge all rows and validate every transaction against the running
         balance equation  bal[i] == bal[i-1] +/- amount[i].
Repair : rows that break the chain get targeted high-res re-OCR; the 상태
         (입금/지급) column is corrected from the balance direction itself.
Merge  : everything is written into one Excel workbook (+ CSV).
"""
import json
import os
import re
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from .config import BATCH_SIZE, MAX_WORKERS, PIPELINE_VERSION, PROFILES
from . import pdfio, extract
from .excel import write_outputs

_worker_doc = None
_worker_pdf = None
_worker_rot = None      # cached content-rotation for rotated scans


def _init_worker(pdf_path):
    global _worker_doc, _worker_pdf, _worker_rot
    _worker_pdf = pdf_path
    _worker_doc = pymupdf.open(pdf_path)
    _worker_rot = None


def _process_batch(args):
    """Runs inside a worker process: extract a contiguous page range."""
    batch_id, p0, p1 = args
    global _worker_doc, _worker_rot
    results, errors = [], []
    for pno in range(p0, p1):
        try:
            gray, rot = pdfio.page_gray(_worker_doc, pno, prefer=_worker_rot)
            res = extract.process_page(gray, pno, rot=rot)
            if res["kind"] != "transactions":
                # IBK (and some other banks) store landscape tables as
                # portrait scans. If auto-detect picked upright (or the
                # wrong 90°), try the remaining orientations.
                tried = {rot}
                for alt in (1, 3, 0, 2):
                    if alt in tried:
                        continue
                    gray2, rot2 = pdfio.page_gray(_worker_doc, pno, force=alt)
                    res2 = extract.process_page(gray2, pno, rot=rot2)
                    if res2["kind"] == "transactions":
                        res = res2
                        break
            if res["kind"] == "transactions" and res.get("rot") in (1, 3):
                _worker_rot = res["rot"]
            results.append(res)
        except Exception:
            errors.append({"page": pno + 1, "error": traceback.format_exc(limit=3)})
            results.append({"page": pno + 1, "kind": "error", "header": {},
                            "rows": [], "confs": [], "geom": None, "text": ""})
    return batch_id, results, errors


class JobRunner:
    def __init__(self, job_dir: Path, pdf_path: Path,
                 batch_size: int = BATCH_SIZE, workers: int = MAX_WORKERS):
        self.job_dir = Path(job_dir)
        self.pdf_path = Path(pdf_path)
        self.batch_size = batch_size
        self.workers = workers
        self.batches_dir = self.job_dir / "batches"
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.status_path = self.job_dir / "status.json"
        self.t0 = time.time()

    # ------------------------------------------------------------ status io
    def _status(self, **kw):
        st = {}
        if self.status_path.exists():
            st = json.loads(self.status_path.read_text())
        st.update(kw)
        st["elapsed_sec"] = round(time.time() - self.t0, 1)
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
        os.replace(tmp, self.status_path)
        return st

    # ------------------------------------------------------------- main run
    def run(self):
        try:
            self._run()
        except Exception:
            self._status(state="error", error=traceback.format_exc(limit=5))
            raise

    def _run(self):
        doc = pymupdf.open(self.pdf_path)
        total = doc.page_count
        doc.close()

        # discard checkpoints written by an incompatible pipeline version
        if self.status_path.exists():
            old = json.loads(self.status_path.read_text())
            if old.get("pipeline_version") != PIPELINE_VERSION:
                for f in self.batches_dir.glob("batch_*.jsonl"):
                    f.unlink()

        batches = []
        for i, p0 in enumerate(range(0, total, self.batch_size)):
            batches.append((i, p0, min(p0 + self.batch_size, total)))

        self._status(state="processing", pages_total=total, pages_done=0,
                     batches_total=len(batches), batches_done=0,
                     batch_size=self.batch_size, workers=self.workers,
                     pipeline_version=PIPELINE_VERSION, error=None)

        # resume support: skip batches that already have a checkpoint
        pending = [b for b in batches if not self._ckpt(b[0]).exists()]
        done_pages = sum(b[2] - b[1] for b in batches if self._ckpt(b[0]).exists())
        done_batches = len(batches) - len(pending)
        all_errors = []

        if pending:
            with ProcessPoolExecutor(max_workers=self.workers,
                                     initializer=_init_worker,
                                     initargs=(str(self.pdf_path),)) as pool:
                futs = {pool.submit(_process_batch, b): b for b in pending}
                for fut in as_completed(futs):
                    batch_id, results, errors = fut.result()
                    all_errors.extend(errors)
                    with open(self._ckpt(batch_id), "w", encoding="utf-8") as f:
                        for r in results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    done_pages += len(results)
                    done_batches += 1
                    self._status(pages_done=done_pages, batches_done=done_batches)

        self._status(state="merging", pages_done=done_pages)
        summary = self._merge(batches, all_errors)
        self._status(state="done", **summary)

    def _ckpt(self, batch_id) -> Path:
        return self.batches_dir / f"batch_{batch_id:04d}.jsonl"

    # ------------------------------------------------- merge + validate + fix
    def _merge(self, batches, errors):
        pages = []
        for b in batches:
            path = self._ckpt(b[0])
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    pages.append(json.loads(line))
        pages.sort(key=lambda p: p["page"])

        pages = self._smooth_accounts(pages)
        rows, issues = self._build_rows(pages)
        fixed, still_bad = self._repair(rows, pages)
        issues.extend(still_bad)
        d_fixed, d_issues = self._repair_dates(rows, pages)
        fixed += d_fixed
        issues.extend(d_issues)

        xlsx, csv = write_outputs(self.job_dir, rows, pages, issues, errors)
        n_txn = len(rows)
        accounts = sorted({r["계좌번호"] for r in rows if r["계좌번호"]})
        ok = sum(1 for r in rows if r["검증"] == "OK")
        return {
            "rows_total": n_txn, "accounts": len(accounts),
            "validated_ok": ok, "validation_rate": round(ok / n_txn * 100, 2) if n_txn else 0,
            "repaired_cells": fixed, "issues": len(issues),
            "output_xlsx": xlsx.name, "output_csv": csv.name,
        }

    @staticmethod
    def _smooth_accounts(pages):
        """Fix isolated OCR glitches in the account number using neighbours,
        then normalize 고객명/상품명 to the most frequent reading per account."""
        tx = [p for p in pages if p["kind"] == "transactions"]
        for i in range(1, len(tx) - 1):
            a, b, c = tx[i - 1], tx[i], tx[i + 1]
            an, bn, cn = (p["header"].get("계좌번호", "") for p in (a, b, c))
            if an and an == cn and bn != an:
                b["header"]["계좌번호"] = an
        # forward fill missing account numbers within the same run
        prev = None
        for p in tx:
            h = p["header"]
            if not h.get("계좌번호") and prev:
                h["계좌번호"] = prev.get("계좌번호", "")
            prev = h
        # majority vote for the descriptive fields (same on every page of an
        # account, so the mode across dozens of pages beats any single read)
        votes = {}
        for p in tx:
            h = p["header"]
            acc = h.get("계좌번호", "")
            for k in ("고객명", "상품명"):
                v = h.get(k, "")
                if v:
                    votes.setdefault((acc, k), Counter())[v] += 1
        for p in tx:
            h = p["header"]
            acc = h.get("계좌번호", "")
            for k in ("고객명", "상품명"):
                c = votes.get((acc, k))
                if c:
                    h[k] = c.most_common(1)[0][0]
        return pages

    @staticmethod
    def _parse_int(s):
        t = re.sub(r"[^\d]", "", s or "")
        return int(t) if t else None

    def _build_rows(self, pages):
        rows, issues = [], []
        for p in pages:
            if p["kind"] != "transactions":
                continue
            h = p["header"]
            for ri, (cells, confs) in enumerate(zip(p["rows"], p["confs"])):
                rows.append({
                    "페이지": p["page"], "행": ri,
                    "계좌번호": h.get("계좌번호", ""),
                    "고객명": h.get("고객명", ""),
                    "상품명": h.get("상품명", ""),
                    "거래일시": cells[0], "상태": cells[1], "거래구분": cells[2],
                    "거래금액": self._parse_int(cells[3]),
                    "표면잔액": self._parse_int(cells[4]),
                    "취급점": cells[5], "적요(의뢰인등)": cells[6],
                    "수표번호": cells[7], "이체상대계좌번호": cells[8],
                    "_confs": confs, "검증": "", "비고": "",
                })
        return rows, issues

    def _repair(self, rows, pages):
        """Balance-chain validation with targeted re-OCR + chain derivation.

        For each broken link, in order:
          1. re-OCR the amount/balance cells at high resolution;
          2. derive the balance as prev±amount and accept it when it makes the
             following links consistent (lookahead handles consecutive
             misreads);
          3. derive the amount as |balance - prev| when the balance itself is
             confirmed by the next link.
        """
        geom_by_page = {p["page"]: p.get("geom") for p in pages}
        rot_by_page = {p["page"]: p.get("rot", 0) for p in pages}
        self._rot_by_page = rot_by_page
        doc = None
        fixed_cells = 0
        issues = []

        by_acct = {}
        for r in rows:
            by_acct.setdefault(r["계좌번호"], []).append(r)

        for acct, rs in by_acct.items():
            prev_bal = None
            prev_ok = False    # previous link chain-validated -> safe anchor
            for i, r in enumerate(rs):
                amt, bal = r["거래금액"], r["표면잔액"]
                ocr_state = r["상태"] if r["상태"] in ("입금", "지급") else None
                state = self._chain_state(prev_bal, amt, bal)
                note = ""

                # composite sub-rows (수표/현금 구성내역): the statement prints
                # them with an empty 상태 and 표면잔액 0. They are not part of
                # the balance chain - keep them but do not advance prev_bal.
                if state is None and bal == 0 and amt and not ocr_state:
                    r["상태"] = ""
                    r["검증"] = "OK(구성행)"
                    continue

                needs_search = prev_bal is not None and (
                    state is None or (ocr_state and ocr_state != state))
                if needs_search:
                    # multi-variant re-OCR + combo search: pick the (amt, bal)
                    # pair that closes the chain, preferring combos anchored by
                    # the next link and agreeing with the OCR-read direction.
                    if doc is None:
                        doc = pymupdf.open(self.pdf_path)
                    found = self._search_combo(doc, geom_by_page, rs, i, r,
                                               prev_bal, ocr_state, amt, bal,
                                               prev_ok)
                    if found:
                        amt2, bal2, state2, n_chg = found
                        if n_chg:
                            note = self._chg_note(amt, bal, amt2, bal2)
                            fixed_cells += n_chg
                        amt, bal, state = amt2, bal2, state2

                r["거래금액"], r["표면잔액"] = amt, bal

                if state:
                    # unresolved direction conflict: a misread balance can
                    # satisfy the equation in the wrong direction, so without
                    # a downstream anchor we flag instead of trusting it.
                    if ocr_state and ocr_state != state and \
                            not self._chain_holds(rs, i + 1, bal, depth=2):
                        r["검증"] = "확인필요"
                        r["비고"] = (f"상태/잔액 방향충돌(판독상태={ocr_state}, "
                                     f"체인={state}, 이전잔액={prev_bal})")
                        issues.append({"페이지": r["페이지"], "행": r["행"],
                                       "계좌번호": acct, "사유": r["비고"]})
                        prev_bal, prev_ok = None, False
                        continue
                    if ocr_state and ocr_state != state:
                        note = (note + "; " if note else "") + \
                            f"상태는 잔액체인기준(판독:{ocr_state})"
                    r["상태"] = state
                    r["검증"] = "OK" if not note else "OK(보정)"
                    r["비고"] = note
                    prev_bal, prev_ok = bal, True
                elif prev_bal is None:
                    if i == 0:
                        r["검증"] = "OK(시작)"
                    else:
                        r["검증"] = "미검증(체인재시작)"
                    prev_bal, prev_ok = bal, False
                else:
                    exp = (f"기대={prev_bal}±{amt}" if amt is not None
                           else f"이전잔액={prev_bal}")
                    r["검증"] = "확인필요"
                    r["비고"] = f"잔액연결불일치(이전잔액={prev_bal}, {exp})"
                    issues.append({"페이지": r["페이지"], "행": r["행"],
                                   "계좌번호": acct, "사유": r["비고"]})
                    prev_bal, prev_ok = None, False
        if doc:
            doc.close()
        return fixed_cells, issues

    def _chain_holds(self, rs, i, prev_bal, depth):
        """True when the balance chain starting at row i is consistent with
        prev_bal, allowing up to `depth` consecutive derived balances."""
        if i >= len(rs):
            return True          # end of account: nothing to contradict
        if depth <= 0:
            return False
        amt, bal = rs[i]["거래금액"], rs[i]["표면잔액"]
        if bal == 0 and amt and rs[i]["상태"] not in ("입금", "지급"):
            # composite sub-row: not part of the chain
            return self._chain_holds(rs, i + 1, prev_bal, depth)
        if amt is None:
            return False
        for sign in (1, -1):
            cand = prev_bal + sign * amt
            if cand < 0:
                continue
            if bal is not None and cand == bal:
                return True      # chain reconnects with observed balance
            if self._chain_holds(rs, i + 1, cand, depth - 1):
                return True
        return False

    @staticmethod
    def _chain_state(prev_bal, amt, bal):
        if prev_bal is None or amt is None or bal is None:
            return None
        if prev_bal + amt == bal:
            return "입금"
        if prev_bal - amt == bal:
            return "지급"
        return None

    def _search_combo(self, doc, geom_by_page, rs, i, row, prev_bal,
                      ocr_state, amt, bal, prev_ok):
        """Try (amount, balance) candidate pairs; return the best chain-valid
        combo or None.

        Candidates come from multi-variant re-OCR (pixel readings) and - when
        the previous link is a validated anchor - from derivation (prev±amt,
        |bal-prev|). Derived components are only accepted when the resulting
        balance is anchored by the following links, so a corrupt read can
        never fabricate its neighbours.

        Score: downstream anchoring (x4) > OCR direction agreement (x2) >
        keeping the original readings (x1 each)."""
        geom = geom_by_page.get(row["페이지"])
        if not geom:
            return None
        gray, _ = pdfio.page_gray(doc, row["페이지"] - 1,
                                  force=self._rot_by_page.get(row["페이지"], 0))
        prof = PROFILES[geom.get("profile", "KB")]
        roles = geom.get("roles") or {s["role"]: i for i, s in
                                      enumerate(prof["columns"]) if s["role"] != "skip"}
        amt_cols = [roles[k] for k in ("amt", "amt_in", "amt_out") if k in roles]
        bal_col = roles["bal"]
        amt_ocr = [amt] if amt is not None else []
        bal_ocr = [bal] if bal is not None else []
        for col, cands in [(c, amt_ocr) for c in amt_cols] + [(bal_col, bal_ocr)]:
            for txt in extract.reocr_cell_candidates(
                    gray, geom, row["행"], col, spec=prof["columns"][col]):
                v = self._parse_int(txt)
                if v is not None and v not in cands:
                    cands.append(v)
        amt_drv, bal_drv = [], []
        if prev_ok:
            for a in amt_ocr:
                for cand in (prev_bal + a, prev_bal - a):
                    if cand >= 0 and cand not in bal_ocr and cand not in bal_drv:
                        bal_drv.append(cand)
            for b in bal_ocr:
                cand = abs(b - prev_bal)
                if cand > 0 and cand not in amt_ocr and cand not in amt_drv:
                    amt_drv.append(cand)

        # candidate evidence tiers: 2 = original reading, 1 = re-OCR variant
        # (pixel evidence), 0 = derived. Lexicographic score: downstream
        # anchoring first, then pixel evidence, then direction agreement -
        # a chain-valid anchored original must never lose to a derivation.
        def tier(v, orig, ocr_list):
            if orig is not None and v == orig:
                return 2
            return 1 if v in ocr_list else 0

        best, best_score = None, (-1, -1, -1)
        for a in amt_ocr + amt_drv:
            for b in bal_ocr + bal_drv:
                ta = tier(a, amt, amt_ocr)
                tb = tier(b, bal, bal_ocr)
                if ta == 0 and tb == 0:
                    continue                      # pure invention
                st = self._chain_state(prev_bal, a, b)
                if not st:
                    continue
                anchored = self._chain_holds(rs, i + 1, b, depth=3)
                if (ta == 0 or tb == 0) and not anchored:
                    continue                      # derived values need proof
                dir_ok = 1 if (ocr_state and st == ocr_state) else 0
                score = (int(anchored), ta + tb, dir_ok)
                if score > best_score:
                    best, best_score = (a, b, st), score
        if best is None:
            return None
        a, b, st = best
        n_chg = int(a != amt) + int(b != bal)
        return a, b, st, n_chg

    @staticmethod
    def _chg_note(amt, bal, amt2, bal2):
        parts = []
        if amt2 != amt:
            parts.append(f"거래금액 재판독({amt}→{amt2})")
        if bal2 != bal:
            parts.append(f"표면잔액 재판독({bal}→{bal2})")
        return "; ".join(parts)

    # ------------------------------------------------------- date validation
    @staticmethod
    def _to_dt(s):
        """Parse 거래일시 -> (datetime|None, has_time)."""
        from datetime import datetime
        s = (s or "").strip()
        for fmt, ht in (("%Y/%m/%d %H:%M:%S", True), ("%Y/%m/%d", False)):
            try:
                return datetime.strptime(s, fmt), ht
            except ValueError:
                continue
        return None, False

    def _repair_dates(self, rows, pages):
        """Dates must be monotonically non-decreasing within an account.
        Unparseable or out-of-order dates get multi-variant re-OCR; a repaired
        value is accepted only when it fits between its valid neighbours."""
        geom_by_page = {p["page"]: p.get("geom") for p in pages}
        doc = None
        fixed = 0
        issues = []

        by_acct = {}
        for r in rows:
            if r["검증"] != "OK(구성행)":
                by_acct.setdefault(r["계좌번호"], []).append(r)

        def reocr_date(r):
            nonlocal doc
            geom = geom_by_page.get(r["페이지"])
            if not geom:
                return []
            if doc is None:
                doc = pymupdf.open(self.pdf_path)
            gray, _ = pdfio.page_gray(
                doc, r["페이지"] - 1,
                force=self._rot_by_page.get(r["페이지"], 0))
            prof = PROFILES[geom.get("profile", "KB")]
            roles = geom.get("roles") or {s["role"]: i for i, s in
                                          enumerate(prof["columns"]) if s["role"] != "skip"}
            texts = []
            if "datetime" in roles:
                col = roles["datetime"]
                texts = extract.reocr_cell_candidates(
                    gray, geom, r["행"], col, spec=prof["columns"][col])
            else:
                dcol = roles["date"]
                dcands = extract.reocr_cell_candidates(
                    gray, geom, r["행"], dcol, spec=prof["columns"][dcol])
                tcands = []
                if "time" in roles:
                    tcol = roles["time"]
                    tcands = extract.reocr_cell_candidates(
                        gray, geom, r["행"], tcol, spec=prof["columns"][tcol])
                if not tcands:
                    m = re.search(r"\d{2}:\d{2}:\d{2}", r["거래일시"] or "")
                    tcands = [m.group(0)] if m else [""]
                for d in dcands:
                    for t in tcands:
                        texts.append((d.replace("-", "/") + " " + t).strip())
            cands = []
            for txt in texts:
                dt, ht = self._to_dt(txt)
                if dt is not None:
                    cands.append((dt, ht, txt))
            cands.sort(key=lambda c: -c[1])   # prefer full datetime readings
            return cands

        for acct, rs in by_acct.items():
            dts = [self._to_dt(r["거래일시"]) for r in rs]

            def window(i):
                """(lo, hi) datetime bounds from nearest valid neighbours."""
                lo = hi = None
                for j in range(i - 1, -1, -1):
                    if j not in suspects and dts[j][0] is not None:
                        lo = dts[j][0]
                        break
                for j in range(i + 1, len(rs)):
                    if j not in suspects and dts[j][0] is not None:
                        hi = dts[j][0]
                        break
                return lo, hi

            def fits(dt, ht, lo, hi):
                from datetime import timedelta
                d0 = dt if ht else dt
                if lo is not None:
                    lo_c = lo.replace(hour=0, minute=0, second=0) if not ht else lo
                    if d0 < lo_c:
                        return False
                if hi is not None:
                    hi_c = hi.replace(hour=23, minute=59, second=59) if not ht else hi
                    if d0 > hi_c:
                        return False
                return True

            # suspects: unparseable, or breaks monotonic order with neighbours
            suspects = set()
            for i, (dt, ht) in enumerate(dts):
                if dt is None:
                    suspects.add(i)
            valid = [i for i in range(len(rs)) if i not in suspects]
            for k in range(1, len(valid)):
                i0, i1 = valid[k - 1], valid[k]
                d0, h0 = dts[i0]
                d1, h1 = dts[i1]
                c0 = d0.date() if not (h0 and h1) else d0
                c1 = d1.date() if not (h0 and h1) else d1
                if c0 > c1:
                    suspects.add(i0)
                    suspects.add(i1)

            for i in sorted(suspects):
                r = rs[i]
                lo, hi = window(i)
                repaired = False
                for dt, ht, txt in reocr_date(r):
                    if fits(dt, ht, lo, hi):
                        if txt != r["거래일시"]:
                            note = f"거래일시 재판독({r['거래일시']}→{txt})"
                            r["비고"] = (r["비고"] + "; " if r["비고"] else "") + note
                            r["거래일시"] = txt
                            if r["검증"] == "OK":
                                r["검증"] = "OK(보정)"
                            fixed += 1
                        dts[i] = (dt, ht)
                        repaired = True
                        break
                if not repaired:
                    orig_dt, _ = self._to_dt(r["거래일시"])
                    if orig_dt is not None and fits(orig_dt, True, lo, hi):
                        continue      # original was fine (partner was bad)
                    note = "거래일시 확인필요(판독불가/순서이상)"
                    r["비고"] = (r["비고"] + "; " if r["비고"] else "") + note
                    issues.append({"페이지": r["페이지"], "행": r["행"],
                                   "계좌번호": acct, "사유": note})
        if doc:
            doc.close()
        return fixed, issues


def run_job(job_dir, pdf_path, batch_size=BATCH_SIZE, workers=MAX_WORKERS):
    JobRunner(job_dir, pdf_path, batch_size, workers).run()
