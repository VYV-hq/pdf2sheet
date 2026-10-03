"""CLI: python -m app.cli input.pdf [-o out.xlsx] [--batch-size N] [--workers N]"""
import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

from .agent import run_job
from .config import BATCH_SIZE, JOBS_DIR, MAX_WORKERS


def main():
    ap = argparse.ArgumentParser(description="PDF -> Excel 변환")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--job-dir", default=None)
    args = ap.parse_args()

    pdf = Path(args.pdf)
    if not pdf.exists():
        sys.exit(f"파일 없음: {pdf}")

    job_dir = Path(args.job_dir) if args.job_dir else \
        JOBS_DIR / ("cli_" + time.strftime("%Y%m%d_%H%M%S"))
    job_dir.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()

    def watch():
        sp = job_dir / "status.json"
        last = ""
        while not stop.is_set():
            if sp.exists():
                st = json.loads(sp.read_text())
                msg = (f"[{st.get('state')}] pages {st.get('pages_done', 0)}"
                       f"/{st.get('pages_total', '?')} "
                       f"batches {st.get('batches_done', 0)}/{st.get('batches_total', '?')} "
                       f"{st.get('elapsed_sec', 0)}s")
                if msg != last:
                    print(msg, flush=True)
                    last = msg
            stop.wait(3)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    run_job(job_dir, pdf, batch_size=args.batch_size, workers=args.workers)
    stop.set()
    t.join(1)

    st = json.loads((job_dir / "status.json").read_text())
    print(json.dumps({k: v for k, v in st.items() if k != "error"},
                     ensure_ascii=False, indent=1))
    if args.output:
        shutil.copy(job_dir / "output.xlsx", args.output)
        csv_out = Path(args.output).with_suffix(".csv")
        shutil.copy(job_dir / "output.csv", csv_out)
        print(f"저장: {args.output}, {csv_out}")
    else:
        print(f"결과: {job_dir}/output.xlsx")


if __name__ == "__main__":
    main()
