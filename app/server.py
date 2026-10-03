"""FastAPI web server: upload a (large) PDF, track progress, download Excel."""
import json
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from .agent import run_job
from .config import BASE_DIR, BATCH_SIZE, JOBS_DIR, MAX_WORKERS

app = FastAPI(title="pdf2sheet")
JOBS_DIR.mkdir(exist_ok=True)

_lock = threading.Lock()
_running: dict[str, threading.Thread] = {}


@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/ibk", response_class=HTMLResponse)
def ibk_page():
    return (BASE_DIR / "static" / "ibk.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(request: Request):
    """Streaming upload: body = raw PDF bytes; filename via X-Filename header
    (also accepts multipart forms from the browser UI)."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    pdf_path = job_dir / "input.pdf"

    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        up = form["file"]
        with open(pdf_path, "wb") as f:
            while chunk := await up.read(1 << 20):
                f.write(chunk)
        name = up.filename or "input.pdf"
    else:
        with open(pdf_path, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
        name = request.headers.get("x-filename", "input.pdf")

    if pdf_path.stat().st_size < 5 or open(pdf_path, "rb").read(5) != b"%PDF-":
        raise HTTPException(400, "PDF 파일이 아닙니다")

    (job_dir / "meta.json").write_text(
        json.dumps({"filename": name}, ensure_ascii=False))

    params = request.query_params
    batch_size = int(params.get("batch_size", BATCH_SIZE))
    workers = int(params.get("workers", MAX_WORKERS))

    t = threading.Thread(target=run_job, args=(job_dir, pdf_path),
                         kwargs={"batch_size": batch_size, "workers": workers},
                         daemon=True)
    with _lock:
        _running[job_id] = t
    t.start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job_dir = JOBS_DIR / job_id
    status_path = job_dir / "status.json"
    if not job_dir.exists():
        raise HTTPException(404, "job not found")
    if not status_path.exists():
        return {"state": "starting"}
    st = json.loads(status_path.read_text())
    meta = job_dir / "meta.json"
    if meta.exists():
        st["filename"] = json.loads(meta.read_text()).get("filename")
    return JSONResponse(st)


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str, fmt: str = "xlsx"):
    ext = "xlsx" if fmt == "xlsx" else "csv"
    path = JOBS_DIR / job_id / f"output.{ext}"
    if not path.exists():
        raise HTTPException(404, "결과 파일이 아직 없습니다")
    meta = JOBS_DIR / job_id / "meta.json"
    base = "output"
    if meta.exists():
        base = Path(json.loads(meta.read_text()).get("filename", "output")).stem
    media = ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
             if ext == "xlsx" else "text/csv")
    return FileResponse(path, media_type=media, filename=f"{base}.{ext}")


@app.get("/api/jobs")
def list_jobs():
    out = []
    for d in sorted(JOBS_DIR.iterdir()):
        sp = d / "status.json"
        if sp.exists():
            st = json.loads(sp.read_text())
            meta = d / "meta.json"
            fname = json.loads(meta.read_text()).get("filename") if meta.exists() else ""
            out.append({"job_id": d.name, "state": st.get("state"),
                        "filename": fname,
                        "pages_done": st.get("pages_done"),
                        "pages_total": st.get("pages_total")})
    return out
