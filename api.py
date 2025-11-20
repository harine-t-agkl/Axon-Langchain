# app.py
import os
import uuid
import json
import traceback
import multiprocessing
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# Import your agent runner and tool registry
from agent import run_agent
from tools.registry import TOOLS

# CONFIG
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)
DEFAULT_TIMEOUT = 60  # seconds for /v1/query if not provided

app = FastAPI(title="Agnikul Agent API", version="0.1")


class QueryRequest(BaseModel):
    query: str
    timeout: Optional[int] = DEFAULT_TIMEOUT


def _worker_run(question: str, out_q: Queue):
    """Worker process target to call run_agent and send result back via queue."""
    try:
        result = run_agent(question)
        out_q.put({"ok": True, "result": result})
    except Exception:
        out_q.put({
            "ok": False,
            "error": traceback.format_exc()
        })


def run_agent_with_timeout(question: str, timeout: int):
    """Spawn a process to run the agent and kill it if it exceeds timeout."""
    q: Queue = multiprocessing.Queue()
    p: Process = multiprocessing.Process(target=_worker_run, args=(question, q), daemon=True)
    p.start()
    try:
        payload = q.get(timeout=timeout)
        # ensure process cleaned up
        p.join(timeout=1)
        if payload.get("ok"):
            return {"status": "ok", "response": payload.get("result")}
        else:
            return {"status": "error", "error": payload.get("error")}
    except Exception as e:
        # Timeout or other errors — ensure child process is terminated
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)
        if isinstance(e, multiprocessing.queues.Empty) or isinstance(e, TimeoutError):
            return {"status": "timeout", "error": f"Agent timed out after {timeout} seconds."}
        return {"status": "error", "error": str(e)}


@app.post("/v1/query")
def query_endpoint(req: QueryRequest):
    """
    Synchronous query endpoint.
    Body: { "query": "<text>", "timeout": <seconds, optional> }
    """
    if not req.query or not isinstance(req.query, str):
        raise HTTPException(status_code=400, detail="`query` must be a non-empty string.")

    # unique request id for tracing
    request_id = str(uuid.uuid4())
    timeout = int(req.timeout or DEFAULT_TIMEOUT)

    # Run agent in a separate process and kill on timeout
    result = run_agent_with_timeout(req.query, timeout)

    body = {
        "request_id": request_id,
        "timeout_seconds": timeout,
        **result
    }

    status_code = 200 if result.get("status") == "ok" else 500 if result.get("status") == "error" else 504
    return JSONResponse(status_code=status_code, content=body)


@app.get("/v1/tools")
def tools_list():
    """List available tools (name + description)."""
    tools = [{"name": t.name, "description": getattr(t, "description", "")} for t in TOOLS]
    return {"tools": tools}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file. Returns the absolute filepath where the file is saved.
    Minimal behavior: file saved to uploads/<uuid>/<original_filename>.
    NOTE: For security, do not expose this endpoint publicly without auth and size limits.
    """
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    # Create a request-scoped folder
    upload_id = str(uuid.uuid4())
    dest_dir = UPLOADS_DIR / upload_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(file.filename).name

    # Save file
    try:
        with dest_path.open("wb") as f:
            contents = await file.read()
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Return the absolute path (you can pass this path as input to tools like bibtex)
    return {"upload_id": upload_id, "filepath": str(dest_path.resolve())}
