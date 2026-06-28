import asyncio
import json
import os
import re
import shutil
import time
import uuid
from typing import List, Optional

import redis.asyncio as aioredis
from redis.exceptions import RedisError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="DoMD FF server")
app.mount("/static", StaticFiles(directory="static"), name="static")

WORKSPACE_BASE = os.getenv("WORKSPACE_BASE", "./workspaces")
TASK_QUEUE = os.getenv("TASK_QUEUE", "md_task_queue")
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

TASK_ID_RE = re.compile(r"^task_[a-f0-9]{8}(?:[a-f0-9]{24})?$")
CLIENT_TASK_ID_RE = re.compile(r"^task_[a-f0-9]{32}$")
TERMINAL_STATES = {"SUCCESS", "PARTIAL", "ERROR"}

os.makedirs(WORKSPACE_BASE, exist_ok=True)

redis_client = aioredis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    health_check_interval=30,
    socket_keepalive=True,
    socket_connect_timeout=5,
    retry_on_timeout=True,
)


def now_ts() -> int:
    return int(time.time())


def make_task_id() -> str:
    return f"task_{uuid.uuid4().hex}"


def is_valid_task_id(task_id: str) -> bool:
    # Status/download/SSE stay backward-compatible with old 8-hex server-generated ids.
    return bool(TASK_ID_RE.fullmatch(task_id or ""))


def normalize_upload_task_id(task_id: Optional[str]) -> str:
    if task_id is None or not str(task_id).strip():
        return make_task_id()

    normalized = str(task_id).strip().lower()
    if not CLIENT_TASK_ID_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id format. Expected task_<32 lowercase hex characters>.",
        )
    return normalized


def task_claim_key(task_id: str) -> str:
    return f"task_claim_{task_id}"


def task_meta_key(task_id: str) -> str:
    return f"task_meta_{task_id}"


def task_status_key(task_id: str) -> str:
    # This key stores terminal states only, for backward compatibility with SSE.
    return f"task_status_{task_id}"


def result_zip_path(task_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")


def local_status_path(task_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, task_id, "task_status.json")


def safe_upload_name(filename: Optional[str]) -> str:
    filename = os.path.basename(filename or "upload.bin")
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)
    return filename or "upload.bin"


def make_task_meta(task_id: str, state: str, message: str, work_dir: str, created_at: Optional[int] = None) -> dict:
    created_at = created_at or now_ts()
    terminal = state in TERMINAL_STATES
    has_result = os.path.exists(result_zip_path(task_id))
    return {
        "task_id": task_id,
        "state": state,
        "terminal": terminal,
        "message": message,
        "created_at": created_at,
        "updated_at": now_ts(),
        "expires_at": created_at + TASK_TTL_SECONDS,
        "work_dir": work_dir,
        "has_result": has_result,
        "download_url": f"/api/download/{task_id}" if terminal and has_result else None,
    }


def read_local_status_if_fresh(task_id: str) -> Optional[dict]:
    path = local_status_path(task_id)
    if not os.path.exists(path):
        return None

    try:
        if now_ts() - int(os.path.getmtime(path)) > TASK_TTL_SECONDS:
            return None
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    state = meta.get("state", "UNKNOWN")
    terminal = state in TERMINAL_STATES or bool(meta.get("terminal"))
    has_result = os.path.exists(result_zip_path(task_id))
    meta.update({
        "task_id": task_id,
        "terminal": terminal,
        "has_result": has_result,
        "download_url": f"/api/download/{task_id}" if terminal and has_result else None,
        "source": "local_status_file",
    })
    return meta


def not_found_payload(task_id: str) -> dict:
    return {
        "status": "success",
        "task_id": task_id,
        "task_state": "NOT_FOUND",
        "terminal": True,
        "has_result": False,
        "download_url": None,
        "message": "Task not found or expired.",
        "expires_in": 0,
        "source": "not_found",
    }


async def get_task_status_payload(task_id: str) -> dict:
    if not is_valid_task_id(task_id):
        return not_found_payload(task_id)

    try:
        meta_raw = await redis_client.get(task_meta_key(task_id))
        ttl = await redis_client.ttl(task_meta_key(task_id))
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
            except json.JSONDecodeError:
                meta = {}

            state = meta.get("state") or "UNKNOWN"
            terminal = state in TERMINAL_STATES or bool(meta.get("terminal"))
            has_result = os.path.exists(result_zip_path(task_id))
            return {
                "status": "success",
                "task_id": task_id,
                "task_state": state,
                "terminal": terminal,
                "has_result": has_result,
                "download_url": f"/api/download/{task_id}" if terminal and has_result else None,
                "message": meta.get("message", ""),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "expires_at": meta.get("expires_at"),
                "expires_in": max(ttl, 0) if ttl is not None else None,
                "source": "redis_meta",
            }

        terminal_state = await redis_client.get(task_status_key(task_id))
        if terminal_state:
            has_result = os.path.exists(result_zip_path(task_id))
            return {
                "status": "success",
                "task_id": task_id,
                "task_state": terminal_state,
                "terminal": terminal_state in TERMINAL_STATES,
                "has_result": has_result,
                "download_url": f"/api/download/{task_id}" if has_result else None,
                "message": f"Task finished with status {terminal_state}.",
                "expires_in": max(await redis_client.ttl(task_status_key(task_id)), 0),
                "source": "redis_terminal_status",
            }

    except RedisError:
        local_status = read_local_status_if_fresh(task_id)
        if local_status:
            state = local_status.get("state", "UNKNOWN")
            return {
                "status": "success",
                "task_id": task_id,
                "task_state": state,
                "terminal": state in TERMINAL_STATES or bool(local_status.get("terminal")),
                "has_result": local_status.get("has_result", False),
                "download_url": local_status.get("download_url"),
                "message": local_status.get("message", "Recovered from local status file."),
                "created_at": local_status.get("created_at"),
                "updated_at": local_status.get("updated_at"),
                "expires_at": local_status.get("expires_at"),
                "expires_in": None,
                "source": "local_status_file",
            }
        raise

    local_status = read_local_status_if_fresh(task_id)
    if local_status:
        state = local_status.get("state", "UNKNOWN")
        return {
            "status": "success",
            "task_id": task_id,
            "task_state": state,
            "terminal": state in TERMINAL_STATES or bool(local_status.get("terminal")),
            "has_result": local_status.get("has_result", False),
            "download_url": local_status.get("download_url"),
            "message": local_status.get("message", "Recovered from local status file."),
            "created_at": local_status.get("created_at"),
            "updated_at": local_status.get("updated_at"),
            "expires_at": local_status.get("expires_at"),
            "expires_in": None,
            "source": "local_status_file",
        }

    return not_found_payload(task_id)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not os.path.exists("templates/index.html"):
        return "<h1>Error: templates/index.html not found in current directory.</h1>"
    return FileResponse("templates/index.html")


@app.post("/api/upload_and_run")
async def upload_and_run(
    files: List[UploadFile] = File(...),
    params_json: str = Form(...),
    task_id: Optional[str] = Form(None),
):
    task_id = normalize_upload_task_id(task_id)
    work_dir = os.path.join(WORKSPACE_BASE, task_id)

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return {"status": "error", "error": "Invalid params_json payload."}

    try:
        claimed = await redis_client.set(
            task_claim_key(task_id),
            json.dumps({"task_id": task_id, "created_at": now_ts()}),
            nx=True,
            ex=TASK_TTL_SECONDS,
        )
    except RedisError as exc:
        return {"status": "error", "error": f"Task queue unavailable: {exc}"}

    if not claimed:
        return {
            "status": "error",
            "task_id": task_id,
            "error": "Task id already exists. Please generate a new task and submit again.",
        }

    file_paths = {
        "mol_file_path": None,
        "index_file_path": None,
    }

    try:
        os.makedirs(work_dir, exist_ok=False)

        for file in files:
            filename = safe_upload_name(file.filename)
            filepath = os.path.join(work_dir, filename)
            with open(filepath, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            lower_name = filename.lower()
            if lower_name.endswith((".pdb", ".sdf")):
                file_paths["mol_file_path"] = filepath
            elif lower_name.endswith(".idx"):
                file_paths["index_file_path"] = filepath

        if not file_paths["mol_file_path"]:
            await redis_client.delete(task_claim_key(task_id))
            shutil.rmtree(work_dir, ignore_errors=True)
            return {"status": "error", "error": "Missing .pdb or .sdf file."}

        created_at = now_ts()
        task_payload = {
            "task_id": task_id,
            "file_paths": file_paths,
            "params": params,
            "work_dir": work_dir,
            "created_at": created_at,
        }
        task_meta = make_task_meta(
            task_id=task_id,
            state="QUEUED",
            message="Task queued and waiting for a worker.",
            work_dir=work_dir,
            created_at=created_at,
        )

        pipe = redis_client.pipeline(transaction=True)
        pipe.set(task_meta_key(task_id), json.dumps(task_meta), ex=TASK_TTL_SECONDS)
        pipe.expire(task_claim_key(task_id), TASK_TTL_SECONDS)
        pipe.rpush(TASK_QUEUE, json.dumps(task_payload))
        await pipe.execute()

    except RedisError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            await redis_client.delete(task_claim_key(task_id), task_meta_key(task_id), task_status_key(task_id))
        except RedisError:
            pass
        return {"status": "error", "error": f"Task queue unavailable: {exc}"}
    except FileExistsError:
        try:
            await redis_client.delete(task_claim_key(task_id))
        except RedisError:
            pass
        return {
            "status": "error",
            "task_id": task_id,
            "error": "Workspace already exists for this task id. Please submit a new task.",
        }
    except OSError as exc:
        shutil.rmtree(work_dir, ignore_errors=True)
        try:
            await redis_client.delete(task_claim_key(task_id))
        except RedisError:
            pass
        return {"status": "error", "error": f"Could not store uploaded files: {exc}"}

    return {
        "status": "success",
        "task_id": task_id,
        "task_state": "QUEUED",
        "expires_in": TASK_TTL_SECONDS,
    }


@app.get("/api/task_status/{task_id}")
async def task_status(task_id: str):
    return await get_task_status_payload(task_id)


@app.get("/api/stream_logs/{task_id}")
async def stream_logs(task_id: str):
    async def event_generator():
        if not is_valid_task_id(task_id):
            yield {"data": "ERROR: Invalid task id."}
            yield {"data": "[[DONE_ERROR]]"}
            return

        pubsub = redis_client.pubsub()
        channel_name = f"log_channel_{task_id}"
        await pubsub.subscribe(channel_name)

        try:
            status_payload = await get_task_status_payload(task_id)
            task_state = status_payload.get("task_state")
            if task_state in TERMINAL_STATES:
                yield {"data": f"[[DONE_{task_state}]]"}
                return

            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15.0)

                if message is not None:
                    data = message["data"]
                    yield {"data": data}
                    if isinstance(data, str) and data.startswith("[[DONE_"):
                        break
                else:
                    yield {"event": "ping", "data": "keepalive"}

        except asyncio.CancelledError:
            pass
        finally:
            try:
                await pubsub.unsubscribe(channel_name)
                await pubsub.close()
            except Exception:
                pass

    return EventSourceResponse(event_generator())


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    if not is_valid_task_id(task_id):
        return {"error": "Invalid task id."}

    status_payload = await get_task_status_payload(task_id)
    if status_payload.get("task_state") == "NOT_FOUND":
        return {"error": "Task not found or expired."}

    zip_path = result_zip_path(task_id)
    if not os.path.exists(zip_path):
        return {"error": "Result archive does not exist or the calculation failed."}

    return FileResponse(
        path=zip_path,
        filename=f"{task_id}_result.zip",
        media_type="application/zip",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
