import asyncio
import copy
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from rdkit.Chem import rdMolHash
from sse_starlette.sse import EventSourceResponse

from ForceField import FF
from lib import print_opls_stats
from misc.io.gmx import write_gro_file, write_top_file, write_list_itp_files
from misc.logger import task_file_log_scope, mol_file_log_scope
from misc.parser import molecule_reader, molecule_reader_list


@asynccontextmanager
async def lifespan(app: FastAPI):
    global job_queue, worker_task
    job_queue = asyncio.Queue()
    worker_task = asyncio.create_task(inprocess_worker())

    try:
        yield
    finally:
        if worker_task is not None:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

app = FastAPI(title="DoMD FF local server", lifespan=lifespan)

# Keep this single-process server easy to run even when the static directory is empty.
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

WORKSPACE_BASE = os.getenv("WORKSPACE_BASE", "./workspaces")
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))

TASK_ID_RE = re.compile(r"^task_[a-f0-9]{32}$")
TERMINAL_STATES = {"SUCCESS", "PARTIAL", "ERROR"}
NON_TERMINAL_STATES = {"QUEUED", "RUNNING"}

os.makedirs(WORKSPACE_BASE, exist_ok=True)

# In-process equivalents of the deploy version's Redis queue, task_meta records, and log pub/sub.
task_meta_store: Dict[str, dict] = {}
task_claims: Dict[str, int] = {}
task_log_queues: Dict[str, asyncio.Queue] = {}
store_lock = threading.RLock()

job_queue: Optional[asyncio.Queue] = None
worker_task: Optional[asyncio.Task] = None


def now_ts() -> int:
    return int(time.time())


def make_task_id() -> str:
    return f"task_{uuid.uuid4().hex}"


def is_valid_task_id(task_id: str) -> bool:
    return bool(TASK_ID_RE.fullmatch(task_id or ""))


def normalize_upload_task_id(task_id: Optional[str]) -> str:
    if task_id is None or not str(task_id).strip():
        return make_task_id()

    normalized = str(task_id).strip().lower()
    if not TASK_ID_RE.fullmatch(normalized):
        raise HTTPException(
            status_code=400,
            detail="Invalid task_id format. Expected task_<32 lowercase hex characters>.",
        )
    return normalized


def result_zip_path(task_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")


def local_meta_path(task_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, task_id, "task_meta.json")


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


def write_local_meta(work_dir: str, meta: dict) -> None:
    if not work_dir:
        return

    try:
        os.makedirs(work_dir, exist_ok=True)
        status_path = os.path.join(work_dir, "task_meta.json")
        tmp_path = f"{status_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, status_path)
    except OSError as exc:
        print(f"[WARN] Could not write local task metadata file: {exc}", flush=True)


def read_local_terminal_meta_if_fresh(task_id: str) -> Optional[dict]:
    """
    After a local debug server restart, in-memory queued/running jobs are gone.
    Therefore local fallback only trusts terminal metadata. This avoids showing a
    stale RUNNING task that is no longer actually executing.
    """
    path = local_meta_path(task_id)
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
    if state not in TERMINAL_STATES:
        return None

    has_result = os.path.exists(result_zip_path(task_id))
    meta.update({
        "task_id": task_id,
        "terminal": True,
        "has_result": has_result,
        "download_url": f"/api/download/{task_id}" if has_result else None,
        "source": "local_meta_file",
    })
    return meta


def update_task_meta(task_id: str, state: str, message: str, work_dir: str, terminal: bool = False) -> dict:
    with store_lock:
        existing = task_meta_store.get(task_id, {})
        created_at = int(existing.get("created_at", now_ts()))
        has_result = os.path.exists(result_zip_path(task_id))
        meta = {
            **existing,
            "task_id": task_id,
            "state": state,
            "terminal": terminal or state in TERMINAL_STATES,
            "message": message,
            "created_at": created_at,
            "updated_at": now_ts(),
            "expires_at": created_at + TASK_TTL_SECONDS,
            "work_dir": work_dir,
            "has_result": has_result,
            "download_url": f"/api/download/{task_id}" if (terminal or state in TERMINAL_STATES) and has_result else None,
        }
        task_meta_store[task_id] = meta

    write_local_meta(work_dir, meta)
    return meta


def build_status_payload(task_id: str, meta: dict, source: str) -> dict:
    state = meta.get("state") or "UNKNOWN"
    terminal = state in TERMINAL_STATES or bool(meta.get("terminal"))
    has_result = os.path.exists(result_zip_path(task_id))
    expires_at = meta.get("expires_at")
    expires_in = max(int(expires_at - now_ts()), 0) if isinstance(expires_at, (int, float)) else None

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
        "expires_at": expires_at,
        "expires_in": expires_in,
        "source": source,
    }


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


def cleanup_expired_memory() -> None:
    current = now_ts()
    expired_task_ids = []

    with store_lock:
        for task_id, meta in task_meta_store.items():
            expires_at = int(meta.get("expires_at", 0))
            if expires_at and current > expires_at:
                expired_task_ids.append(task_id)

        for task_id in expired_task_ids:
            task_meta_store.pop(task_id, None)
            task_claims.pop(task_id, None)
            task_log_queues.pop(task_id, None)

        expired_claims = [
            task_id for task_id, created_at in task_claims.items()
            if current - int(created_at) > TASK_TTL_SECONDS
        ]
        for task_id in expired_claims:
            task_claims.pop(task_id, None)


async def get_task_status_payload(task_id: str) -> dict:
    if not is_valid_task_id(task_id):
        return not_found_payload(task_id)

    cleanup_expired_memory()

    with store_lock:
        meta = copy.deepcopy(task_meta_store.get(task_id))

    if meta:
        return build_status_payload(task_id, meta, "memory_meta")

    local_meta = read_local_terminal_meta_if_fresh(task_id)
    if local_meta:
        return build_status_payload(task_id, local_meta, "local_meta_file")

    return not_found_payload(task_id)


def get_log_queue(task_id: str) -> Optional[asyncio.Queue]:
    with store_lock:
        return task_log_queues.get(task_id)


def emit_task_log(task_id: str, message: str, loop: asyncio.AbstractEventLoop) -> None:
    queue = get_log_queue(task_id)
    if queue is None:
        return
    loop.call_soon_threadsafe(queue.put_nowait, message)


def setup_task_logger(task_id: str, log_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    class AsyncQueueHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.INFO:
                msg = self.format(record)
                loop.call_soon_threadsafe(log_queue.put_nowait, msg)

    queue_handler = AsyncQueueHandler()
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(queue_handler)
    return logger


def close_task_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def finalize_task(task_id: str, state: str, work_dir: str, message: str, loop: asyncio.AbstractEventLoop) -> None:
    update_task_meta(task_id, state, message, work_dir, terminal=True)
    emit_task_log(task_id, f"[[DONE_{state}]]", loop)


def run_heavy_compute(
    task_id: str,
    file_paths: dict,
    params: dict,
    work_dir: str,
    log_queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
):
    compute_status = "SUCCESS"
    zip_path = result_zip_path(task_id)
    web_logger = setup_task_logger(task_id, log_queue, loop)

    update_task_meta(
        task_id,
        "RUNNING",
        "Task is running on the local debug server.",
        work_dir,
        terminal=False,
    )

    try:
        with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
            web_logger.info(f"Starting task {task_id} on local debug server.")
            web_logger.info(
                f"Parameters: useGMX={params.get('useGMX')} useBOSS={params.get('useBOSS')} "
                f"useML={params.get('useML')} overwrite={params.get('overwrite')} "
                f"charge_factor={params.get('charge_factor')}"
            )

            try:
                web_logger.info(f"Molecular file: {file_paths.get('mol_file_path')}")

                if params.get("run_mode") == "top_mode":
                    output_gro_path = os.path.join(work_dir, "output.gro")
                    output_top_path = os.path.join(work_dir, "output.top")
                    obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(
                        file_paths.get("mol_file_path")
                    )
                    web_logger.info(f"Starting OPLS force-field setup for {file_paths.get('mol_file_path')}.")

                    forcefield = FF("opls")
                    forcefield.setup(
                        rdmol,
                        obmol,
                        useGMX=params.get("useGMX"),
                        useBOSS=params.get("useBOSS"),
                        useML=params.get("useML"),
                        overwrite=params.get("overwrite"),
                        charge_factor=params.get("charge_factor"),
                    )

                    print_opls_stats(forcefield, web_logger, "info" if forcefield.success else "warning")

                    if not forcefield.success:
                        raise ValueError("Force-field parameterization failed. Please check the log files.")

                    web_logger.info("Writing GRO file.")
                    atom_names = [f"{atom.GetSymbol()}" for atom in rdmol.GetAtoms()]
                    write_gro_file(output_gro_path, coordinates, box_tensor, res_names, res_ids, atom_names)
                    web_logger.info("Writing TOP file.")
                    write_top_file(output_top_path, forcefield, res_names, res_ids)

                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        zipf.write(output_gro_path, arcname="output.gro")
                        zipf.write(output_top_path, arcname="output.top")
                        if os.path.exists(debug_log_path):
                            zipf.write(debug_log_path, arcname="debug.log")

                elif params.get("run_mode") == "itp_mode":
                    cache = {}
                    atomtypes_path = os.path.join(work_dir, "atomtypes.itp")
                    mol_list = molecule_reader_list(file_paths.get("mol_file_path"))
                    itp_fns, forcefields, mol_names, mol_log_paths = [], [], [], []

                    web_logger.info(f"Total number of molecule fragments: {len(mol_list)}.")
                    num_success = 0

                    for idx, mol in enumerate(mol_list):
                        wl_hash = rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
                        ret = cache.get(wl_hash)
                        if ret is not None:
                            if not ret["notfound"]:
                                forcefield = FF("opls")
                                forcefield.params = ret["params"]
                                forcefield.charges = ret["charges"]
                                itp_fns.append(os.path.join(work_dir, f"{idx:06d}_{ret['idx']:06d}.itp"))
                                mol_names.append(f"{idx:06d}")
                                forcefields.append(forcefield)
                                web_logger.info(f"Molecule {idx:06d} parameterization succeeded using cache {ret['idx']}.")
                                num_success += 1
                            else:
                                web_logger.info(f"Molecule {idx:06d} parameterization failed using cache {ret['idx']}.")
                        else:
                            with mol_file_log_scope(idx, work_dir) as mol_log_path:
                                mol_log_paths.append(mol_log_path)
                                web_logger.info(f"Starting parameterization for molecule {idx:06d}.")

                                forcefield = FF("opls")
                                forcefield.setup(
                                    mol,
                                    obmol=None,
                                    useGMX=params.get("useGMX"),
                                    useBOSS=params.get("useBOSS"),
                                    useML=params.get("useML"),
                                    overwrite=params.get("overwrite"),
                                    charge_factor=params.get("charge_factor"),
                                )

                                if not forcefield.success:
                                    web_logger.error(f"Force-field parameterization failed for molecule {idx:06d}.")
                                    cache[wl_hash] = {"notfound": True, "idx": idx}
                                else:
                                    itp_fns.append(os.path.join(work_dir, f"{idx:06d}.itp"))
                                    mol_names.append(f"{idx:06d}")
                                    forcefields.append(forcefield)
                                    web_logger.info(f"Molecule {idx:06d} parameterization succeeded.")
                                    num_success += 1
                                    cache[wl_hash] = {
                                        "params": copy.deepcopy(forcefield.params),
                                        "charges": copy.deepcopy(forcefield.charges),
                                        "idx": idx,
                                        "notfound": False,
                                    }
                                print_opls_stats(forcefield, web_logger, "info" if forcefield.success else "warning")

                    if num_success == len(mol_list):
                        compute_status = "SUCCESS"
                    elif num_success == 0:
                        compute_status = "ERROR"
                    else:
                        compute_status = "PARTIAL"

                    if num_success > 0:
                        web_logger.info("Writing ITP files.")
                        write_list_itp_files(itp_fns, forcefields, mol_names)

                    web_logger.info("Packaging ITP results and logs into ZIP archive.")
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                        if os.path.exists(debug_log_path):
                            zipf.write(debug_log_path, arcname="debug_master.log")
                        for itp_file in itp_fns:
                            if os.path.exists(itp_file):
                                zipf.write(itp_file, arcname=os.path.basename(itp_file))
                        for ml_log in mol_log_paths:
                            if os.path.exists(ml_log):
                                zipf.write(ml_log, arcname=os.path.basename(ml_log))
                        if os.path.exists(atomtypes_path):
                            zipf.write(atomtypes_path, arcname=os.path.basename(atomtypes_path))
                    web_logger.info("ZIP archive created successfully.")

                else:
                    raise ValueError(f"Unsupported run_mode: {params.get('run_mode')}")

            except Exception as exc:
                compute_status = "ERROR"
                web_logger.error(f"Error: {str(exc)}")
                print(traceback.format_exc(), flush=True)

                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    if "debug_log_path" in locals() and os.path.exists(debug_log_path):
                        zipf.write(debug_log_path, arcname="debug_error.log")
                    else:
                        zipf.writestr("debug_error.log", str(exc))

    finally:
        finalize_task(
            task_id,
            compute_status,
            work_dir,
            f"Task finished with status {compute_status}.",
            loop,
        )
        close_task_logger(web_logger)


async def inprocess_worker() -> None:
    global job_queue

    loop = asyncio.get_running_loop()
    print("[SYSTEM] Local in-process worker online. Waiting for tasks.", flush=True)

    while True:
        payload = await job_queue.get()
        task_id = payload["task_id"]
        work_dir = payload["work_dir"]
        created_at = int(payload.get("created_at", now_ts()))

        try:
            if now_ts() - created_at > TASK_TTL_SECONDS:
                update_task_meta(
                    task_id,
                    "ERROR",
                    "Task expired before the local worker could start it.",
                    work_dir,
                    terminal=True,
                )
                emit_task_log(task_id, "[[DONE_ERROR]]", loop)
                continue

            queue = get_log_queue(task_id)
            if queue is None:
                queue = asyncio.Queue()
                with store_lock:
                    task_log_queues[task_id] = queue

            print(f"[SYSTEM] Local task {task_id} received. Executing.", flush=True)
            await asyncio.to_thread(
                run_heavy_compute,
                task_id,
                payload["file_paths"],
                payload["params"],
                work_dir,
                queue,
                loop,
            )
            print(f"[SYSTEM] Local task {task_id} completed.", flush=True)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[ERROR] Local worker task loop error: {exc}", flush=True)
            update_task_meta(
                task_id,
                "ERROR",
                f"Local worker task loop error: {exc}",
                work_dir,
                terminal=True,
            )
            emit_task_log(task_id, "[[DONE_ERROR]]", loop)
        finally:
            job_queue.task_done()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    for candidate in (Path("templates/index.html"), Path("index.html")):
        if candidate.exists():
            return FileResponse(str(candidate))
    return "<h1>Error: templates/index.html or index.html not found in current directory.</h1>"


@app.post("/api/upload_and_run")
async def upload_and_run(
    files: List[UploadFile] = File(...),
    params_json: str = Form(...),
    task_id: Optional[str] = Form(None),
):
    cleanup_expired_memory()

    task_id = normalize_upload_task_id(task_id)
    work_dir = os.path.join(WORKSPACE_BASE, task_id)

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        return {"status": "error", "error": "Invalid params_json payload."}

    with store_lock:
        if task_id in task_claims or task_id in task_meta_store:
            return {
                "status": "error",
                "task_id": task_id,
                "error": "Task id already exists. Please generate a new task and submit again.",
            }
        task_claims[task_id] = now_ts()

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
            with store_lock:
                task_claims.pop(task_id, None)
            shutil.rmtree(work_dir, ignore_errors=True)
            return {"status": "error", "error": "Missing .pdb or .sdf file."}

        created_at = now_ts()
        task_meta = make_task_meta(
            task_id=task_id,
            state="QUEUED",
            message="Task queued and waiting for the local worker.",
            work_dir=work_dir,
            created_at=created_at,
        )
        log_queue = asyncio.Queue()

        with store_lock:
            task_meta_store[task_id] = task_meta
            task_log_queues[task_id] = log_queue

        write_local_meta(work_dir, task_meta)

        payload = {
            "task_id": task_id,
            "file_paths": file_paths,
            "params": params,
            "work_dir": work_dir,
            "created_at": created_at,
        }

        if job_queue is None:
            raise RuntimeError("Local worker queue is not initialized.")

        await job_queue.put(payload)

    except FileExistsError:
        with store_lock:
            task_claims.pop(task_id, None)
        return {
            "status": "error",
            "task_id": task_id,
            "error": "Workspace already exists for this task id. Please submit a new task.",
        }
    except OSError as exc:
        with store_lock:
            task_claims.pop(task_id, None)
            task_meta_store.pop(task_id, None)
            task_log_queues.pop(task_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "error": f"Could not store uploaded files: {exc}"}
    except Exception as exc:
        with store_lock:
            task_claims.pop(task_id, None)
            task_meta_store.pop(task_id, None)
            task_log_queues.pop(task_id, None)
        shutil.rmtree(work_dir, ignore_errors=True)
        return {"status": "error", "error": f"Could not queue local task: {exc}"}

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

        status_payload = await get_task_status_payload(task_id)
        task_state = status_payload.get("task_state")
        if task_state in TERMINAL_STATES:
            yield {"data": f"[[DONE_{task_state}]]"}
            return
        if task_state == "NOT_FOUND":
            yield {"data": "ERROR: Task not found or expired."}
            yield {"data": "[[DONE_ERROR]]"}
            return

        queue = get_log_queue(task_id)
        if queue is None:
            yield {"data": "ERROR: Live log queue is unavailable for this task."}
            yield {"data": "[[DONE_ERROR]]"}
            return

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": "keepalive"}
                continue

            yield {"data": msg}
            if isinstance(msg, str) and msg.startswith("[[DONE_"):
                break

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
