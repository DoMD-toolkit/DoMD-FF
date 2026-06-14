import asyncio
import json
import logging
import os
import shutil
import traceback
import uuid
import zipfile
from typing import List

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

from ForceField import FF
from misc.io.gmx import write_gro_file, write_top_file
from misc.logger import task_file_log_scope
from misc.parser import molecule_reader

app = FastAPI(title="P2P FF parameterizer server")

WORKSPACE_BASE = "./workspaces"
os.makedirs(WORKSPACE_BASE, exist_ok=True)

task_log_queues = {}


def setup_task_logger(task_id: str, log_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    class AsyncQueueHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.INFO:
                msg = self.format(record)
                loop.call_soon_threadsafe(log_queue.put_nowait, msg)

    queue_handler = AsyncQueueHandler()
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(queue_handler)
    return logger


def run_heavy_compute(task_id: str, file_paths: dict, params: dict, work_dir: str, log_queue: asyncio.Queue,
                      loop: asyncio.AbstractEventLoop):
    # 记录状态
    compute_status = "SUCCESS"
    zip_path = os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")
    output_gro_path = os.path.join(work_dir, "output.gro")
    output_top_path = os.path.join(work_dir, "output.top")
    web_logger = setup_task_logger(task_id, log_queue, loop)

    with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
        try:
            web_logger.info(f"Molecular File: {file_paths.get('mol_file_path')}")

            obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(file_paths.get('mol_file_path'))

            web_logger.info(f"Staring to set OPLS force field for system {file_paths.get('mol_file_path')}")
            forcefield = FF('opls')
            forcefield.setup(rdmol, obmol, useGMX=params.get("useGMX"),
                             useBOSS=params.get("useBOSS"), useML=params.get("useML"))
            if not forcefield.success:
                raise ValueError("Force field parametrization failed, please check the log files.")

            # output
            web_logger.info("Writing GRO file...")
            write_gro_file(output_gro_path, coordinates, res_names, res_ids, box_tensor)
            web_logger.info("Writing TOP file...")
            params_atom, params_bonded, params_improper = forcefield.params
            write_top_file(output_top_path, params_atom, params_bonded, params_improper, res_names, res_ids)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(output_gro_path, arcname="output.gro")
                zipf.write(output_top_path, arcname="output.top")
                zipf.write(debug_log_path, arcname="debug.log")

        except Exception as e:
            compute_status = "ERROR"
            web_logger.error(f"Error: {str(e)}")
            web_logger.debug(traceback.format_exc())

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(debug_log_path, arcname="debug_error.log")

    loop.call_soon_threadsafe(log_queue.put_nowait, f"[[DONE_{compute_status}]]")
    shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not os.path.exists("templates/index.html"):
        return "<h1>Error: index.html not found in current directory!</h1>"
    return FileResponse("templates/index.html")


@app.post("/api/upload_and_run")
async def upload_and_run(files: List[UploadFile] = File(...), params_json: str = Form(...)):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    work_dir = os.path.join(WORKSPACE_BASE, task_id)
    os.makedirs(work_dir, exist_ok=True)

    file_paths = {
        "mol_file_path": None,
        "index_file_path": None
    }

    for file in files:
        filepath = os.path.join(work_dir, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.lower().endswith(('.pdb', '.sdf')):
            file_paths["mol_file_path"] = filepath
        elif file.filename.lower().endswith('.idx'):
            file_paths["index_file_path"] = filepath

    params = json.loads(params_json)

    if not file_paths["mol_file_path"]:
        return {"status": "error", "error": "Missing .pdb or .sdf file"}

    log_queue = asyncio.Queue()
    task_log_queues[task_id] = log_queue
    loop = asyncio.get_running_loop()

    asyncio.create_task(asyncio.to_thread(
        run_heavy_compute, task_id, file_paths, params, work_dir, log_queue, loop
    ))

    return {"status": "success", "task_id": task_id}


@app.get("/api/stream_logs/{task_id}")
async def stream_logs(task_id: str):
    queue = task_log_queues.get(task_id)
    if not queue:
        return {"error": "任务不存在或已清理"}

    async def event_generator():
        try:
            while True:
                msg = await queue.get()
                if msg == "[[DONE]]":
                    yield {"data": "[[DONE]]"}
                    break
                yield {"data": msg}
        finally:
            task_log_queues.pop(task_id, None)

    return EventSourceResponse(event_generator())


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    zip_path = os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")
    if not os.path.exists(zip_path):
        return {"error": "File not exists or calculation failed."}

    return FileResponse(
        path=zip_path,
        filename=f"{task_id}_result.zip",
        media_type="application/zip"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
