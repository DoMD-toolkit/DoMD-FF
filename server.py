import asyncio
import contextlib
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
import copy
from rdkit.Chem import rdMolDescriptors, rdMolHash

from ForceField import FF
from misc.io.gmx import write_gro_file, write_top_file, write_list_itp_files
from misc.logger import task_file_log_scope, mol_file_log_scope
from misc.parser import molecule_reader, molecule_reader_list

app = FastAPI(title="P2P FF parameterizer server")

WORKSPACE_BASE = "./workspaces"
os.makedirs(WORKSPACE_BASE, exist_ok=True)

task_log_queues = {}


@contextlib.contextmanager
def sub_molecule_log_scope(logger_to_hook, log_filepath):
    """
    临时为指定的 logger 挂载一个文件 Handler。
    退出 with 块时自动卸载，绝不污染后续的日志。
    """
    file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))

    logger_to_hook.addHandler(file_handler)
    try:
        yield log_filepath
    finally:
        logger_to_hook.removeHandler(file_handler)
        file_handler.close()


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
    web_logger = setup_task_logger(task_id, log_queue, loop)

    with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
        web_logger.info(f"Starting TASK {task_id}...")
        web_logger.info(f"Parameters: useGMX: {params.get('useGMX')} useBOSS: {params.get('useBOSS')}"
                        f" useML: {params.get('useML')} overwrite: {params.get('overwrite')} "
                        f"charge_factor: {params.get('charge_factor')}")
        try:
            web_logger.info(f"Molecular File: {file_paths.get('mol_file_path')}")
            if params.get("run_mode") == "top_mode":
                output_gro_path = os.path.join(work_dir, "output.gro")
                output_top_path = os.path.join(work_dir, "output.top")
                obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(
                    file_paths.get('mol_file_path'))
                web_logger.info(f"Staring to set OPLS force field for system {file_paths.get('mol_file_path')}")
                forcefield = FF('opls')
                forcefield.setup(rdmol, obmol, useGMX=params.get("useGMX"),
                                 useBOSS=params.get("useBOSS"), useML=params.get("useML"),
                                 overwrite=params.get("overwrite"), charge_factor=params.get("charge_factor"))
                if not forcefield.success:
                    raise ValueError("Force field parametrization failed, please check the log files.")

                # output
                web_logger.info("Writing GRO file...")
                write_gro_file(output_gro_path, coordinates, res_names, res_ids, box_tensor)
                web_logger.info("Writing TOP file...")
                write_top_file(output_top_path, forcefield, res_names, res_ids)

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(output_gro_path, arcname="output.gro")
                    zipf.write(output_top_path, arcname="output.top")
                    zipf.write(debug_log_path, arcname="debug.log")

            elif params.get("run_mode") == "itp_mode":
                _cache = {}
                atomtypes_path = os.path.join(work_dir, "atomtypes.itp")
                mol_list = molecule_reader_list(file_paths.get('mol_file_path'))
                itp_fns = []
                forcefields = []
                mol_names = []
                mol_log_paths = []
                web_logger.info(f"--- Total number of molecule fragments {len(mol_list)} ---")
                num_success = 0
                for idx, mol in enumerate(mol_list):
                    wl_hash = rdMolHash.MolHash(mol, rdMolHash.HashFunction.AnonymousGraph)
                    ret = _cache.get(wl_hash)
                    if ret is not None:
                        if not ret['notfound']:
                            forcefield = FF('opls')
                            forcefield.params = ret['params']
                            forcefield.charges = ret['charges']
                            itp_fns.append(os.path.join(work_dir, f"{idx:06d}_{ret['idx']:06d}.itp"))
                            mol_names.append(f"{idx:06d}")
                            forcefields.append(forcefield)
                            web_logger.info(f"Molecule {idx:06d} parametrization success (cache {ret['idx']}).")
                            num_success += 1
                        else:
                            web_logger.info(f"Molecule {idx:06d} parametrization failed (cache {ret['idx']}).")
                    else:
                        with mol_file_log_scope(idx, work_dir) as mol_log_path:
                            mol_log_paths.append(mol_log_path)
                            web_logger.info(f"--- Starting parametrization for Molecule {idx:06d} ---")
                            forcefield = FF('opls')
                            forcefield.setup(mol, obmol=None, useGMX=params.get("useGMX"),
                                             useBOSS=params.get("useBOSS"), useML=params.get("useML"),
                                             overwrite=params.get("overwrite"), charge_factor=params.get("charge_factor"))
                            if not forcefield.success:
                                web_logger.error(f"Force field for molecule {idx:06d} parametrization "
                                                 f"failed, please check this log file.")
                                _cache[wl_hash] = {'notfound': True, 'idx': idx}
                            else:
                                itp_fns.append(os.path.join(work_dir, f"{idx:06d}.itp"))
                                mol_names.append(f"{idx:06d}")
                                forcefields.append(forcefield)
                                web_logger.info(f"Molecule {idx:06d} parametrization success.")
                                num_success += 1
                                _cache[wl_hash] = {'params': copy.deepcopy(forcefield.params),
                                                   'charges': copy.deepcopy(forcefield.charges),
                                                   'idx': idx,
                                                   'notfound': False}
                if num_success == len(mol_list):
                    compute_status = 'SUCCESS'
                if num_success == 0:
                    compute_status = 'ERROR'
                if 0 < num_success < len(mol_list):
                    compute_status = "PARTIAL"
                web_logger.info("Writing ITP files...")
                write_list_itp_files(itp_fns, forcefields, mol_names)
                web_logger.info("Packaging ITP results and logs into ZIP archive...")
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    if os.path.exists(debug_log_path):
                        zipf.write(debug_log_path, arcname="debug_master.log")
                    for itp_file in itp_fns:
                        if os.path.exists(itp_file):
                            zipf.write(itp_file, arcname=os.path.basename(itp_file))
                    for ml_log in mol_log_paths:
                        if os.path.exists(ml_log):
                            zipf.write(ml_log, arcname=os.path.basename(ml_log))
                    zipf.write(atomtypes_path, arcname=os.path.basename(atomtypes_path))
                web_logger.info("Zip package created successfully.")

        except Exception as e:
            compute_status = "ERROR"
            web_logger.error(f"Error: {str(e)}")
            web_logger.debug(traceback.format_exc())

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(debug_log_path, arcname="debug_error.log")

    loop.call_soon_threadsafe(log_queue.put_nowait, f"[[DONE_{compute_status}]]")
    # shutil.rmtree(work_dir, ignore_errors=True)


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
        return {"error": "File not found."}

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

    uvicorn.run(app, host="0.0.0.0", port=8000)
