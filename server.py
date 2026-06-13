import os
import json
import uuid
import time
import shutil
import asyncio
import logging
import zipfile
import traceback

from aiohttp.log import web_logger
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse
from typing import List
from misc.parser import molecule_reader
from misc.io.gmx import write_gro_file, write_top_file
from misc.logger import task_file_log_scope
from ForceField import FF


app = FastAPI(title="P2P FF parameterizer server")

# 工作区根目录
WORKSPACE_BASE = "./workspaces"
os.makedirs(WORKSPACE_BASE, exist_ok=True)

# 存放每个任务的 SSE 实时日志队列
task_log_queues = {}


# ==========================================
# 1. 任务专属的双路日志配置中心
# ==========================================
def setup_task_logger(task_id: str, work_dir: str, log_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    # 为每个任务创建一个独立的 logger，防止并发时日志串号
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    # [通路 B]：推送到前端队列，只放行 INFO 及以上级别
    class AsyncQueueHandler(logging.Handler):
        def emit(self, record):
            if record.levelno >= logging.INFO:
                msg = self.format(record)
                # 跨线程安全地把日志塞进异步队列
                loop.call_soon_threadsafe(log_queue.put_nowait, msg)

    queue_handler = AsyncQueueHandler()
    queue_handler.setLevel(logging.INFO)
    queue_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

    logger.addHandler(queue_handler)
    return logger


# ==========================================
# 2. 核心大计算逻辑 (在后台独立线程运行)
# ==========================================


def run_heavy_compute(task_id: str, file_paths: dict, params: dict, work_dir: str, log_queue: asyncio.Queue,
                      loop: asyncio.AbstractEventLoop):
    # 记录状态
    compute_status = "SUCCESS"
    zip_path = os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")
    output_gro_path = os.path.join(work_dir, "output.gro")
    output_top_path = os.path.join(work_dir, "output.top")
    web_logger = setup_task_logger(task_id, work_dir, log_queue, loop)

    with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
        try:
            web_logger.info(f"Molecular File: {file_paths.get('mol_file_path')}")

            obmol, rdmol, coordinates, res_names, res_ids, box_tensor = molecule_reader(file_paths.get('mol_file_path'))

            # 修复了你原代码中 f-string 双引号嵌套的语法隐患
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

            # 成功打包：此时 debug_log_path 还在 with 作用域内，文件绝对存在且正在写入
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(output_gro_path, arcname="output.gro")
                zipf.write(output_top_path, arcname="output.top")
                zipf.write(debug_log_path, arcname="debug.log")

        except Exception as e:
            compute_status = "ERROR"
            web_logger.error(f"Error: {str(e)}")
            web_logger.debug(traceback.format_exc())

            # 失败打包：把记录了详细报错和底层 debug 轨迹日志打包
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(debug_log_path, arcname="debug_error.log")

    # 【核心改动 2】：必须在 with 块结束后（即 Handler 彻底 close 释放文件锁之后）再向前端发信号和删目录
    # 这样可以 100% 避免在 Windows/Linux 异步线程中由于文件被占用引发的 PermissionError
    loop.call_soon_threadsafe(log_queue.put_nowait, f"[[DONE_{compute_status}]]")
    shutil.rmtree(work_dir, ignore_errors=True)


# ==========================================
# 3. FastAPI 路由接口
# ==========================================

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """主页接口：提供复古风 Web UI"""
    if not os.path.exists("templates/index.html"):
        return "<h1>Error: index.html not found in current directory!</h1>"
    return FileResponse("templates/index.html")


@app.post("/api/upload_and_run")
async def upload_and_run(files: List[UploadFile] = File(...), params_json: str = Form(...)):
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    work_dir = os.path.join(WORKSPACE_BASE, task_id)
    os.makedirs(work_dir, exist_ok=True)

    # 构建传给计算逻辑的文件路径字典
    file_paths = {
        "mol_file_path": None,
        "index_file_path": None
    }

    for file in files:
        filepath = os.path.join(work_dir, file.filename)
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 根据后缀名自动分拣
        if file.filename.lower().endswith(('.pdb', '.sdf')):
            file_paths["mol_file_path"] = filepath
        elif file.filename.lower().endswith('.idx'):
            file_paths["index_file_path"] = filepath

    params = json.loads(params_json)

    # 如果没有找到分子文件，直接在 API 层拦截（虽然前端也会拦）
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
    """前端 SSE 实时连接此接口获取 INFO 日志"""
    queue = task_log_queues.get(task_id)
    if not queue:
        return {"error": "任务不存在或已清理"}

    async def event_generator():
        try:
            while True:
                # 异步等待新日志
                msg = await queue.get()
                if msg == "[[DONE]]":
                    yield {"data": "[[DONE]]"}
                    break
                yield {"data": msg}
        finally:
            # 清理队列内存
            task_log_queues.pop(task_id, None)

    return EventSourceResponse(event_generator())


@app.get("/api/download/{task_id}")
async def download_result(task_id: str):
    """下载最终的 ZIP 包"""
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

    # 直接在内网 8000 端口起飞
    uvicorn.run(app, host="0.0.0.0", port=8000)