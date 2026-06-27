import json
import os
import shutil
import uuid
from typing import List

import redis.asyncio as aioredis
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

app = FastAPI(title="P2P FF parameterizer server")

WORKSPACE_BASE = "./workspaces"
os.makedirs(WORKSPACE_BASE, exist_ok=True)

redis_client = aioredis.Redis(host='localhost', port=6379, decode_responses=True)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not os.path.exists("templates/index.html"):
        return "<h1>Error: templates/index.html not found in current directory!</h1>"
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

    task_payload = {
        "task_id": task_id,
        "file_paths": file_paths,
        "params": params,
        "work_dir": work_dir
    }
    await redis_client.rpush('md_task_queue', json.dumps(task_payload))
    return {"status": "success", "task_id": task_id}


@app.get("/api/stream_logs/{task_id}")
async def stream_logs(task_id: str):
    async def event_generator():
        pubsub = redis_client.pubsub()
        channel_name = f"log_channel_{task_id}"
        await pubsub.subscribe(channel_name)

        try:
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    data = message['data']
                    yield {"data": data}

                    if data.startswith("[[DONE_"):
                        break
        finally:
            await pubsub.unsubscribe(channel_name)

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
