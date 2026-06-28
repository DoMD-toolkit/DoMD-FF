import os
import re
import shutil
import time
from pathlib import Path

import redis

WORKSPACE_BASE = Path(os.getenv("WORKSPACE_BASE", "./workspaces"))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

RETENTION_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
GRACE_SECONDS = int(os.getenv("CLEANUP_GRACE_SECONDS", "3600"))

DRY_RUN = "1" #os.getenv("DRY_RUN", "0") == "1"

TASK_DIR_RE = re.compile(r"^task_[a-f0-9]{8}([a-f0-9]{24})?$")
TASK_ZIP_RE = re.compile(r"^(task_[a-f0-9]{8}([a-f0-9]{24})?)_result\.zip$")


def now_ts() -> float:
    return time.time()


def is_old_enough(path: Path) -> bool:
    try:
        age = now_ts() - path.stat().st_mtime
    except FileNotFoundError:
        return False

    return age > RETENTION_SECONDS + GRACE_SECONDS


def redis_task_keys(task_id: str) -> list[str]:
    return [
        f"task_meta_{task_id}",
        f"task_status_{task_id}",
        f"task_claim_{task_id}",
    ]


def redis_has_task(r: redis.Redis, task_id: str) -> bool:
    return r.exists(*redis_task_keys(task_id)) > 0


def remove_path(path: Path) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] Would remove: {path}")
        return

    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()

    print(f"[CLEANUP] Removed: {path}")


def cleanup() -> None:
    if not WORKSPACE_BASE.exists():
        print(f"[INFO] Workspace base does not exist: {WORKSPACE_BASE}")
        return

    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
    )

    try:
        r.ping()
    except redis.RedisError as exc:
        print(f"[WARN] Redis is unavailable. Cleanup skipped. Details: {exc}")
        return

    candidates: dict[str, list[Path]] = {}
    for path in WORKSPACE_BASE.iterdir():
        name = path.name

        if path.is_dir() and TASK_DIR_RE.fullmatch(name):
            task_id = name
            candidates.setdefault(task_id, []).append(path)
            continue

        if path.is_file():
            match = TASK_ZIP_RE.fullmatch(name)
            if match:
                task_id = match.group(1)
                candidates.setdefault(task_id, []).append(path)

    for task_id, paths in sorted(candidates.items()):
        try:
            if redis_has_task(r, task_id):
                continue
        except redis.RedisError as exc:
            print(f"[WARN] Redis check failed for {task_id}. Skipping. Details: {exc}")
            continue

        old_paths = [p for p in paths if is_old_enough(p)]
        if not old_paths:
            continue

        for path in old_paths:
            remove_path(path)


if __name__ == "__main__":
    cleanup()
