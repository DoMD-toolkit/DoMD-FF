import copy
import json
import logging
import os
import time
import zipfile

import redis
from redis.exceptions import RedisError, TimeoutError, ConnectionError
from rdkit.Chem import rdMolHash

from ForceField import FF
from misc.io.gmx import write_gro_file, write_top_file, write_list_itp_files
from misc.logger import task_file_log_scope, mol_file_log_scope
from misc.parser import molecule_reader, molecule_reader_list

WORKSPACE_BASE = os.getenv("WORKSPACE_BASE", "./workspaces")
TASK_QUEUE = os.getenv("TASK_QUEUE", "md_task_queue")
TASK_TTL_SECONDS = int(os.getenv("TASK_TTL_SECONDS", "86400"))
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

TERMINAL_STATES = {"SUCCESS", "PARTIAL", "ERROR"}


def create_redis_client() -> redis.Redis:
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
        socket_connect_timeout=5,
    )


redis_client = create_redis_client()


def now_ts() -> int:
    return int(time.time())


def task_meta_key(task_id: str) -> str:
    return f"task_meta_{task_id}"


def task_status_key(task_id: str) -> str:
    # This key stores terminal states only, for backward compatibility with SSE.
    return f"task_status_{task_id}"


def result_zip_path(task_id: str) -> str:
    return os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")


def local_status_path(work_dir: str) -> str:
    return os.path.join(work_dir, "task_status.json")


def write_local_status(work_dir: str, meta: dict) -> None:
    if not work_dir:
        return

    try:
        os.makedirs(work_dir, exist_ok=True)
        status_path = local_status_path(work_dir)
        tmp_path = f"{status_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=True, indent=2)
        os.replace(tmp_path, status_path)
    except OSError as exc:
        print(f"[WARN] Could not write local task status file: {exc}", flush=True)


def update_task_meta(task_id: str, state: str, message: str, work_dir: str, terminal: bool = False) -> dict:
    created_at = now_ts()
    meta = {}

    try:
        existing_raw = redis_client.get(task_meta_key(task_id))
        if existing_raw:
            meta = json.loads(existing_raw)
            created_at = int(meta.get("created_at", created_at))
    except (RedisError, json.JSONDecodeError, ValueError) as exc:
        print(f"[WARN] Could not read existing Redis task metadata: {exc}", flush=True)

    has_result = os.path.exists(result_zip_path(task_id))
    meta.update({
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
    })

    write_local_status(work_dir, meta)

    try:
        redis_client.set(task_meta_key(task_id), json.dumps(meta), ex=TASK_TTL_SECONDS)
    except RedisError as exc:
        print(f"[WARN] Could not update Redis task metadata: {exc}", flush=True)

    return meta


def publish_best_effort(channel: str, message: str) -> None:
    try:
        redis_client.publish(channel, message)
    except RedisError as exc:
        print(f"[WARN] Redis publish failed; frontend live log may miss this message: {exc}", flush=True)


def finalize_task(task_id: str, state: str, work_dir: str, message: str) -> None:
    update_task_meta(task_id, state, message, work_dir, terminal=True)

    try:
        redis_client.set(task_status_key(task_id), state, ex=TASK_TTL_SECONDS)
    except RedisError as exc:
        print(f"[WARN] Could not write terminal Redis task status: {exc}", flush=True)

    publish_best_effort(f"log_channel_{task_id}", f"[[DONE_{state}]]")


def setup_worker_logger(task_id: str):
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    class RedisPubSubHandler(logging.Handler):
        def __init__(self, channel):
            super().__init__()
            self.channel = channel
            self.next_warning_at = 0

        def emit(self, record):
            if record.levelno < logging.INFO:
                return

            msg = self.format(record)
            try:
                redis_client.publish(self.channel, msg)
            except RedisError as exc:
                # Logging must never break the scientific computation.
                current_ts = time.time()
                if current_ts >= self.next_warning_at:
                    print(
                        f"[WARN] Redis publish failed; frontend live log may miss messages: {exc}",
                        flush=True,
                    )
                    self.next_warning_at = current_ts + 30

    channel_name = f"log_channel_{task_id}"
    pubsub_handler = RedisPubSubHandler(channel_name)
    pubsub_handler.setLevel(logging.INFO)
    pubsub_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(pubsub_handler)
    return logger


def close_worker_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def run_heavy_compute(task_id: str, file_paths: dict, params: dict, work_dir: str):
    compute_status = "SUCCESS"
    zip_path = result_zip_path(task_id)

    update_task_meta(
        task_id,
        "RUNNING",
        "Task is running on the compute node.",
        work_dir,
        terminal=False,
    )

    web_logger = setup_worker_logger(task_id)

    try:
        with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
            web_logger.info(f"Starting task {task_id} on compute node.")
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
        )
        close_worker_logger(web_logger)


def reconnect_redis(delay_seconds: int) -> None:
    global redis_client

    try:
        redis_client.close()
    except Exception:
        pass

    print(f"[SYSTEM] Reconnecting to Redis after {delay_seconds}s.", flush=True)
    time.sleep(delay_seconds)
    redis_client = create_redis_client()


def main():
    print("[SYSTEM] Worker node online. Waiting for tasks from Redis.", flush=True)
    retry_delay = 1

    while True:
        try:
            # BLPOP with timeout=0 blocks until a task is available. This is not polling.
            result = redis_client.blpop(TASK_QUEUE, timeout=0)
            retry_delay = 1

            if result is None:
                continue

            queue_name, message = result
            task_data = json.loads(message)
            task_id = task_data["task_id"]
            created_at = int(task_data.get("created_at", now_ts()))

            if now_ts() - created_at > TASK_TTL_SECONDS:
                print(f"[WARN] Dropping expired task {task_id}; queue payload is older than retention window.", flush=True)
                continue

            print(f"[SYSTEM] Task {task_id} received from {queue_name}. Executing.", flush=True)
            run_heavy_compute(task_id, task_data["file_paths"], task_data["params"], task_data["work_dir"])
            print(f"[SYSTEM] Task {task_id} completed.", flush=True)

        except (TimeoutError, ConnectionError) as exc:
            print(f"[WARN] Redis connection failed. Reconnect will be attempted. Details: {exc}", flush=True)
            reconnect_redis(retry_delay)
            retry_delay = min(retry_delay * 2, 30)

        except RedisError as exc:
            print(f"[ERROR] Redis command failed. Reconnect will be attempted. Details: {exc}", flush=True)
            reconnect_redis(retry_delay)
            retry_delay = min(retry_delay * 2, 30)

        except Exception as exc:
            print(f"[ERROR] Worker task loop error: {exc}", flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
