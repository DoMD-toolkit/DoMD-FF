import copy
import json
import logging
import os
import traceback
import zipfile

import redis
from rdkit.Chem import rdMolHash

from ForceField import FF
from misc.io.gmx import write_gro_file, write_top_file, write_list_itp_files
from misc.logger import task_file_log_scope, mol_file_log_scope
from misc.parser import molecule_reader, molecule_reader_list

WORKSPACE_BASE = "./workspaces"
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)


def setup_worker_logger(task_id: str):
    logger = logging.getLogger(f"task_{task_id}")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    class RedisPubSubHandler(logging.Handler):
        def __init__(self, r_client, channel):
            super().__init__()
            self.r_client = r_client
            self.channel = channel

        def emit(self, record):
            if record.levelno >= logging.INFO:
                msg = self.format(record)
                self.r_client.publish(self.channel, msg)

    channel_name = f"log_channel_{task_id}"
    pubsub_handler = RedisPubSubHandler(redis_client, channel_name)
    pubsub_handler.setLevel(logging.INFO)
    pubsub_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    logger.addHandler(pubsub_handler)
    return logger


def run_heavy_compute(task_id: str, file_paths: dict, params: dict, work_dir: str):
    compute_status = "SUCCESS"
    zip_path = os.path.join(WORKSPACE_BASE, f"{task_id}_result.zip")

    web_logger = setup_worker_logger(task_id)

    with task_file_log_scope(task_name=task_id, log_dir=work_dir) as debug_log_path:
        web_logger.info(f"Starting TASK {task_id} on compute node...")
        web_logger.info(f"Parameters: useGMX: {params.get('useGMX')} useBOSS: {params.get('useBOSS')} "
                        f"useML: {params.get('useML')} overwrite: {params.get('overwrite')} "
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
                    raise ValueError("Force field parameterization failed, please check the log files.")

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
                itp_fns, forcefields, mol_names, mol_log_paths = [], [], [], []

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
                                             overwrite=params.get("overwrite"),
                                             charge_factor=params.get("charge_factor"))

                            if not forcefield.success:
                                web_logger.error(f"Force field for molecule {idx:06d} parametrization failed.")
                                _cache[wl_hash] = {'notfound': True, 'idx': idx}
                            else:
                                itp_fns.append(os.path.join(work_dir, f"{idx:06d}.itp"))
                                mol_names.append(f"{idx:06d}")
                                forcefields.append(forcefield)
                                web_logger.info(f"Molecule {idx:06d} parametrization success.")
                                num_success += 1
                                _cache[wl_hash] = {
                                    'params': copy.deepcopy(forcefield.params),
                                    'charges': copy.deepcopy(forcefield.charges),
                                    'idx': idx, 'notfound': False
                                }

                if num_success == len(mol_list):
                    compute_status = 'SUCCESS'
                elif num_success == 0:
                    compute_status = 'ERROR'
                else:
                    compute_status = "PARTIAL"

                if num_success > 0:
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
                    if os.path.exists(atomtypes_path):
                        zipf.write(atomtypes_path, arcname=os.path.basename(atomtypes_path))
                web_logger.info("Zip package created successfully.")

        except Exception as e:
            compute_status = "ERROR"
            web_logger.error(f"Error: {str(e)}")
            web_logger.error(traceback.format_exc())

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(debug_log_path, arcname="debug_error.log")

    redis_client.publish(f"log_channel_{task_id}", f"[[DONE_{compute_status}]]")


def main():
    print("[SYSTEM] Worker Node Online. Waiting for tasks from Redis...")
    while True:
        try:
            queue_name, message = redis_client.blpop('md_task_queue', 0)
            task_data = json.loads(message)

            task_id = task_data['task_id']
            file_paths = task_data['file_paths']
            params = task_data['params']
            work_dir = task_data['work_dir']

            print(f"[SYSTEM] Task {task_id} received. Executing...")
            run_heavy_compute(task_id, file_paths, params, work_dir)
            print(f"[SYSTEM] Task {task_id} completed.")

        except Exception as e:
            print(f"[FATAL] Worker loop error: {e}")


if __name__ == "__main__":
    main()
