"""
nnU-Net v2 is driven through CLI entry points (``nnUNetv2_train`` etc.).
This module runs those commands as subprocesses and forwards their output to our logger.
"""

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from src.nnunet.mlflow_logging import log_new_metrics


logger = logging.getLogger(__name__)


def locate_file(path_to_walk: str, file_to_locate: str) -> str:
    """
    Locate a file by (sub)name beneath ``path_to_walk`` and return its full path.

    Useful for dynamically created files whose directory is not known in advance
    (e.g. ``postprocessing.pkl``). Logs a warning if zero or multiple matches are
    found; in the multiple-match case the last match is returned.
    """

    file_ctr = 0
    located_file_path = None
    for (root, dirs, files) in os.walk(path_to_walk, topdown=True):
        for file in files:
            if file_to_locate in file:
                located_file_path = os.path.join(root, file)
                logger.info(
                    f"File located: {file_to_locate}. Path to located file: {located_file_path}"
                )
                file_ctr += 1

    if file_ctr == 0:
        logger.warning(f"File could not be located: {file_to_locate}")
    elif file_ctr > 1:
        logger.warning(
            f"More than one file located named: {file_to_locate}. Returning file: {located_file_path}"
        )

    return located_file_path


def forward_stream(pipe):
    """Forward each line of a subprocess to logger.*."""
    for line in pipe:
        logger.info(line.rstrip())


def run_command(
    cmd,
    artifact_dir,
    fold=None,
    configuration=None,
    poll_seconds: float = 120,
    log_metrics: bool = False,
):
    """
    Run nnU-Net CLI command ``cmd``, forwarding its output to our logger.

    When ``log_metrics`` is True, ``checkpoint_latest.pth`` in
    ``artifact_dir`` is polled every ``poll_seconds`` and any newly available epoch
    metrics are logged to MLflow.
    """

    os.makedirs(artifact_dir, exist_ok=True)
    ckpt_latest = Path(artifact_dir) / "checkpoint_latest.pth"
    ckpt_final = Path(artifact_dir) / "checkpoint_final.pth"
    last_mtime = None
    last_epoch = -1
    missing_checkpoint_warned = False

    # Start the nnunet process.
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, # output goes to memory not terminal (process.stdout)
        stderr=subprocess.STDOUT, # None; merged in stdout so both can be sent to logger
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}, # send line-by-line rather than chunks
    )

    # Logging workflow using a thread:
    # subprocess.PIPE -> process.stdout -> forward_stream -> logger -> console + run.log
    # start thread below reads process.stdout as it comes and executes forward_stream (sends each line to logger).
    forward_thread = threading.Thread(target= forward_stream, args=(process.stdout,), daemon=True)
    forward_thread.start()

    if log_metrics:
        while process.poll() is None:
            if ckpt_latest.exists():
                missing_checkpoint_warned = False
                mtime = ckpt_latest.stat().st_mtime
                if last_mtime is None or mtime > last_mtime:
                    last_mtime = mtime
                    last_epoch = log_new_metrics(
                        checkpoint_path=str(ckpt_latest),
                        fold=fold,
                        last_epoch=last_epoch,
                        configuration=configuration,
                    )
                    logger.info(
                        f"Logged new metrics to MLFlow from checkpoint:"
                        f" {ckpt_latest} at epoch: {last_epoch}."
                    )
            elif not missing_checkpoint_warned:
                logger.info(
                    f"No periodic checkpoint yet at {ckpt_latest}; metrics logged when training ends."
                )
                missing_checkpoint_warned = True
            time.sleep(poll_seconds)

    return_code = process.wait() # wait for nnU-Net to exit
    forward_thread.join() # wait for thread to finish emptying process.stdout and move on

    if return_code != 0:
        # No special failure artifacts needed: the whole run (including this command's output)
        # is in run.log, which train.py sends to MLflow whether or not the run succeeded.
        logger.error(f"Command failed: {cmd} (exit {return_code})")
        raise subprocess.CalledProcessError(return_code, cmd)

    if log_metrics and ckpt_final.exists():
        log_new_metrics(
            checkpoint_path=str(ckpt_final),
            fold=fold,
            last_epoch=last_epoch,
            configuration=configuration,
        )
