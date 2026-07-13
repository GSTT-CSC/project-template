"""
nnU-Net v2 is driven through CLI entry points (``nnUNetv2_train`` etc.).
This module wraps those subprocess calls and streamlines mlflow logging.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import mlflow
import torch


logger = logging.getLogger(__name__)

artifacts_static_list = {
    "inference_instructions.txt",
    "inference_information.json",
    "dataset.json",
    "dataset_fingerprint.json",
    "plans.json",
    "postprocessing.json",
    "postprocessing.pkl",
    "debug.json",
    "progress.png",
    "checkpoint_best.pth",
    "checkpoint_final.pth",
    "summary.json",
    "config_log.txt",
    "data_manifest.csv",
}

artifacts_dynamic_list = {
    "training_log",
    ".nii.gz",
}


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


def _tee_stream(pipe, log_file):
    """Forward each line of a subprocess pipe to both the console and ``log_file``."""
    for line in pipe:
        sys.stdout.write(line)
        sys.stdout.flush()
        log_file.write(line)
        log_file.flush()


def run_command(
    cmd,
    artifact_dir,
    fold=None,
    configuration=None,
    poll_seconds: float = 120,
    log_metrics: bool = False,
):
    """
    Run nnU-Net CLI command ``cmd``, send result to console and subprocess.log (which ends in mlflow).

    When ``log_metrics`` is True, ``checkpoint_latest.pth`` in
    ``artifact_dir`` is polled every ``poll_seconds`` and any newly available epoch
    metrics are logged to MLflow.
    """

    os.makedirs(artifact_dir, exist_ok=True)
    subprocess_log_path = os.path.join(artifact_dir, "subprocess.log")
    ckpt_latest = Path(artifact_dir) / "checkpoint_latest.pth"
    ckpt_final = Path(artifact_dir) / "checkpoint_final.pth"
    last_mtime = None
    last_epoch = -1
    missing_checkpoint_warned = False

    with open(subprocess_log_path, "w", encoding="utf-8") as subprocess_log_file:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, # output goes to memory for now
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # subprocess.PIPE -> process.stdout -> _tee_stream -> console + subprocess.log
        tee_thread = threading.Thread(target=_tee_stream, args=(process.stdout, subprocess_log_file), daemon=True)
        tee_thread.start()

        if log_metrics:
            while process.poll() is None:
                if ckpt_latest.exists():
                    missing_checkpoint_warned = False
                    mtime = ckpt_latest.stat().st_mtime
                    if last_mtime is None or mtime > last_mtime:
                        last_mtime = mtime
                        last_epoch = _log_new_metrics(
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
                    logger.warning(
                        f"Checkpoint not found at expected location: {ckpt_latest}."
                    )
                    missing_checkpoint_warned = True
                time.sleep(poll_seconds)

        return_code = process.wait()
        tee_thread.join()

    if return_code != 0:
        logger.error(f"Command failed: {cmd} (exit {return_code})")
        _log_failure_artifacts(
            artifact_dir=artifact_dir,
            subprocess_log_path=subprocess_log_path,
        )
        raise subprocess.CalledProcessError(return_code, cmd)

    if log_metrics and ckpt_latest.exists():
        _log_new_metrics(
            checkpoint_path=str(ckpt_final),
            fold=fold,
            last_epoch=last_epoch,
            configuration=configuration,
        )


RUN_OVERVIEW_DESCRIPTION = """\
# nnU-Net v2 logging info

## Per-epoch metrics (logged as `fold_<f>_<config>_<name>`)
- **mean_fg_dice** — mean pseudo-Dice (estimated on patches) over the foreground labels on the validation split, per epoch.
- **ema_fg_dice** — exponential moving average of `mean_fg_dice` (smoothed).

## Artifacts
- **fold_<f>/** — validation metrics per fold.
- **crossval_results/** — validation metrics averaged over all folds, plus per-structure
  Dice / size figures (`cross_validation_dice.png`, `cross_validation_structure_size.png`).
- **test_set/labelsTs_predicted/** — model predictions on the test set.
- **test_set/labelsTs_predicted_pp/** — the same test predictions after nnU-Net's selected
  post-processing.
- **logs/run.log** — terminal log for the run
"""


def log_run_overview_description(description: str = RUN_OVERVIEW_DESCRIPTION):
    """Set the MLflow run Description (Overview tab) via the ``mlflow.note.content`` tag.

    The Description box renders markdown, so this documents — inside the run itself — what
    the logged metrics and artifact folders mean.
    """

    mlflow.set_tag("mlflow.note.content", description)


def log_nnunet_artifacts(path_to_walk: str):
    """
    Log selected nnU-Net v2 artifacts to MLflow.

    Any files not listed in artifacts_static_list and artifacts_dynamic_list are ignored, 
    such as large .npz files and more.

    Inputs:
        path_to_walk - folder to begin walking from, e.g. DatasetXXX folder in nnUNetV2 directory structure
       
    """

    for root, _, files in os.walk(path_to_walk, topdown=True):
        artifact_path = _resolve_artifact_path(root)
        for file_name in files:
            if not _should_log_artifact(file_name):
                continue

            full_file_path = os.path.join(root, file_name)
            if not os.path.isfile(full_file_path):
                logger.warning(f"Skipping missing nnUNet artifact: {full_file_path}")
                continue

            if artifact_path is None:
                logger.info(f"Logging nnUNet artifact: {file_name}")
                mlflow.log_artifact(full_file_path)
            else:
                logger.info(f"Logging nnUNet artifact to {artifact_path}: {file_name}")
                mlflow.log_artifact(full_file_path, artifact_path=artifact_path)


def log_validation_plots(crossval_dir: str):
    """Log to mlflow some figures displaying metrics for the cross-validation data.
    cross-validation data -> average over all K folds of each validation set.

    Reads ``<crossval_dir>/postprocessed/summary.json`` (metrics after nnU-Net's selected
    post-processing) and ``<crossval_dir>/dataset.json`` for structure names, and logs figs
    under the ``crossval_results`` artifact folder.
    """

    try:
        summary_path = os.path.join(crossval_dir, "postprocessed", "summary.json")
        if not os.path.isfile(summary_path):
            logger.warning(f"skipping validation plots; {summary_path} not found.")
            return

        dataset_json_path = os.path.join(crossval_dir, "dataset.json")
        if not os.path.isfile(dataset_json_path):
            dataset_json_path = None  # fall back to Label NN axis labels

        from src.nnunet import plots # only if valid

        figure_paths = plots.plot_validation_summary(
            summary_path,
            output_dir=crossval_dir,
            dataset_json_path=dataset_json_path,
        )
        for figure_path in figure_paths:
            mlflow.log_artifact(figure_path, artifact_path="crossval_results")
    
    except Exception:
        logger.exception("Failed to generate/log validation plots.")


def _log_failure_artifacts(artifact_dir, subprocess_log_path=None):
    try:
        if subprocess_log_path and os.path.isfile(subprocess_log_path):
            mlflow.log_artifact(subprocess_log_path, artifact_path="error_logs")
        training_log_path = locate_file(artifact_dir, "training_log")
        if training_log_path and os.path.isfile(training_log_path):
            mlflow.log_artifact(training_log_path, artifact_path="error_logs")
    except Exception:
        logger.exception("Failed to log failure artifact to MLflow.")


def _log_new_metrics(checkpoint_path, fold, last_epoch, configuration=None):
    state = torch.load(checkpoint_path, map_location="cpu")
    logger_state = state["logging"]
    keys = ["train_losses", "val_losses", "mean_fg_dice", "ema_fg_dice", "lrs"]
    n_epochs = max(len(logger_state[k]) for k in keys)
    metric_prefix = f"fold_{fold}_"
    if configuration:
        metric_prefix += f"{configuration}_"

    for epoch in range(last_epoch + 1, n_epochs):
        for k in keys:
            try:
                value = float(logger_state[k][epoch])
                mlflow.log_metric(f"{metric_prefix}{k}", value, step=epoch)
            except IndexError:
                logger.warning(
                    f"Expected metric {k} not found for epoch {epoch}"
                    f" in checkpoint: {checkpoint_path}."
                    f" Skipping logging of this metric for this epoch."
                )

    return n_epochs - 1


def _should_log_artifact(file_name: str) -> bool:

    is_static_file = file_name in artifacts_static_list
    is_dynamic_file = any(pattern in file_name for pattern in artifacts_dynamic_list)
    
    return (is_static_file or is_dynamic_file)


def _resolve_artifact_path(root: str):
    root_parts = os.path.normpath(root).split(os.sep)

    fold_dir = next((part for part in root_parts if part.startswith("fold_")), None)
    if fold_dir is not None:
        if "validation" in root_parts:
            return os.path.join(fold_dir, "validation")
        return fold_dir

    if any("crossval_results" in part for part in root_parts):
        return "crossval_results"

    # Test-set predictions (labelsTs_predicted / labelsTs_predicted_pp) grouped under a
    # single test_set/ folder in MLflow (on-disk nnU-Net names are left unchanged).
    labels_ts_dir = next((part for part in root_parts if part.startswith("labelsTs")), None)
    if labels_ts_dir is not None:
        return os.path.join("test_set", labels_ts_dir)

    return None
