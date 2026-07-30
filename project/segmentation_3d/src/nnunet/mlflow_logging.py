"""
Set of functions for logging nnU-Net v2 runs to MLflow. 
"""

import logging
import os

import mlflow
import torch


logger = logging.getLogger(__name__)

artifact_filenames = {
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

artifact_filename_patterns = {
    "training_log",
    ".nii.gz",
}


default_run_overview_description = """\
# nnU-Net v2 logging info

## Per-epoch metrics (logged as `fold_<f>_<config>_<name>`)
- **mean_fg_dice** — mean pseudo-Dice (estimated on patches) over the foreground labels on the validation split, per epoch.
- **ema_fg_dice** — exponential moving average of `mean_fg_dice` (smoothed).

## Artifacts
- **fold_<f>/** — validation metrics per fold.
- **crossval_results/** — validation metrics averaged over all folds, plus per-structure
  Dice / size figures (`dice.png`, `structure_size.png`).
- **test_set/** — test-set Dice / size figures (`dice.png`, `structure_size.png`) scored
  against ground truth.
- **test_set/labelsTs_predicted/** — model predictions on the test set.
- **test_set/labelsTs_predicted_pp/** — post-processed test predictions + `summary.json`
  (metrics vs ground truth).
- **logs/run.log** — full terminal log for the run (logger.* + nnU-Net subprocess output).
"""


def log_run_overview_description(description: str = default_run_overview_description):
    """Set the MLflow run Description (Overview tab) via the ``mlflow.note.content`` tag.

    This is so users are not left wondering what the various metrics/artifacts mean in MLFlow.
    """

    mlflow.set_tag("mlflow.note.content", description)


def log_nnunet_artifacts(path_to_walk: str):
    """
    Log selected nnU-Net v2 artifacts to MLflow.

    Any files not listed in artifact_filenames and artifact_filename_patterns are ignored,
    such as large .npz files and more.

    The input is path_to_walk, a folder to begin walking from, e.g. DatasetXXX folder in nnUNetV2 directory structure
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


def log_metric_plots(summary_path, dataset_json_path, artifact_path):
    """Generate metric figures (Dice, structure size) from a summary.json and log them to mlflow.

    Shared by the cross-validation and test summaries. ``artifact_path`` sets the mlflow folder
    and doubles as the figure title label. Figures are written next to the summary.
    """

    try:
        if not os.path.isfile(summary_path):
            logger.warning(f"skipping {artifact_path} plots; {summary_path} not found.")
            return

        if not (dataset_json_path and os.path.isfile(dataset_json_path)):
            dataset_json_path = None  # fall back to Label NN axis labels

        from src.nnunet import plots # only if valid

        figure_paths = plots.plot_metric_figures(
            summary_path,
            output_dir=os.path.dirname(summary_path),
            dataset_json_path=dataset_json_path,
            title=artifact_path.replace("_", " "),
        )
        for figure_path in figure_paths:
            mlflow.log_artifact(figure_path, artifact_path=artifact_path)

    except Exception:
        logger.exception(f"Failed to generate/log {artifact_path} plots.")


def log_new_metrics(checkpoint_path, fold, last_epoch, configuration=None):
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

    is_exact_match = file_name in artifact_filenames
    is_pattern_match = any(pattern in file_name for pattern in artifact_filename_patterns)

    return (is_exact_match or is_pattern_match)


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
