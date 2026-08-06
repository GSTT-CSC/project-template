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
    "dice.png",
    "structure_size.png",
    "checkpoint_best.pth",
    "checkpoint_final.pth",
    "summary.json",
    "config_log.txt",
    "data_manifest.csv",
}

artifact_filename_patterns = {
    "training_log",
    ".nii.gz",
    ".dcm",  # test-set predictions as RTSTRUCT
}


default_run_overview_description = """\
# nnU-Net v2 logging info

## Per-epoch metrics (logged as `fold_<f>_<config>_<name>`)
- **mean_fg_dice** — mean pseudo-Dice (estimated on patches) over the foreground labels on the validation split, per epoch.
- **ema_fg_dice** — exponential moving average of `mean_fg_dice` (smoothed).

## Artifacts
They are sorted out using nnU-Net's own output folder structure. Just be aware that validation and test runs include
`postprocessed/` folders which should be used for final metrics.
"""


def log_run_overview_description(description: str = default_run_overview_description):
    """Set the MLflow run Description (Overview tab) via the ``mlflow.note.content`` tag.

    This is so users are not left wondering what the various metrics/artifacts mean in MLFlow.
    """

    mlflow.set_tag("mlflow.note.content", description)


def log_nnunet_artifacts(path_to_walk: str):
    """
    Log selected nnU-Net v2 artifacts to MLflow, mirroring their layout on disk.

    Any files not listed in artifact_filenames and artifact_filename_patterns are ignored,
    such as large .npz files and more.

    The input is path_to_walk, a folder to begin walking from, e.g. DatasetXXX folder in nnUNetV2 directory structure
    """

    for root, _, files in os.walk(path_to_walk, topdown=True):
        # None at the top level, so those files land at the root of the MLflow artifact tree.
        relative_dir = os.path.relpath(root, path_to_walk)
        artifact_path = None if relative_dir == os.curdir else relative_dir

        for file_name in files:
            if not _should_log_artifact(file_name):
                continue

            full_file_path = os.path.join(root, file_name)
            if not os.path.isfile(full_file_path):
                logger.warning(f"Skipping missing nnUNet artifact: {full_file_path}")
                continue

            logger.info(f"Logging nnUNet artifact: {os.path.join(relative_dir, file_name)}")
            mlflow.log_artifact(full_file_path, artifact_path=artifact_path)


def make_metric_plots(summary_path, dataset_json_path, title):
    """Generate metric figures (dice.png, structure_size.png) from a summary.json.

    Shared by the cross-validation and test summaries; ``title`` labels the figures. They are
    written next to the summary.
    """

    try:
        if not os.path.isfile(summary_path):
            logger.warning(f"skipping {title} plots; {summary_path} not found.")
            return

        if not (dataset_json_path and os.path.isfile(dataset_json_path)):
            dataset_json_path = None  # fall back to Label NN axis labels

        from src.nnunet import plots # only if valid

        plots.plot_metric_figures(
            summary_path,
            output_dir=os.path.dirname(summary_path),
            dataset_json_path=dataset_json_path,
            title=title,
        )

    except Exception:
        logger.exception(f"Failed to generate {title} plots.")


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
