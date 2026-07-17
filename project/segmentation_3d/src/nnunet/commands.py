"""
List of nnU-Net v2 system commands and config translation functions.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


# ======================================================================================
# Validated config objects
# ======================================================================================
@dataclass(frozen=True)
class NNUNetModelSpec:
    """Validated description of the nnU-Net model/training run."""

    dataset_id: str
    dataset_name: str
    folds: List[int]
    trainer_name: str = "nnUNetTrainerNoMirroring"
    plans_identifier: str = "nnUNetPlans"
    configuration: str = "3d_fullres"
    planner: str = "ExperimentPlanner"
    gpu_memory_target: Optional[int] = None
    use_npz: bool = False

    def __post_init__(self):
        # Variable validation only.

        if not str(self.dataset_id) or not str(self.dataset_id).isdigit():
            raise ValueError(f"NNUNetModelSpec.dataset_id must be a non-empty numeric string, got: {self.dataset_id}")

        # These fields must each be a single, non-empty string.
        for field_name in ("dataset_name", "trainer_name", "plans_identifier", "planner"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"NNUNetModelSpec.{field_name} must be a single non-empty string, got: {value}")

        # configuration must also be one config. List means several configurations or ensembling,
        # not currently supported; error should refelect this.
        if not isinstance(self.configuration, str):
            raise ValueError(
                "NNUNetModelSpec.configuration must be a single configuration string; multiple "
                "configurations / ensembling are not supported yet (requires dropping "
                "--disable_ensembling and using nnUNetv2_ensemble)."
            )
        
        if not self.configuration:
            raise ValueError("NNUNetModelSpec.configuration must be a non-empty string.")

        # 3d_cascade_fullres is also unsupported at the moment.
        if self.configuration == "3d_cascade_fullres":
            raise ValueError(
                "NNUNetModelSpec.configuration '3d_cascade_fullres' is not supported yet: cascade "
                "test inference requires a two-stage predict that this template does not implement. "
                "Use 3d_fullres, 3d_lowres or 2d."
            )

        is_folds_non_empty_list_of_ints = (
            isinstance(self.folds, list)
            and bool(self.folds)
            and all(isinstance(fold, int) and not isinstance(fold, bool) for fold in self.folds)
        )
        if not is_folds_non_empty_list_of_ints:
            raise ValueError(f"NNUNetModelSpec.folds must be a non-empty list of integers, got:{self.folds}.")
        
        if self.gpu_memory_target is not None and self.gpu_memory_target <= 0:
            raise ValueError(f"NNUNetModelSpec.gpu_memory_target must be positive when provided, got: {self.gpu_memory_target}.")

    @property
    def preprocess_configurations(self):
        """
        Configurations to preprocess (cascade will use fullres and lowres).
        """

        if self.configuration == "3d_cascade_fullres":
            return ["3d_fullres", "3d_lowres"]
        return [self.configuration]

    @property
    def training_configurations(self):
        """
        Configurations to train, in order (cascade will use lowres then cascade).
        """

        if self.configuration == "3d_cascade_fullres":
            return ["3d_lowres", "3d_cascade_fullres"]
        return [self.configuration]


# ======================================================================================
# Build config objects from config
# ======================================================================================

def nnunet_npz_enabled(config):
    """
    Whether to export softmax .npz (needed for find_best_configuration / postprocessing).
    """

    value = config["nnunet"]["NNUNET_NPZ"].strip().lower()
    if value == "true" or value == "--npz":
        return True
    if value == "false":
        return False
    raise ValueError(f"NNUNET_NPZ must be set to True or False, got: {value}.")


def parse_nnunet_folds(config):
    """Parse nnunet.NNUNET_FOLD as JSON"""

    raw_folds = config["nnunet"]["NNUNET_FOLD"].strip()
    try:
        return json.loads(raw_folds)
    except json.JSONDecodeError as exc:
        raise ValueError(f"nnunet.NNUNET_FOLD must be a non-empty JSON list of integers, e.g. [0], got: {raw_folds}.") from exc


def parse_gpu_memory_target(config):
    """Parse optional nnunet.NNUNET_GPU_MEMORY_TARGET (GB). int>0 or blank """
    
    raw = config["nnunet"]["NNUNET_GPU_MEMORY_TARGET"].strip()
    if not raw:
        return None
    if raw.isdigit():
        return int(raw)
    raise ValueError(f"nnunet.NNUNET_GPU_MEMORY_TARGET must be a positive integer (GB), got: {raw}.")


def build_nnunet_model_spec(config, dm):
    """Build NNUNetModelSpec dataclass from cleaned up config + the prepared DataModule."""
    
    return NNUNetModelSpec(
        dataset_id="".join(c for c in dm.dataset_dir_name if c.isdigit()),
        dataset_name=dm.dataset_dir_name.strip(),
        trainer_name=config["nnunet"]["NNUNET_TRAINER"].strip(),
        plans_identifier=config["nnunet"]["NNUNET_PLANS_IDENTIFIER"].strip(),
        configuration=config["nnunet"]["NNUNET_UNET_CONFIGURATION"].strip(),
        folds=parse_nnunet_folds(config),
        planner=config["nnunet"]["NNUNET_PLANNER"].strip(),
        gpu_memory_target=parse_gpu_memory_target(config),
        use_npz=nnunet_npz_enabled(config),
    )


# ======================================================================================
# Pure nnU-Net CLI command builders
# ======================================================================================

def multi_gpu_args(device):
    """
    Return ['-num_gpus', N] when more than one CUDA device is visible, else [].
    """

    if device != "cuda":
        return []
    n_visible = len(
        [gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    )
    return ["-num_gpus", str(n_visible)] if n_visible > 1 else []


def fold_artifact_dir(results_dir, spec, fold, configuration):
    """Directory nnU-Net writes a fold's checkpoints/logs to (used for live metric logging)."""

    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{configuration}",
        f"fold_{fold}",
    )


def crossval_results_dir(results_dir, spec):
    """Grab dir that find_best_configuration writes the cross-validation results to.

    Current understanding looks like ``crossval_results_folds_0_1_2_3_4``.
    """

    folds_str = "_".join(str(fold) for fold in spec.folds)
    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{spec.configuration}",
        f"crossval_results_folds_{folds_str}",
    )


def plan_and_preprocess_command(spec):
    """ Run CLI command as defined -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py"""
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", spec.dataset_id,
        "-pl", spec.planner,
        "-c", *spec.preprocess_configurations,
        "--verify_dataset_integrity",
    ]
    if spec.gpu_memory_target is not None:
        cmd += ["-gpu_memory_target", str(spec.gpu_memory_target)]
    return cmd


def train_command(spec, fold, configuration, device):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/run/run_training.py.
    
    Training variants (spec.trainer_name) loosely defined in
    https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants.
    """
    
    cmd = [
        "nnUNetv2_train",
        spec.dataset_id,
        configuration,
        str(fold),
        "-tr", spec.trainer_name,
        "-p", spec.plans_identifier,
        "-device", device,
    ]
    if spec.use_npz:
        cmd += ["--npz"]
    return cmd + multi_gpu_args(device)


def find_best_configuration_command(spec):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/evaluation/find_best_configuration.py
    """

    # --disable_ensembling below is adequate on a single config. Extra guard here
    # such that if multiple configs are supported in the future, --disable_ensembling
    # can be removed. 
    if not isinstance(spec.configuration, str):
        raise ValueError("remove --disable_ensembling for multiple configurations.")
    return [
        "nnUNetv2_find_best_configuration",
        spec.dataset_id,
        "-p", spec.plans_identifier,
        "-c", spec.configuration,
        "-tr", spec.trainer_name,
        "--disable_ensembling",
        "-f", *[str(fold) for fold in spec.folds],
    ]


def predict_command(spec, input_dir, output_dir, device):
    """
    Runs CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
    """

    # -c uses the single trained configuration. Extra guard here such that if multiple
    # configs are supported in the future and the optimal inference configuration is
    # determined by find_best_configuration, the final configuration should be read from
    # find_best_configuration's inference_instructions.txt instead of the training list.
    if not isinstance(spec.configuration, str):
        raise ValueError("read -c from inference_instructions.txt for multiple configurations.")

    return [
        "nnUNetv2_predict",
        "-d", spec.dataset_id,
        "-i", input_dir,
        "-o", output_dir,
        "-c", spec.configuration,
        "-tr", spec.trainer_name,
        "-p", spec.plans_identifier,
        "-device", device,
        "-f", *[str(fold) for fold in spec.folds],
    ]


def apply_postprocessing_command(spec, input_dir, output_dir, postprocessing_file):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/postprocessing/remove_connected_components.py
    """

    # if multiple configs are supported in the future, -plans_json and -dataset_json must be
    # passed explicitly (from an ensemble member).
    if not isinstance(spec.configuration, str):
        raise ValueError("pass -plans_json / -dataset_json for multiple configurations.")
    return [
        "nnUNetv2_apply_postprocessing",
        "-i", input_dir,
        "-o", output_dir,
        "-pp_pkl_file", postprocessing_file,
    ]


def evaluate_folder_command(gt_dir, pred_dir, dataset_json, plans_json):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/evaluation/evaluate_predictions.py

    Scores predicted segmentations against ground truth, writing summary.json into pred_dir.
    --chill: don't crash if a ground-truth case has no matching prediction.
    """

    return [
        "nnUNetv2_evaluate_folder",
        gt_dir,
        pred_dir,
        "-djfile", dataset_json,
        "-pfile", plans_json,
        "--chill",
    ]
