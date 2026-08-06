# segmentation_3d

Project template for 3D image segmentation that pulls data from
XNAT, trains with [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet), and logs
in MLflow via `csc-mlops`.

The template roughly follows the conventions of the `classifier_2d` template but
swaps the Lightning training loop for nnU-Net's own engine, called via its
command-line entry points.

## Workflow

1. Upload each subject's image volume and label structures to XNAT (pre-converted to
   NIfTI).
2. Edit a `regions.json` (see `regions/example/regions.json`) to declare which image file
   and contour files to train on, plus the input channel name (`CT`, `MR`, ...).
3. Set up config file (see `config/config.cfg` as starting point) to see your XNAT
   project, MLflow server, and adjust nnU-Net settings. The config file is well commented
   and it should be straighforward to figure out what to modify.
4. Run the pipeline (via `csc-mlops`), for instance:
```shell
mlops run mlops/train.py -c <path_to_config> --include_path ../shared
```

## What a run creates on disk

Everything a run produces goes under one timestamped working folder
(`TMP_WORKING_DIR_<datetime>_<uuid>`, see `[system] TMP_WORKING_DIR`). Numbers in brackets
show which step wrote each thing:

| | step | what runs it |
|---|---|---|
| `[1]` | build the dataset from XNAT | `dm.setup()` (`src/datamodule.py`) |
| `[2]` | `nnUNetv2_plan_and_preprocess` | measure data, plan, preprocess |
| `[3]` | `nnUNetv2_train` | once per fold |
| `[4]` | `nnUNetv2_find_best_configuration` | postprocessing decision + crossval summary |
| `[5]` | `nnUNetv2_predict` | test set |
| `[6]` | `nnUNetv2_apply_postprocessing` | test set |
| `[7]` | `nnUNetv2_evaluate_folder` | score test set vs ground truth |
| `[8]` | `train.py` | figures, config log, manifest, MLflow upload |
| `[9]` | `train.py` | test predictions as DICOM RTSTRUCT |

```
TMP_WORKING_DIR_<datetime>_<uuid>/
│
├── nnUNet_raw/DatasetXXX_<model>/                                          [1]
│   ├── imagesTr/Case_NNN_0000.nii.gz          training images, NNN is the subject number
│   ├── labelsTr/Case_NNN.nii.gz               training masks (contours combined into one)
│   ├── imagesTs/Case_NNN_0000.nii.gz          test images
│   ├── labelsTs/Case_NNN.nii.gz               test ground truth (never trained on)
│   └── dataset.json                           channel names + label map
│
├── dicom_Ts/Case_NNN/                         test subjects only: the source CT DICOM  [1]
│                                              series, needed to write RTSTRUCTs
│
├── nnUNet_preprocessed/DatasetXXX_<model>/
│   ├── dataset_fingerprint.json                                           [2]
│   ├── <plans identifier>.json                patch size, batch size, architecture   [2]
│   ├── dataset.json                                                       [2]
│   ├── gt_segmentations/                                                  [2]
│   ├── <plans identifier>_<configuration>/    preprocessed cases (compressed files) [2]
│   └── splits_final.json                      the K folds; written by the fist train call [3]
│
└── nnUNet_results/DatasetXXX_<model>/          <-- this is the folder MLflow looks at to create artifacts
    ├── <trainer>__<plans identifier>__<configuration>/
    │   ├── plans.json, dataset.json, dataset_fingerprint.json   copies     [3]
    │   ├── fold_N/                                                         [3]
    │   │   ├── checkpoint_best.pth             best EMA pseudo-dice
    │   │   ├── checkpoint_final.pth            last epoch
    │   │   ├── checkpoint_latest.pth           periodic, polled for live MLflow metrics
    │   │   ├── progress.png, training_log_*.txt, debug.json
    │   │   └── validation/                     this fold's own validation cases
    │   │       ├── Case_NNN.nii.gz
    │   │       └── Case_NNN.npz                only if NNUNET_NPZ=True (unused here)
    │   └── crossval_results_folds_<folds>/                                [4]
    │       ├── Case_NNN.nii.gz                 validation predictions, copied from each fold
    │       ├── summary.json                    metrics before postprocessing
    │       ├── postprocessing.pkl              the chosen postprocessing (pp) method
    │       ├── postprocessing.json             before/after Dice for that pp method
    │       └── postprocessed/
    │           ├── Case_NNN.nii.gz             same cases, postprocessing applied
    │           ├── summary.json                metrics after postprocessing
    │           └── dice.png, structure_size.png                            [8]
    ├── inference_information.json              winning configuration+paths [4]
    ├── inference_instructions.txt              commands for inference + pp [4]
    ├── labelsTs_predicted/                     test-set predictions        [5]
    ├── labelsTs_predicted_pp/                  + postprocessing            [6]
    │   ├── Case_NNN.nii.gz
    │   ├── summary.json                        test metrics vs labelsTs    [7]
    │   └── dice.png, structure_size.png                                    [8]
    ├── labelsTs_predicted_rtstruct/                                        [9]
    │   └── Case_NNN.dcm                        the same predictions as RTSTRUCT
    ├── config_log.txt                          non-secret config values    [8]
    └── data_manifest.csv                       per-subject table           [8]

<system tmp>/run.log                            whole console log           [8]
```

### Test predictions as DICOM RTSTRUCT

`labelsTs_predicted_rtstruct/Case_NNN.dcm` is the same test-set prediction as
`labelsTs_predicted_pp/Case_NNN.nii.gz`, but in rtstruct format for easy import. 
It is built from the files `nnUNetv2_evaluate_folder` scored, so the test Dice
in MLflow describes what is in the dicom files.

Have a look at `rtstruct_tags()` in `src/utils/rtstruct_tools.py` to see how 
the dicom metadata is adjusted.

### What goes to MLflow

A subset of `nnUNet_results/DatasetXXX_<model>/` is written to MLflow by some functions
(`log_nnunet_artifacts`, `log_run_to_mlflow`). The MLflow tree mirrors the disk structure
above, and everything is saved as an artifact. 

### Adjustable parameters

The network architecture, patch size and batch size are not set in the config file. nnU-Net
derives them from your data (image spacing, median image size, dataset size) inside a GPU
memory budget that `NNUNET_PLANNER` sets.

So the way to train a larger network on larger patches is to move up the `NNUNET_PLANNER` preset options:

```
ExperimentPlanner  ->  nnUNetPlannerResEncM  ->  nnUNetPlannerResEncL  ->  nnUNetPlannerResEncXL
```

`ExperimentPlanner` is nnU-Net's default and targets ~8 GB of whatever GPU you are on. The ResEnc
presets use a residual-encoder network and targets roughly 8, 24 and 40 GB. Run the default first
as a baseline, then consider moving up if model performance is unsatisfactory.

To see what a run settled on, read `patch_size` and `batch_size` from the plans file
(`<plans identifier>.json`, logged to MLflow).

## Not included (follow-ups)

- **`3d_cascade_fullres` inference.** is not supported.
