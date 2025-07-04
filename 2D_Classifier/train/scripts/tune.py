
import configparser
import logging
import multiprocessing
import os

import mlflow
import pytorch_lightning as pl
from ray.air.integrations.mlflow import setup_mlflow
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import is_available as cuda_available

from project.DataModule import DataModule
from project.DataModule import label_dict
from project.Network import Network
from project.XNATDataImport import XNATDataImport

import optuna
logger = logging.getLogger(__name__)

# Obtain hyperparameters for this trial
def suggest_hyperparameters(trial):

    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    max_lr = trial.suggest_categorical("max_lr",[5e-4,1e-3])
    model = trial.suggest_categorical("model", ["convnextv2_tiny.fcmae_ft_in22k_in1k","convnextv2_base.fcmae_ft_in22k_in1k"])
    batch_size = trial.suggest_int("batch_size", 4, 16, step=4)
    pretrained = trial.suggest_categorical("pretrained", ["TRUE"])
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)
    grad_batches = trial.suggest_int('grad_batches',1,4)
    image_size = trial.suggest_categorical("image_size", [224])

    params = {
        "dropout": dropout,
        "lr": lr,
        "max_lr": max_lr,
        "model": model,
        "batch_size": batch_size,
        "pretrained": pretrained,
        "label_smoothing": label_smoothing,
        "grad_batches": grad_batches,
        "image_size": image_size
    }
    return params

def objective(trial,data,config):

    max_workers = 32
    num_workers = (
        max_workers
        if max_workers < multiprocessing.cpu_count()
        else multiprocessing.cpu_count()
    )

    best_val_loss = float('Inf')
    
    # Start a new mlflow run
    with mlflow.start_run(nested=True):

        params = suggest_hyperparameters(trial)
        mlflow.log_params(params)
  
        # initialise network and datamodule
        dm = DataModule(
            data = data,
            dm_batch_size = int(config['params']['dm_batch_size']),
            test_fraction = float(config['params']['test_fraction']),
            num_workers = num_workers,
            random_seed = int(config['params']['random_seed']),
            image_size = params['image_size']
        )

        dm.setup()
        
        n_classes = len(set([x for x in label_dict.values() if x is not None]))
        mlflow.log_param('n_classes', n_classes)

        train_class_weights = dm.data_manifest["train"]["class_weights"]
        validation_class_weights = dm.data_manifest["validation"]["class_weights"]

        net = Network(
            model_name = params['model'],
            pretrained = params['pretrained'],
            n_classes = n_classes,
            dropout = params['dropout'],
            weighted_loss = config['params']['weighted_loss'],
            train_class_weights = train_class_weights,
            validation_class_weights = validation_class_weights,
            learning_rate = params['lr'],
            max_lr = params['max_lr'],
            nw_batch_size = params['batch_size'],
            label_smoothing = params['label_smoothing']
        )

        # Callbacks
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath= f"./checkpoint_trial_{trial.number}/",
        )
        callbacks.append(checkpoint_callback)

        # configure trainer
        trainer = pl.Trainer(
            precision="32" if cuda_available() else "16",
            callbacks=callbacks,
            devices= 1 if cuda_available() else "auto",
            accelerator="gpu" if cuda_available() else "auto",
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=int(config['params']['max_epochs']),
            accumulate_grad_batches= params['grad_batches']
        )

        trainer.fit(net, dm)

        best_model_path = checkpoint_callback.best_model_path
        best_model = net.load_from_checkpoint(best_model_path)
        net.evaluate_best_model(best_model)

        val_loss = checkpoint_callback.best_model_score.item()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

def tune(config):

    os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["cuda_visible_devices"]

    xnat_configuration = {'server': config['xnat']['SERVER'],
                          'user': config['xnat']['USER'],
                          'password': config['xnat']['PASSWORD'],
                          'project': config['xnat']['PROJECT'],
                          'verify': config.getboolean('xnat', 'VERIFY')}

    max_workers = 32
    num_workers = (
        max_workers
        if max_workers < multiprocessing.cpu_count()
        else multiprocessing.cpu_count()
    )

    importer = XNATDataImport(
        xnat_configuration = xnat_configuration,
        num_workers = num_workers
        )

    # Import raw data
    raw_data = importer.import_xnat_data()

    # Download images from XNAT
    data = importer.xnat_image_download(raw_data)
    
    # Set up mflow experiment
    setup_mlflow(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_id=mlflow.get_experiment_by_name(
            config["project"]["name"]
        ).experiment_id
        if mlflow.get_experiment_by_name(config["project"]["name"])
        else mlflow.create_experiment(config["project"]["name"]),
    )

    mlflow.pytorch.autolog(log_models=False)

    # Create optuna study (hyperparameter tuning framework)
    study = optuna.create_study(study_name="scaphx-tune", direction="minimize")
    study.optimize(lambda trial: objective(trial, data, config), n_trials=50)

    with open(('tune_log.txt'), 'w') as f:
        f.write("Study statistics: \n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")

        trial = study.best_trial

        f.write(f"  Trial number: {trial.number}\n")
        f.write(f"  Loss (trial value): {trial.value}\n")

        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"      {key} = {value}\n")
    
    mlflow.log_artifact('tune_log.txt')
    mlflow.log_artifact('val_transform_failures.csv')
    mlflow.log_artifact('train_transform_failures.csv')
    
if __name__ == '__main__':

    
    config_path = 'config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    tune(config)
