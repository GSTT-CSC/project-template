import sys
import configparser
import logging
import os
import multiprocessing
import json

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from ray.air.integrations.mlflow import setup_mlflow
from torch.cuda import is_available as cuda_available

from project.DataModule import DataModule
from project.Network import Network
from project.DataModule import label_dict

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

logger = logging.getLogger(__name__)

def train(config):
    # Set up mflow experiment
    setup_mlflow(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_id=mlflow.get_experiment_by_name(
            config["project"]["name"]
        ).experiment_id
        if mlflow.get_experiment_by_name(config["project"]["name"])
        else mlflow.create_experiment(config["project"]["name"]),
    )

    # Set up variables from config
    xnat_configuration = {'server': config['xnat']['SERVER'],
                        'user': config['xnat']['USER'],
                        'password': config['xnat']['PASSWORD'],
                        'project': config['xnat']['PROJECT'],
                        'verify': config.getboolean('xnat', 'VERIFY')}

    max_workers = int(config['params']['MAX_WORKERS'])
    num_workers = (
        max_workers
        if max_workers < multiprocessing.cpu_count()
        else multiprocessing.cpu_count()
    )
       
    save_best_model = True

    mlflow.pytorch.autolog(log_models=False)

    # Set up datamodule
    dm = DataModule(
        xnat_configuration = xnat_configuration,
        num_workers = num_workers,
        batch_size = int(config['params']['batch_size']),
        )

    dm.prepare_data()

    n_classes = len(set([x for x in label_dict.values() if x is not None]))
    mlflow.log_param('n_classes', n_classes)

    train_class_weights = dm.data_manifest["train"]["class_weights"]
    validation_class_weights = dm.data_manifest["validation"]["class_weights"]

    # Set up network
    net = Network(
        # Include params here from config such as dropout, learning_rate as shown
        n_classes = n_classes,
        train_class_weights = train_class_weights,
        validation_class_weights = validation_class_weights,
        dropout = float(config['params']['dropout']),
        learning_rate = float(config['params']['learning_rate']),
    )

    # Callbacks
    callbacks = []
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoint/",
    )
    callbacks.append(checkpoint_callback)

    # configure trainer
    trainer = pl.Trainer(
        precision="16-mixed" if cuda_available() else "32-true",
        callbacks=callbacks,
        devices= "gpu" if cuda_available() else "auto",
        accelerator="gpu" if cuda_available() else "auto",
        log_every_n_steps=1,
        max_epochs=int(config['params']['max_epochs']),
    )
    
    trainer.fit(net, dm)

    logging.info("Training complete: running test set")

    # Create and log classification report
    if save_best_model:
        net_trained = Network.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            n_classes = n_classes,
            train_class_weights = train_class_weights,
            validation_class_weights = validation_class_weights,
            dropout = float(config['params']['dropout']),
            learning_rate = float(config['params']['learning_rate']),
            )
        trainer.test(
            net_trained,
            datamodule=dm,
        )
        checkpoint = torch.load(checkpoint_callback.best_model_path)

        file_path = (
            f"model-{config['project']['name']}-{mlflow.active_run().info.run_name}.pt"
        )

        checkpoint_info = {}
        checkpoint_info["monitored_metric"] = checkpoint_callback.monitor
        checkpoint_info["metric_value"] = checkpoint_callback.best_model_score.item()
        checkpoint_info["mode"] = checkpoint_callback.mode
        checkpoint_info["epoch"] = checkpoint["epoch"]

        with open("checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        script = torch.jit.script(net_trained._model)

        mlflow.pytorch.log_model(
            script, file_path, extra_files=["checkpoint_info.json"]
        )

        _ = [
            os.remove(fp) for fp in ["checkpoint_info.json"]
        ]  # remove the temporary files after logging to mlflow


if __name__ == '__main__':

    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = 'config/local_config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)

