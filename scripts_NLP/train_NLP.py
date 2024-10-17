import os
import logging
import sys
import configparser
import torch
import mlflow
import pytorch_lightning as pl
from torch.cuda import is_available as cuda_available
from transformers import BertTokenizerFast as BertTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
import datetime

from project_NLP.NLPDataModule import NLPDataModule
from project_NLP.NLPNetwork import NLPNetwork
from project_NLP.utils.tools import wrap_and_log


def train(
    data_path,
    n_epochs=1,
    batch_size=6,
    model_name="bert-base-uncased",
    learning_rate=1e-5,
    use_kfold=False,
    n_folds=0,
    random_state=42,
    save_model=False,
):
    """
    This function is used to train any of the EndominerAi models (which model depends on the config).
    The function uses PyTorch Lightning's built-in ModelCheckpoint callback to save the best model (based on validation
    accuracy).
    The best trained model is then wrapped and logged to MLFlow for deployment.

    Args:
        data_path (str): The path to the data.
        n_epochs (int, optional): The number of epochs to train for. Defaults to 1.
        batch_size (int, optional): The batch size for training. Defaults to 6.

    Returns:
        None
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    data_module = NLPDataModule(
        data_path=data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_token_len=512,
        random_state=random_state,
    )

    data_module.setup()
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.end_run()

    # KFold, data preparation
    with mlflow.start_run(run_name="Parent_Run_{current_time}"):
        if use_kfold and n_folds > 1:

            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kfold.split(data_module.df)):
                logging.info(f"Training fold {fold+1}/{n_folds}")

                run_name = f"{'KFold_' + str(fold + 1) if fold >= 0 else 'Train'}_{current_time}"

                mlflow.pytorch.autolog(log_models=False)
                with mlflow.start_run(run_name=run_name, nested=True):

                    # Initialize DataModule for current fold
                    data_module = NLPDataModule(
                        data_path=data_path,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        max_token_len=512,
                        fold_indices=(train_idx, val_idx),
                        random_state=random_state,
                    )
                    data_module.setup()
                    total_training_steps = data_module.steps_per_epoch() * n_epochs
                    warmup_steps = total_training_steps // 5

                    model = NLPNetwork(
                        n_classes=data_module.num_classes,
                        n_warmup_steps=warmup_steps,
                        n_training_steps=total_training_steps,
                        learning_rate=learning_rate,
                        label_columns=data_module.label_columns,
                    )

                    mlflow_logging_and_checkpoint(
                        model=model,
                        data_module=data_module,
                        fold=-1,
                        n_epochs=n_epochs,
                        tokenizer=tokenizer,
                        run_name=run_name,
                        save=save_model,
                    )

        else:
            mlflow.pytorch.autolog(log_models=False)
            with mlflow.start_run(run_name="child_Run_{current_time}", nested=True):

                total_training_steps = data_module.steps_per_epoch() * n_epochs
                warmup_steps = total_training_steps // 5

                model = NLPNetwork(
                    n_classes=data_module.length_label_dict,
                    n_warmup_steps=warmup_steps,
                    n_training_steps=total_training_steps,
                    learning_rate=learning_rate,
                    label_columns=data_module.label_columns,
                )

                run = mlflow.active_run()
                run_name = run.info.run_id
                # Setup MLflow and checkpointing for standard training
                mlflow_logging_and_checkpoint(
                    model=model,
                    data_module=data_module,
                    fold=-1,
                    n_epochs=n_epochs,
                    tokenizer=tokenizer,
                    run_name=run_name,
                    save=save_model,
                )


def mlflow_logging_and_checkpoint(
    model, data_module, fold, n_epochs, tokenizer, run_name, save
):

    checkpoint_callback = ModelCheckpoint(
        monitor="avg_val_accuracy",
        dirpath="./checkpoints/",
        filename=f"{run_name}-best_model",
        save_top_k=1,
        mode="max",
    )

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=n_epochs,
        logger=True,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
    )

    trainer.fit(model, data_module)

    # Log the best model for each fold or standard training run
    if save:
        print(f"Logging best model for {run_name}")
        checkpoint = torch.load(checkpoint_callback.best_model_path)
        model.load_state_dict(checkpoint["state_dict"])
        wrap_and_log(model, tokenizer)


if __name__ == "__main__":
    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = "config_NLP/local_config.cfg"

    config = configparser.ConfigParser()
    config.read(config_path)

    data_path = config["data"]["DATA_PATH"]
    learning_rate = float(config["training"]["LEARNING_RATE"])
    batch_size = int(config["training"]["BATCH_SIZE"])
    model_name = config["training"]["MODEL_NAME"]
    n_epochs = int(config["training"]["N_EPOCHS"])
    use_kfold = config.getboolean("training", "KFOLD")
    n_folds = int(config["training"]["N_FOLDS"]) if use_kfold else 0
    save = config["training"]["SAVE_MODEL"]

    random_state = int(config["training"]["RANDOM_STATE"])

    train(
        data_path,
        n_epochs,
        batch_size,
        model_name,
        learning_rate,
        use_kfold,
        n_folds,
        random_state=random_state,
        save_model=save,
    )
