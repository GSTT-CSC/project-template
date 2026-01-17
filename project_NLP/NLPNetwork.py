import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.functional import auroc
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizerFast as BertTokenizer
import mlflow
from torchmetrics.functional import precision, recall, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


class NLPNetwork(pl.LightningModule):

    def __init__(
        self,
        n_classes: int,
        n_training_steps=None,
        n_warmup_steps=None,
        learning_rate=None,
        label_columns: list = None,
    ):
        super(NLPNetwork, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_classes = n_classes
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.BCELoss()
        self.learning_rate = learning_rate
        self.label_columns = label_columns
        self.training_step_outputs = []
        self.training_step_labels = []
        self.training_step_loss = []
        self.val_step_outputs = []
        self.val_step_labels = []
        self.val_step_loss = []
        self.val_step_id = []

    def forward(self, input_ids, attention_mask, labels=None):
        """
        The forward pass for the model.

        Args:
            input_ids: Input features from the tokenizer.
            attention_mask: Attention mask values. Identifies which tokens should be attended to by the model.
            labels: Actual labels. Only provided during training.

        Returns:
            loss: Loss calculated using the Binary Cross Entropy Loss function.
            output: Output from the classifier.
        """

        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        """
        Defines a single step during training. It calculates loss, predictions and accuracy.

        Args:
            batch: Data batch that is loaded from the DataLoader.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing loss, predictions, labels and accuracy for the step.
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        accuracy = self.accuracy(outputs, labels)
        self.training_step_outputs.append(outputs.detach().cpu())
        self.training_step_labels.append(labels.detach().cpu())
        self.training_step_loss.append(loss.detach().cpu())
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "accuracy": accuracy,
        }

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step during validation. Similar to the training step, it calculates loss, predictions, and accuracy.

        Args:
            batch: Data batch that is loaded from the DataLoader.
            batch_idx: Index of the batch.

        Returns:
            A dictionary containing loss, predictions, labels and accuracy for the step.
        """

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        accuracy = self.accuracy(outputs, labels)

        self.val_step_id.append(batch["input_ids"].detach().cpu())
        self.val_step_outputs.append(outputs.detach().cpu())
        self.val_step_labels.append(labels.detach().cpu())
        self.val_step_loss.append(loss.detach().cpu())
        return {
            "loss": loss,
            "predictions": outputs,
            "labels": labels,
            "accuracy": accuracy,
        }

    def accuracy(self, logits, labels):
        """
        Compute accuracy.

        Args:
            logits: Model's predictions.
            labels: True labels.

        Returns:
            Tensor value of the accuracy of the model.
        """
        _, x = torch.max(logits.data, 1)
        _, y = torch.max(labels.data, 1)
        correct = (x == y).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def on_train_epoch_end(self):
        """
        Operations to perform at the end of each training epoch.
        """

        avg_loss = torch.stack(self.training_step_loss).mean()
        avg_acc = torch.stack(
            [
                self.accuracy(outputs, labels)
                for outputs, labels in zip(
                    self.training_step_outputs, self.training_step_labels
                )
            ]
        ).mean()
        self.log("avg_train_loss", avg_loss)
        self.log("avg_train_accuracy", avg_acc)

        self.training_step_outputs.clear()
        self.training_step_labels.clear()
        self.training_step_loss.clear()

    def on_validation_epoch_end(self):
        """
        Operations to perform at the end of each validation epoch. Logs any relevant metrics/misclassified sentences to mlflow.
        """

        labels = []
        predictions = []
        input_ids = []
        for i in range(len(self.val_step_labels)):
            if i == len(self.val_step_labels) - 1:
                labels.append(
                    self.val_step_labels[i][: self.val_step_outputs[i].size(0)].int()
                )
                predictions.append(self.val_step_outputs[i])
                input_ids.append(self.val_step_id[i])
            else:
                labels.append(self.val_step_labels[i].int())
                predictions.append(self.val_step_outputs[i])
                input_ids.append(self.val_step_id[i])

        labels = torch.cat(labels, dim=0)
        predictions = torch.cat(predictions, dim=0)

        for i, name in enumerate(self.label_columns):
            # Logging F1 score for each class
            class_roc_auc = f1_score(predictions[:, i], labels[:, i], task="binary")
            self.log(f"{name}_roc_auc/Validation", float(class_roc_auc))

            # Logging Precision for each class
            class_precision = precision(predictions[:, i], labels[:, i], task="binary")
            self.log(f"{name}_precision/Validation", float(class_precision))

            # Logging Recall for each class
            class_recall = recall(predictions[:, i], labels[:, i], task="binary")
            self.log(f"{name}_recall/Validation", float(class_recall))

        # Log misclassified sentences
        if self.trainer.current_epoch == self.trainer.max_epochs - 1:
            misclassified_sentences = []
            for idx, (output, label) in enumerate(zip(predictions, labels)):
                predicted_label = torch.argmax(output)
                true_label = torch.argmax(label)
                if (predicted_label != true_label) and (idx < len(input_ids[0])):
                    sentence = self.tokenizer.decode(input_ids[0][idx])
                    sentence_info = {
                        "sentence": sentence,
                        "predicted_label": predicted_label.item(),
                        "true_label": true_label.item(),
                    }
                    misclassified_sentences.append(sentence_info)

                file_path = "misclassified_sentences.txt"
                with open(file_path, "w", encoding="utf-8") as file:
                    for idx, sentence_info in enumerate(misclassified_sentences):
                        file.write(
                            f"Misclassified Sentence {idx+1}: {sentence_info['sentence']}\n"
                        )
                        file.write(
                            f"Predicted Label {idx+1}: {sentence_info['predicted_label']}\n"
                        )
                        file.write(
                            f"True Label {idx+1}: {sentence_info['true_label']}\n\n"
                        )

            # Log the misclassified sentences file as an artifact in MLflow
            mlflow.log_artifact(file_path, artifact_path="misclassified_sentences")

        avg_loss = torch.stack(self.val_step_loss).mean()
        self.log("avg_val_loss", avg_loss)
        avg_acc = torch.stack(
            [
                self.accuracy(outputs, labels)
                for outputs, labels in zip(self.val_step_outputs, self.val_step_labels)
            ]
        ).mean()
        self.log("avg_val_accuracy", avg_acc)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        self.val_step_loss.clear()
        self.val_step_id.clear()

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for the training.

        Returns:
            A dictionary containing optimizer and lr_scheduler.
        """

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps,
        )

        return dict(
            optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step")
        )
