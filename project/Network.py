from abc import ABC

import mlflow
import pytorch_lightning
import torch
from timm import create_model
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall, AUROC
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt

class Network(pytorch_lightning.LightningModule, ABC):

    """
    Network object defines the model architecture, inherits from LightningModule.

    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self,
                model_name,
                pretrained,
                n_classes,
                dropout,
                weighted_loss,
                train_class_weights,
                validation_class_weights,
                learning_rate,
                batch_size,
                **kwargs):
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.n_classes = n_classes
        self.dropout = dropout
        self.weighted_loss = weighted_loss
        self.train_class_weights = train_class_weights
        self.validation_class_weights = validation_class_weights
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.test_step_outputs = []
        
        self._model = create_model(
            self.model_name,
            pretrained=self.pretrained,
            in_chans=1,
            num_classes=self.n_classes,
            drop_rate=self.dropout,
        )

        mlflow.log_params(
            {"trainable_params": sum(p.numel() for p in self._model.parameters())}
        )

        self.train_loss_function = CrossEntropyLoss(
            weight=torch.tensor(self.train_class_weights)
            if self.weighted_loss
            else None
        )

        self.validation_loss_function = CrossEntropyLoss(
            weight=torch.tensor(self.validation_class_weights)
            if self.weighted_loss
            else None
        )
        
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
            top_k=1,
        )
        self.train_f1 = F1Score(
            task="multiclass",
            num_classes=self.n_classes,
            top_k=1,
        )

        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
            top_k=1,
        )
        self.val_f1 = F1Score(
            task="multiclass",
            num_classes=self.n_classes,
            top_k=1,
        )
        self.val_recall = Recall(
            task="multiclass",
            average="macro",
            num_classes=self.n_classes,
        )
        self.val_precision = Precision(
            task="multiclass",
            average="macro",
            num_classes=self.n_classes,
        )

        self.val_auroc = AUROC(
            task="multiclass",
            num_classes = self.n_classes
        )

        self.test_acc = Accuracy(
            task="multiclass",
            num_classes=self.n_classes,
            top_k=1,
        )

        self.confmat = ConfusionMatrix(
            task="multiclass",
            num_classes=self.n_classes,
        )

    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        return self._model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.train_loss_function(y_hat, y)
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size
        )

        # additional metrics
        self.train_acc(y_hat, y)
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        self.train_f1(y_hat, y)
        self.log(
            "train_f1-score",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)

        loss = self.validation_loss_function(y_hat, y)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, batch_size=self.batch_size
        )

        # additional metrics
        self.val_acc(y_hat, y)
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        self.val_f1(y_hat, y)
        self.val_recall(y_hat, y)
        self.val_precision(y_hat, y)
        self.val_auroc(y_hat, y)

        self.log(
            "val_f1-score",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_recall",
            self.val_recall,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val_auroc",
            self.val_auroc,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        """
        test step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        self.test_step_outputs.append({"y_hat": y_hat, "y": y})

        self.test_acc(y_hat, y)
        self.log(
            "test_acc",
            self.test_acc,
            on_epoch=True,
            batch_size=1,
        )

    def configure_optimizers(self):
        """
        Setup optimiser
        :return: Optimizer
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_test_end(self) -> None:
        """Actions to perform after testing is finished. Currently the confusion matrix, and classification report are calculated"""
        self.test_step_outputs
        target = torch.cat([x["y"] for x in self.test_step_outputs]).to(self.device)
        y_hat = torch.cat([x["y_hat"] for x in self.test_step_outputs]).to(self.device)
        pred = torch.argmax(y_hat, dim=1)

        # test data confusion matrix
        self.confmat(pred, target)
        fig_, ax_ = self.confmat.plot()
        mlflow.log_figure(fig_, "confusion_matrix.png")

        # test data classification report
        report = classification_report(
            target.cpu().numpy(),
            pred.cpu().numpy(),
        )
        mlflow.log_text(report, "classification_report.txt")

        # roc curve

        soft = torch.nn.Softmax(dim=1)
        predicted = soft(y_hat)
        predicted = predicted[:,1]

        fpr, tpr, threshold = roc_curve(target.cpu(), predicted.cpu())
        roc_auc = auc(fpr, tpr)

        fig_2, ax_2 = plt.subplots()

        ax_2.set_title('Receiver Operating Characteristic')
        ax_2.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        ax_2.legend(loc = 'lower right')
        ax_2.plot([0, 1], [0, 1],'r--')
        ax_2.set_xlim([0, 1])
        ax_2.set_ylim([0, 1])
        ax_2.set_ylabel('True Positive Rate')
        ax_2.set_xlabel('False Positive Rate')

        mlflow.log_figure(fig_2, "roc.png")




