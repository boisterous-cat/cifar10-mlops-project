import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy


class ImageClassifier(pl.LightningModule):
    """Module for training and evaluation models
    for the classification task
    """

    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, logits = batch
        preds = self(data)
        acc = self.accuracy(preds, logits)
        loss = self.loss_fn(preds, logits)
        self.log("val_accuracy", acc)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        data, logits = batch
        preds = self(data)
        acc = self.accuracy(preds, logits)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return {"test_acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
