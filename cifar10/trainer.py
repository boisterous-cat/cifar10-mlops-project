import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy


class ImageClassifier(pl.LightningModule):
    """
    A PyTorch Lightning Module for image classification.

    This class encapsulates the model architecture, training, validation,
    and testing processes for a multi-class image classification task.

    Args:
        model (torch.nn.Module): The model architecture to be trained.
        lr (float): The learning rate for the optimizer.

    Attributes:
        model (torch.nn.Module): The model architecture to be trained.
        lr (float): The learning rate for the optimizer.
        accuracy (torchmetrics.Accuracy): Accuracy metric for evaluating model performance.
        loss_fn (torch.nn.CrossEntropyLoss): Cross-entropy loss function for training.

    """

    def __init__(self, model, lr):
        """
        Initializes the ImageClassifier with the specified model and learning rate.

        Args:
            model (torch.nn.Module): The model architecture to be trained.
            lr (float): The learning rate for the optimizer.
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor to the model.

        Returns:
            torch.Tensor: Output tensor from the model.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (tuple): A tuple containing (data, labels) for the current batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the current batch.
        """
        data, logits = batch
        preds = self(data)
        loss = self.loss_fn(preds, logits)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (tuple): A tuple containing (data, labels) for the current batch.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Computed loss for the current batch.
        """
        data, logits = batch
        preds = self(data)
        acc = self.accuracy(preds, logits)
        loss = self.loss_fn(preds, logits)
        self.log("val_accuracy", acc)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the test step.

        Args:
            batch (tuple): A tuple containing (data, labels) for the current batch.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the test accuracy.
        """
        data, logits = batch
        preds = self(data)
        acc = self.accuracy(preds, logits)
        self.log("test_acc", acc, prog_bar=True, on_epoch=True)
        return {"test_acc": acc}

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The optimizer to be used for training.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
