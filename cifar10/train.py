import os

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from constants import (
    BATCH_SIZE,
    DATA_PATH,
    IMAGE_MEAN,
    IMAGE_SIZE,
    IMAGE_STD,
    LR,
    MODELS_PATH,
    N_EPOCHS,
    NUM_WORKERS,
)
from model import MyResNet
from torch.utils.data import DataLoader
from trainer import ImageClassifier


def main():
    # Define the training data transformations
    composed_train = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.1),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=0.1),
            transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=0.1),
            transforms.RandomApply([transforms.ColorJitter(saturation=0.1)], p=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            transforms.RandomErasing(
                p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False
            ),
        ]
    )

    # Define the test data transformations
    composed_val = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "train"), transform=composed_train
    )

    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(DATA_PATH, "test"), transform=composed_val
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = MyResNet()
    module = ImageClassifier(model, LR)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath=MODELS_PATH,
        filename="model_{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=N_EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(module, train_loader, test_loader)


if __name__ == "__main__":
    main()
