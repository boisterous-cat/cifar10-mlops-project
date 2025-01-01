import os
import subprocess

import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from hydra import compose, initialize
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .model import MyResNet
from .trainer import ImageClassifier


def main(config_name: str = "config", config_path: str | None = None) -> None:
    config_path = config_path or "../conf"
    with initialize(version_base=None, config_path=config_path):
        # Load config
        # we need return_hydra_config=True for resolve hydra.runtime.cwd etc
        cfg: DictConfig = compose(config_name=config_name, return_hydra_config=True)

        # Pull train data from DVC
        subprocess.run(["dvc", "pull", "data/train"], check=True)
        # Define the training data transformations
        composed_train = transforms.Compose(
            [
                transforms.Resize(
                    (cfg["model"]["image_size"], cfg["model"]["image_size"])
                ),
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.1),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.1)], p=0.1),
                transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=0.1),
                transforms.RandomApply([transforms.ColorJitter(saturation=0.1)], p=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg["model"]["image_mean"], cfg["model"]["image_std"]
                ),
                transforms.RandomErasing(
                    p=0.75, scale=(0.02, 0.1), value=1.0, inplace=False
                ),
            ]
        )

        # Define the test data transformations
        composed_val = transforms.Compose(
            [
                transforms.Resize(
                    (cfg["model"]["image_size"], cfg["model"]["image_size"])
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    cfg["model"]["image_mean"], cfg["model"]["image_std"]
                ),
            ]
        )

        train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(cfg["data_loading"]["train_data_path"]),
            transform=composed_train,
        )

        val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(cfg["data_loading"]["val_data_path"]), transform=composed_val
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"]["num_workers"],
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
        )

        model = MyResNet()
        module = ImageClassifier(model, lr=cfg["training"]["lr"])

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=cfg["model"]["model_local_path"],
            filename="model_{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=cfg["training"]["n_epochs"],
            accelerator="auto",
            devices="auto",
            callbacks=[checkpoint_callback],
        )

        trainer.fit(module, train_loader, val_loader)


# if __name__ == "__main__":
#     # subprocess.run(["dvc", "pull", "data/train.dvc"])
#     #fire.Fire(main)
#     main()
