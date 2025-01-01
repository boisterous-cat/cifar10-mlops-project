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


def main(
    checkpoint_name: str, config_name: str = "config", config_path: str | None = None
) -> None:
    config_path = config_path or "../conf"
    with initialize(version_base=None, config_path=config_path):
        cfg: DictConfig = compose(config_name=config_name, return_hydra_config=True)
        # Pull train data from DVC
        subprocess.run(["dvc", "pull", "data/test"], check=True)

    # Define the test data transformations
    composed_test = transforms.Compose(
        [
            transforms.Resize((cfg["model"]["image_size"], cfg["model"]["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["model"]["image_mean"], cfg["model"]["image_std"]),
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(cfg["data_loading"]["test_data_path"]), transform=composed_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    model = MyResNet()
    model_path = os.path.join(cfg["model"]["model_local_path"])
    module = ImageClassifier.load_from_checkpoint(
        f"{model_path}/{checkpoint_name}", model=model, lr=cfg["training"]["lr"]
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    results = trainer.test(module, dataloaders=test_loader)
    print(results)
