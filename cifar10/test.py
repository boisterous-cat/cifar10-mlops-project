import fire
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
    NUM_WORKERS,
)
from model import MyResNet
from torch.utils.data import DataLoader
from trainer import ImageClassifier


def main(test_dir: str, checkpoint_name: str) -> None:
    # Define the test data transformations
    composed_test = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    test_dataset = torchvision.datasets.ImageFolder(
        f"{DATA_PATH}/{test_dir}", transform=composed_test
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = MyResNet()
    module = ImageClassifier.load_from_checkpoint(
        f"{MODELS_PATH}/{checkpoint_name}", model=model, lr=LR
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        log_every_n_steps=50,
    )

    results = trainer.test(module, dataloaders=test_loader)
    print(results)


if __name__ == "__main__":
    fire.Fire(main)
