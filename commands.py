import fire

from cifar10.infer import main as infer_main
from cifar10.train import main as train_main

# from hydra import compose, initialize_config_dir


class Commands:
    def train(self):
        """Запускает процесс обучения."""
        train_main()

    def infer(self, checkpoint_name: str):
        """Запускает процесс инференса.

        Args:
            test_dir: Путь к директории с тестовыми данными.
            checkpoint_name: Имя контрольной точки для загрузки.
        """
        infer_main(checkpoint_name)


if __name__ == "__main__":
    fire.Fire(Commands)
