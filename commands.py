import fire

from cifar10.infer import main as infer_main
from cifar10.train import main as train_main


class Commands:
    def train(self):
        """Запускает процесс обучения."""
        train_main()

    def infer(self, test_dir: str, checkpoint_name: str):
        """Запускает процесс инференса.

        Args:
            test_dir: Путь к директории с тестовыми данными.
            checkpoint_name: Имя контрольной точки для загрузки.
        """
        infer_main(test_dir, checkpoint_name)


if __name__ == "__main__":
    fire.Fire(Commands)
