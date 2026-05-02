from multiprocessing import cpu_count
from torchvision import datasets, transforms
import torch
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloader(root_path, image_size, batch_size, workers=multiprocessing.cpu_count()):
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    root = Path(root_path)

    try:
        # Official torchvision StanfordCars layout:
        # <root>\stanford_cars\cars_train, cars_test, devkit, ...
        dataset_train = datasets.StanfordCars(root=str(root), split="train", download=False, transform=transform)
        dataset_test = datasets.StanfordCars(root=str(root), split="test", download=False, transform=transform)
    except Exception:
        # Kaggle layout:
        # <root>\stanford_cars\car_data\train|test
        train_dir = root / "stanford_cars" / "car_data" / "train"
        test_dir = root / "stanford_cars" / "car_data" / "test"
        if not train_dir.exists() or not test_dir.exists():
            raise RuntimeError(
                f"Dataset not found.\nExpected either StanfordCars layout under {root / 'stanford_cars'} "
                f"or Kaggle layout under {root / 'stanford_cars' / 'car_data'}."
            )
        dataset_train = datasets.ImageFolder(str(train_dir), transform=transform)
        dataset_test = datasets.ImageFolder(str(test_dir), transform=transform)

    dataset = ConcatDataset([dataset_train, dataset_test])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
