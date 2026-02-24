import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ImageNet normalization (padrão para Transfer Learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(train: bool = True):

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4
):
    train_dir = os.path.join(data_dir, "seg_train")
    test_dir = os.path.join(data_dir, "seg_test")

    # Dataset base (apenas para split)
    base_train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=True)
    )

    val_size = int(len(base_train_dataset) * val_split)
    train_size = len(base_train_dataset) - val_size

    train_indices, val_indices = random_split(
        range(len(base_train_dataset)),
        [train_size, val_size]
    )

    # Dataset de treino
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=True)
    )
    train_dataset.samples = [train_dataset.samples[i] for i in train_indices.indices]

    # Dataset de validação
    val_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_transforms(train=False)
    )
    val_dataset.samples = [val_dataset.samples[i] for i in val_indices.indices]

    # Dataset de teste (REAL)
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
