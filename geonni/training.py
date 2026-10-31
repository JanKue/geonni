from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import trange


def get_dataloaders(root_path: Path, batch_size: int, out_size: int | tuple[int, int]) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train/val/test dataloaders for the dataset at a given root path. The train split data is augmented and shuffled."""
    shared_transforms = v2.Compose([v2.Resize(size=out_size),
                                   v2.ToImage(),
                                   v2.ToDtype(torch.float32, scale=True),
                                   v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
    deterministic_transforms = v2.Compose([v2.CenterCrop(size=180), shared_transforms])
    augment_transforms = v2.Compose([v2.RandomCrop(size=180),
                                     v2.RandomHorizontalFlip(),
                                     v2.RandomRotation(degrees=10),
                                     v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                     shared_transforms])

    train_dataset = ImageFolder(root_path / "train", transform=augment_transforms)
    val_dataset = ImageFolder(root_path / "val", transform=deterministic_transforms)
    test_dataset = ImageFolder(root_path / "test", transform=deterministic_transforms)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    DATASET_PATH = Path(__file__).parent.parent / "datasets/street-location-images/preprocessed_mapped"
    BATCH_SIZE = 32
    H, W = 90, 90 # final image height and width
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(DATASET_PATH, BATCH_SIZE, (H, W))
    N_CLASSES = len(train_dataloader.dataset.classes)

    # Ready training
    TRAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # use predefined ResNet as placeholder
    model = resnet18(num_classes=N_CLASSES)
    # loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    N_EPOCHS = 10
    model.train().to(TRAIN_DEVICE)
    for epoch in trange(N_EPOCHS, desc="Training", unit="epoch"):
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(TRAIN_DEVICE), targets.to(TRAIN_DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

    print("Done training! Proceeding with evaluation.")

    # Validation loop (placeholder)
    EVAL_DEVICE = TRAIN_DEVICE
    correct_pred, total_pred = 0, 0
    model.eval().to(EVAL_DEVICE)
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(EVAL_DEVICE), targets.to(EVAL_DEVICE)
            outputs = model(inputs)
            predictions = outputs.argmax(axis=-1)
            correct_pred += sum(predictions == targets)
            total_pred += targets.size(0)

    print(f"Validation accuracy: {correct_pred / total_pred:.3%}")
