import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
from torchvision.transforms import v2
from tqdm import trange


def make_dataloaders(root_path: Path, batch_size: int, out_size: int | tuple[int, int]) -> tuple[DataLoader, DataLoader, DataLoader]:
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


def train_model(model: nn.Module, dataloader: DataLoader, num_epochs: int, device: torch.Device = None) -> nn.Module:
    """Train a model on some data for some number of epochs. Return the trained model."""
    # loss function, optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # training loop
    model.train().to(device=device)
    for epoch in trange(num_epochs, desc="Training", unit="epoch"):
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device=device), targets.to(device=device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

    return model


def eval_model(model: nn.Module, dataloader: DataLoader, device: torch.Device = None) -> float:
    """Evaluate a model on some data. Return accuracy measurement."""
    correct_pred, total_pred = 0, 0
    model.eval().to(device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device=device), targets.to(device=device)
            outputs = model(inputs)
            predictions = outputs.argmax(axis=-1)
            correct_pred += sum(predictions == targets)
            total_pred += targets.size(0)

    return (correct_pred / total_pred).item()


def count_model_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


if __name__ == "__main__":
    # Setup command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", default="street-location-images/preprocessed_mapped",
                        help="Path to the dataset folder. Relative paths are interpreted to be within the datasets directory. Absolute paths are used directly.")
    parser.add_argument("--batch", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device used for training. Option cuda falls back on cpu if unavailable.")

    args = parser.parse_args()

    # Dataset preparation
    arg_path = Path(args.datapath)
    if arg_path.is_absolute():
         final_dataset_path = arg_path
    else:
        final_dataset_path = Path(__file__).parent.parent / "datasets" / arg_path
    H, W = 90, 90 # final image height and width
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(final_dataset_path, args.batch, (H, W))
    N_CLASSES = len(train_dataloader.dataset.classes)

    # Training preparation and execution
    if args.device == "cuda" and torch.cuda.is_available():
        TRAIN_DEVICE = torch.device("cuda")
    else:
        TRAIN_DEVICE = torch.device("cpu")
    
    model = resnet18(num_classes=N_CLASSES) # use predefined ResNet as placeholder
    print(f"Model has {count_model_parameters(model):,} parameters.")
    model = train_model(model, train_dataloader, args.epochs, TRAIN_DEVICE)
    print("Done training! Proceeding with evaluation.")

    # Evaluation (placeholder)
    EVAL_DEVICE = TRAIN_DEVICE
    val_accuracy = eval_model(model, val_dataloader, EVAL_DEVICE)
    print(f"Validation accuracy: {val_accuracy:.3%}")
    test_accuracy = eval_model(model, test_dataloader, EVAL_DEVICE)
    print(f"Test accuracy: {test_accuracy:.3%}")
