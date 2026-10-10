from torchvision import datasets
from torch.utils.data import DataLoader

from preprocessing import train_transform, test_transform


# Dataset paths
TRAIN_DIR = "../dataset_split/train"
VAL_DIR = "../dataset_split/validation"
TEST_DIR = "../dataset_split/test"


# Load datasets
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Display information
print("Classes:", train_dataset.classes)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))
print("Testing images:", len(test_dataset))

print("Dataset loader created successfully!")