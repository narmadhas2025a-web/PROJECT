import torch
import torch.nn as nn
import torch.optim as optim

from dataset_loader import train_loader, val_loader
from model import create_model


# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Create model
# -----------------------------
model = create_model()
model = model.to(device)


# -----------------------------
# Loss function
# -----------------------------
criterion = nn.CrossEntropyLoss()


# -----------------------------
# Optimizer
# -----------------------------
optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001
)


# -----------------------------
# Training settings
# -----------------------------
EPOCHS = 15

best_val_accuracy = 0.0

patience = 3
patience_counter = 0


# -----------------------------
# Training
# -----------------------------
print("\nStarting training...\n")


for epoch in range(EPOCHS):

    # =========================
    # TRAINING
    # =========================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        train_total += labels.size(0)

        train_correct += (
            predictions == labels
        ).sum().item()


    train_accuracy = (
        100 * train_correct / train_total
    )


    # =========================
    # VALIDATION
    # =========================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()


    val_accuracy = (
        100 * val_correct / val_total
    )


    # =========================
    # PRINT RESULTS
    # =========================

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Accuracy: "
        f"{train_accuracy:.2f}% | "
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


    # =========================
    # SAVE BEST MODEL
    # =========================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "../models/pcos_model.pth"
        )

        print(
            "  ✓ Best model saved!"
        )

        patience_counter = 0

    else:

        patience_counter += 1

        print(
            f"  No improvement "
            f"({patience_counter}/{patience})"
        )


    # =========================
    # EARLY STOPPING
    # =========================

    if patience_counter >= patience:

        print(
            "\nEarly stopping triggered."
        )

        break


print("\n==============================")
print("Training completed!")
print("==============================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    "Best model saved to:"
    " models/pcos_model.pth"
)