import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from dataset_loader import test_loader
from model import create_model


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Create model
model = create_model()

# Load trained model
model.load_state_dict(
    torch.load("../models/pcos_model.pth", map_location=device)
)

model = model.to(device)
model.eval()


all_labels = []
all_predictions = []
all_probabilities = []


# Test the model
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        probabilities = torch.softmax(outputs, dim=1)

        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )


# Accuracy
accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# Confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

tn, fp, fn, tp = cm.ravel()


# Sensitivity
sensitivity = tp / (tp + fn)


# Specificity
specificity = tn / (tn + fp)


# AUC
auc = roc_auc_score(
    all_labels,
    all_probabilities
)


print("\n========== TEST RESULTS ==========")

print(f"Accuracy    : {accuracy:.4f}")
print(f"Sensitivity : {sensitivity:.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"AUC         : {auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_loader.dataset.classes
    )
)