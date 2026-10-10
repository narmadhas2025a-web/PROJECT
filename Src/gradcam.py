import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision import models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# 1. PROJECT PATHS


# Location of this file:
# PCOS_Project/Src/gradcam.py

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up:
# PCOS_Project
PROJECT_DIR = os.path.dirname(SRC_DIR)

# Define project folders
MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "pcos_model.pth"
)

DATASET_PATH = os.path.join(
    PROJECT_DIR,
    "Dataset_split",
    "test",
    "PCOS",
    "Image_resize_004.jpg"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)

OUTPUT_PATH = os.path.join(
    RESULTS_DIR,
    "gradcam_result.jpg"
)


# 2. CREATE RESULTS FOLDER


os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)



# 3. DEVICE


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)



# 4. CHECK FILES


print("\nChecking files...")

print("Model path:")
print(MODEL_PATH)

print("\nImage path:")
print(DATASET_PATH)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel file not found:\n{MODEL_PATH}"
    )

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"\nImage file not found:\n{DATASET_PATH}"
    )

print("\nRequired files found successfully!")



# 5. LOAD TRAINED RESNET50 MODEL


print("\nLoading ResNet50 model...")

model = models.resnet50(
    weights=None
)

num_features = model.fc.in_features

model.fc = torch.nn.Linear(
    num_features,
    2
)


# Load trained weights
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")



# 6. IMAGE PREPROCESSING


transform = transforms.Compose([
    
    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# 7. LOAD TEST IMAGE


print("\nLoading test image...")

original_image = Image.open(
    DATASET_PATH
).convert("RGB")

print("Image loaded successfully!")



# 8. PREPARE INPUT TENSOR


input_tensor = transform(
    original_image
).unsqueeze(0)

input_tensor = input_tensor.to(device)



# 9. PREDICTION


print("\nRunning prediction...")

with torch.no_grad():

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()



# 10. CLASS INFORMATION


classes = [
    "Non_PCOS",
    "PCOS"
]

predicted_label = classes[
    predicted_class
]

confidence = probabilities[
    0,
    predicted_class
].item()



print("        MODEL PREDICTION")


print(
    "Prediction:",
    predicted_label
)

print(
    f"Confidence: {confidence * 100:.2f}%"
)




# 11. GRAD-CAM


print("\nGenerating Grad-CAM...")

# Last convolutional layer of ResNet50
target_layer = model.layer4[-1]


cam = GradCAM(
    model=model,
    target_layers=[
        target_layer
    ]
)


# Explain the predicted class
targets = [
    ClassifierOutputTarget(
        predicted_class
    )
]


grayscale_cam = cam(
    input_tensor=input_tensor,
    targets=targets
)[0]



# 12. PREPARE ORIGINAL IMAGE


rgb_image = np.array(
    original_image.resize(
        (224, 224)
    )
)

rgb_image = rgb_image.astype(
    np.float32
) / 255.0



# 13. CREATE GRAD-CAM VISUALIZATION


visualization = show_cam_on_image(
    rgb_image,
    grayscale_cam,
    use_rgb=True
)


# 14. SAVE GRAD-CAM RESULT


plt.imsave(
    OUTPUT_PATH,
    visualization
)



# 15. FINAL OUTPUT



print("       GRAD-CAM COMPLETED")


print(
    "Grad-CAM saved to:"
)

print(
    OUTPUT_PATH
)

