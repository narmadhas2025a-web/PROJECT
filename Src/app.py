import os

import streamlit as st
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from torchvision import models, transforms
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PCOS AI Screening",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

# app.py is inside:
# PCOS_Project/Src/app.py

# Therefore:
# PROJECT_DIR = PCOS_Project

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Model path

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "pcos_model.pth"
)


# Results folder

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results"
)


# Create results folder if it does not exist

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# CHECK MODEL FILE
# ============================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"Model file not found:\n\n{MODEL_PATH}"
    )

    st.stop()


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # Create ResNet50

    model = models.resnet50(
        weights=None
    )


    # Get number of features

    num_features = model.fc.in_features


    # Replace final layer

    model.fc = nn.Linear(
        num_features,
        2
    )


    # Load trained weights

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )


    model.load_state_dict(
        checkpoint
    )


    # Move model to device

    model = model.to(
        device
    )


    # Evaluation mode

    model.eval()


    return model


# Load model

model = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

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


# ============================================================
# CLASS NAMES
# ============================================================

classes = [
    "Non-PCOS",
    "PCOS"
]


# ============================================================
# HEADER
# ============================================================

st.title(
    "🩺 PCOS AI Screening System"
)

st.write(
    "AI-based ovarian ultrasound image "
    "screening and explainable prediction system."
)


st.warning(
    "Research prototype only. "
    "This system is not a substitute for professional "
    "medical diagnosis."
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "System Information"
)

st.sidebar.write(
    "**Model:** ResNet50"
)

st.sidebar.write(
    "**Task:** PCOS vs Non-PCOS"
)

st.sidebar.write(
    f"**Device:** {device}"
)

st.sidebar.write(
    "**Input:** Ovarian Ultrasound Image"
)

st.sidebar.write(
    "**Explainability:** Grad-CAM"
)


# ============================================================
# UPLOAD IMAGE
# ============================================================

st.header(
    "Upload Ovarian Ultrasound Image"
)

uploaded_file = st.file_uploader(
    "Choose an ultrasound image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Uploaded Ultrasound Image"
        )

        st.image(
            image,
            use_container_width=True
        )


    # --------------------------------------------------------
    # Preprocess image
    # --------------------------------------------------------

    input_tensor = transform(
        image
    ).unsqueeze(0)


    input_tensor = input_tensor.to(
        device
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with col2:

        st.subheader(
            "AI Prediction"
        )


        with torch.no_grad():

            outputs = model(
                input_tensor
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()


        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_label = classes[
            predicted_class
        ]


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = probabilities[
            0,
            predicted_class
        ].item()


        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        st.metric(
            "Prediction",
            predicted_label
        )


        st.metric(
            "Confidence",
            f"{confidence * 100:.2f}%"
        )


        # ====================================================
        # CLASS PROBABILITIES
        # ====================================================

        st.subheader(
            "Class Probabilities"
        )


        non_pcos_probability = probabilities[
            0,
            0
        ].item()


        pcos_probability = probabilities[
            0,
            1
        ].item()


        # Non-PCOS

        st.write(
            f"Non-PCOS: "
            f"{non_pcos_probability * 100:.2f}%"
        )

        st.progress(
            non_pcos_probability
        )


        # PCOS

        st.write(
            f"PCOS: "
            f"{pcos_probability * 100:.2f}%"
        )

        st.progress(
            pcos_probability
        )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.divider()

    st.header(
        "🔍 Grad-CAM Explainability"
    )


    st.write(
        "The Grad-CAM heatmap highlights image regions "
        "that contributed to the model's prediction."
    )


    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.layer4[-1]


    # --------------------------------------------------------
    # Create Grad-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=[
            target_layer
        ]
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Specify the target class
    #
    # This fixes:
    #
    # TypeError:
    # BaseCAM.__call__() missing 1 required
    # positional argument: 'targets'
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]


    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )


    grayscale_cam = grayscale_cam[
        0
    ]


    # --------------------------------------------------------
    # Prepare original image
    # --------------------------------------------------------

    rgb_image = np.array(
        image.resize(
            (224, 224)
        )
    )


    rgb_image = (
        rgb_image.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # Generate visualization
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # ========================================================
    # DISPLAY GRAD-CAM
    # ========================================================

    col3, col4 = st.columns(2)


    with col3:

        st.subheader(
            "Original Image"
        )

        st.image(
            rgb_image,
            use_container_width=True
        )


    with col4:

        st.subheader(
            "Grad-CAM Heatmap"
        )

        st.image(
            visualization,
            use_container_width=True
        )


    # ========================================================
    # SAVE GRAD-CAM
    # ========================================================

    output_path = os.path.join(
        RESULTS_DIR,
        "gradcam_result.jpg"
    )


    plt.imsave(
        output_path,
        visualization
    )


    st.success(
        f"Grad-CAM result saved successfully."
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.divider()

    st.subheader(
        "Model Information"
    )


    info_col1, info_col2, info_col3 = st.columns(3)


    with info_col1:

        st.write(
            "**Architecture**"
        )

        st.write(
            "ResNet50"
        )


    with info_col2:

        st.write(
            "**Classification**"
        )

        st.write(
            "Binary"
        )


    with info_col3:

        st.write(
            "**Explainability**"
        )

        st.write(
            "Grad-CAM"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PCOS AI Screening Research Prototype"
)