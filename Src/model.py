import torch.nn as nn
from torchvision import models


def create_model():
    # Load pretrained ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Number of features entering the final layer
    num_features = model.fc.in_features

    # Replace final layer for 2 classes:
    # 0 = Normal
    # 1 = PCOS
    model.fc = nn.Linear(num_features, 2)

    return model


if __name__ == "__main__":
    model = create_model()

    print("Model created successfully!")
    print("Number of classes:", 2)
    print(model.fc)