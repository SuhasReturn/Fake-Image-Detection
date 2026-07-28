"""
ResNet-50 Model for Fake Image Detection.
Mirrors the MATLAB architecture: resnet50 with replaced FC layer for 2 classes.
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
from config import MODEL_PATH, INPUT_SIZE, CLASS_NAMES, NUM_CLASSES


def get_transforms(augment=False):
    """Get image transforms. Augmentation mirrors MATLAB's imageDataAugmenter."""
    if augment:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),  # RandXReflection
            transforms.RandomRotation(20),       # RandRotation [-20, 20]
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_model(pretrained=True):
    """
    Create ResNet-50 model with modified final layer.
    Mirrors MATLAB:
        lgraph = replaceLayer(lgraph, 'fc1000',
            fullyConnectedLayer(2, 'Name', 'newFC',
            'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

    # Replace the final fully connected layer for 2-class classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    # Initialize the new FC layer
    nn.init.xavier_uniform_(model.fc.weight)
    nn.init.zeros_(model.fc.bias)

    return model


def load_model():
    """Load trained model from disk. Returns None if no model found."""
    if not os.path.exists(MODEL_PATH):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(pretrained=False)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_image(model, image):
    """
    Predict if an image is REAL or FAKE.
    
    Args:
        model: Trained PyTorch model
        image: PIL Image
    
    Returns:
        dict with prediction, confidence, and class scores
    """
    device = next(model.parameters()).device
    transform = get_transforms(augment=False)

    # Handle grayscale images
    if image.mode != "RGB":
        image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    confidence_pct = confidence.item() * 100

    scores = {CLASS_NAMES[i]: probabilities[0][i].item() * 100
              for i in range(NUM_CLASSES)}

    return {
        "prediction": predicted_class,
        "confidence": round(confidence_pct, 2),
        "scores": scores,
        "is_fake": predicted_class == "FAKE",
    }


def get_model_info():
    """Get model metadata."""
    info = {
        "architecture": "ResNet-50",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "model_exists": os.path.exists(MODEL_PATH),
    }

    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        info["accuracy"] = checkpoint.get("accuracy", None)
        info["epochs_trained"] = checkpoint.get("epoch", None)
        info["trained_date"] = checkpoint.get("trained_date", None)

    return info
