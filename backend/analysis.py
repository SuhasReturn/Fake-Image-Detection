"""
Explainable AI Analysis Module.
Mirrors MATLAB showExtraFeatures.m:
  - Error Level Analysis (ELA)
  - Frequency Analysis (FFT)
  - Grad-CAM Attention Maps
"""
import io
import base64
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from config import INPUT_SIZE, ELA_QUALITY, ELA_SCALE


def image_to_base64(pil_image):
    """Convert PIL Image to base64 string for frontend display."""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def numpy_to_base64(arr):
    """Convert numpy array to base64 PNG."""
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    return image_to_base64(img)


def perform_ela(image):
    """
    Error Level Analysis (ELA).
    Mirrors MATLAB:
        imwrite(testImage, tempfile, 'jpeg', 'Quality', 90);
        compressed = imread(tempfile);
        ela = uint8(abs(double(testImage) - double(compressed)) * 12);
    """
    try:
        # Resize to match model input
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save at reduced quality and reload
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=ELA_QUALITY)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Compute ELA: abs(original - compressed) * scale
        orig_arr = np.array(image, dtype=np.float64)
        comp_arr = np.array(compressed, dtype=np.float64)
        ela_arr = np.abs(orig_arr - comp_arr) * ELA_SCALE

        ela_arr = np.clip(ela_arr, 0, 255).astype(np.uint8)

        return {
            "image": numpy_to_base64(ela_arr),
            "description": "Bright areas indicate potential manipulation. "
                           "Uniform brightness suggests authenticity.",
        }
    except Exception as e:
        return {"image": None, "error": str(e)}


def perform_fft(image):
    """
    Frequency Domain Analysis (FFT).
    Mirrors MATLAB:
        gray = rgb2gray(testImage);
        F = fftshift(fft2(gray));
        fftMag = log(abs(F) + 1);
    """
    try:
        # Resize and convert to grayscale
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        if image.mode != "L":
            gray = image.convert("L")
        else:
            gray = image

        gray_arr = np.array(gray, dtype=np.float64)

        # FFT analysis matching MATLAB
        f_transform = np.fft.fft2(gray_arr)
        f_shifted = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shifted) + 1)

        # Normalize to 0-255 for display
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255
        magnitude = magnitude.astype(np.uint8)

        # Apply colormap for better visualization
        from PIL import ImageOps
        fft_img = Image.fromarray(magnitude, mode="L")

        return {
            "image": image_to_base64(fft_img),
            "description": "Shows frequency-domain patterns. AI-generated images "
                           "often lack high-frequency natural noise.",
        }
    except Exception as e:
        return {"image": None, "error": str(e)}


def perform_gradcam(model, image, predicted_class_idx=None):
    """
    Grad-CAM visualization.
    Mirrors MATLAB:
        scoreMap = gradCAM(trainedNet, testImage, predictedLabel,
            'FeatureLayer', 'activation_49_relu');
    Uses the last conv layer of ResNet-50 (layer4[-1]).
    """
    try:
        device = next(model.parameters()).device

        # Prepare image
        image_resized = image.resize((INPUT_SIZE, INPUT_SIZE))
        if image_resized.mode != "RGB":
            image_resized = image_resized.convert("RGB")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        input_tensor = transform(image_resized).unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)

        # Hook into the last convolutional layer (equivalent to activation_49_relu)
        gradients = []
        activations = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        target_layer = model.layer4[-1]
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        # Forward pass
        output = model(input_tensor)
        if predicted_class_idx is None:
            predicted_class_idx = output.argmax(dim=1).item()

        # Backward pass for the predicted class
        model.zero_grad()
        output[0, predicted_class_idx].backward()

        # Compute Grad-CAM
        grads = gradients[0].cpu().data.numpy()[0]
        acts = activations[0].cpu().data.numpy()[0]

        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to image size
        cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
            (INPUT_SIZE, INPUT_SIZE), Image.BILINEAR
        ))

        # Create heatmap overlay (matching MATLAB jet colormap with alpha=0.65)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        heatmap = cm.jet(cam_resized / 255.0)[:, :, :3]
        heatmap = (heatmap * 255).astype(np.uint8)

        # Overlay on original image
        orig_arr = np.array(image_resized)
        overlay = (orig_arr * 0.35 + heatmap * 0.65).astype(np.uint8)

        # Clean up hooks
        fh.remove()
        bh.remove()

        return {
            "image": numpy_to_base64(overlay),
            "description": "Red/warm regions show where the model focused its attention. "
                           "These areas most influenced the prediction.",
        }
    except Exception as e:
        return {"image": None, "error": str(e)}


def full_analysis(model, image):
    """Run all analyses and return combined results."""
    results = {
        "ela": perform_ela(image),
        "fft": perform_fft(image),
    }

    if model is not None:
        results["gradcam"] = perform_gradcam(model, image)
    else:
        results["gradcam"] = {
            "image": None,
            "error": "Model not loaded. Train the model first.",
        }

    return results
