"""
Flask API Server for Fake Image Detection.
Serves the React frontend and provides prediction/training APIs.
"""
import os
import io
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import base64

from config import (
    HOST, PORT, DEBUG, TRAINING_PASSWORD,
    TRAIN_DIR, TEST_DIR, CLASS_NAMES, MODEL_PATH
)
from model import load_model, predict_image, get_model_info
from analysis import full_analysis, perform_ela, perform_fft, perform_gradcam, image_to_base64
from train import train_model, training_state

app = Flask(__name__, static_folder=None)
CORS(app)

# Global model instance
current_model = None


def get_or_load_model():
    """Lazy-load the model on first request."""
    global current_model
    if current_model is None:
        current_model = load_model()
    return current_model


# ========================
# API Routes
# ========================

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Upload image and get prediction + analysis."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream)
        if image.mode != "RGB":
            image = image.convert("RGB")

        model = get_or_load_model()
        if model is None:
            return jsonify({
                "error": "Model not trained yet. Please train the model first from Settings.",
                "model_exists": False,
            }), 503

        # Get prediction
        prediction = predict_image(model, image)

        # Get analysis
        analysis = full_analysis(model, image)

        # Get original image as base64 for display
        original_b64 = image_to_base64(image.resize((224, 224)))

        return jsonify({
            "prediction": prediction,
            "analysis": analysis,
            "original_image": original_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start model training (password-protected)."""
    data = request.get_json()

    if not data or data.get("password") != TRAINING_PASSWORD:
        return jsonify({"error": "Invalid password"}), 403

    if training_state["is_training"]:
        return jsonify({"error": "Training already in progress"}), 409

    # Start training in background thread
    def run_training():
        global current_model
        result = train_model()
        if result.get("success"):
            # Reload model after training
            current_model = load_model()

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return jsonify({"message": "Training started", "status": "started"})


@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    """Get current training progress."""
    return jsonify(training_state)


@app.route("/api/model/info", methods=["GET"])
def api_model_info():
    """Get model metadata."""
    info = get_model_info()
    return jsonify(info)


@app.route("/api/dataset/stats", methods=["GET"])
def api_dataset_stats():
    """Get dataset statistics."""
    stats = {"train": {}, "test": {}}

    # Count training images
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats["train"][class_name] = count

    # Count test images
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TEST_DIR, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            stats["test"][class_name] = count

    stats["total_train"] = sum(stats["train"].values())
    stats["total_test"] = sum(stats["test"].values())
    stats["total"] = stats["total_train"] + stats["total_test"]

    return jsonify(stats)


@app.route("/api/dataset/samples", methods=["GET"])
def api_dataset_samples():
    """Get random sample images from the dataset for showcase."""
    import random

    samples = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            selected = random.sample(files, min(3, len(files)))
            for fname in selected:
                try:
                    img_path = os.path.join(class_dir, fname)
                    img = Image.open(img_path).convert("RGB").resize((224, 224))
                    samples.append({
                        "image": image_to_base64(img),
                        "label": class_name,
                        "filename": fname,
                    })
                except Exception:
                    continue

    return jsonify(samples)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint for deployment."""
    return jsonify({
        "status": "healthy",
        "model_loaded": current_model is not None,
        "model_exists": os.path.exists(MODEL_PATH),
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  Fake Image Detection API Server")
    print("=" * 50)
    print(f"  Server: http://{HOST}:{PORT}")
    print(f"  Model exists: {os.path.exists(MODEL_PATH)}")
    print("=" * 50)

    # Pre-load model if available
    model = get_or_load_model()
    if model:
        print("✅ Model loaded successfully.")
    else:
        print("⚠️  No trained model found. Train via Settings panel.")

    app.run(host=HOST, port=PORT, debug=DEBUG)
