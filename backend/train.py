"""
Training script for Fake Image Detection model.
Replicates MATLAB FreshTraining_4Epochs.m exactly.
Called via the Settings panel in the web UI (password-protected).
"""
import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import numpy as np

from config import (
    TRAIN_DIR, MODEL_PATH, CHECKPOINT_DIR, MODEL_DIR,
    TRAINING_CONFIG, INPUT_SIZE, CLASS_NAMES
)
from model import create_model, get_transforms


# Global training state (shared with Flask app via import)
training_state = {
    "is_training": False,
    "progress": 0,
    "epoch": 0,
    "total_epochs": TRAINING_CONFIG["max_epochs"],
    "batch": 0,
    "total_batches": 0,
    "loss": 0.0,
    "accuracy": 0.0,
    "val_accuracy": 0.0,
    "status": "idle",
    "log": [],
}


def log_message(msg):
    """Add message to training log."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    training_state["log"].append(entry)
    print(entry)


def train_model(progress_callback=None):
    """
    Train the ResNet-50 model.
    Mirrors FreshTraining_4Epochs.m:
      - ResNet-50 with pretrained ImageNet weights
      - SGD with momentum 0.9
      - LR=0.001, piecewise drop 0.1x every 2 epochs
      - Data augmentation: random horizontal flip, ±20° rotation
      - 4 epochs, batch size 8
    """
    global training_state

    if training_state["is_training"]:
        return {"error": "Training already in progress"}

    training_state["is_training"] = True
    training_state["progress"] = 0
    training_state["status"] = "initializing"
    training_state["log"] = []

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_message(f"=== Fresh Training - {TRAINING_CONFIG['max_epochs']} Epochs ===")
        log_message(f"Device: {device}")

        # 1. Load Dataset
        log_message("Loading dataset...")
        train_transform = get_transforms(augment=True)
        val_transform = get_transforms(augment=False)

        full_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
        log_message(f"Total images found: {len(full_dataset)}")
        log_message(f"Classes: {full_dataset.classes}")

        # Split: 70% train, 15% val, 15% test (matching MATLAB)
        total = len(full_dataset)
        train_size = int(total * TRAINING_CONFIG["train_split"])
        val_size = int(total * TRAINING_CONFIG["val_split"])
        test_size = total - train_size - val_size

        indices = list(range(total))
        np.random.seed(42)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        train_subset = Subset(full_dataset, train_indices)
        # Validation uses non-augmented transforms
        val_dataset = datasets.ImageFolder(TRAIN_DIR, transform=val_transform)
        val_subset = Subset(val_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=TRAINING_CONFIG["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        log_message(f"Training images: {len(train_subset)}")
        log_message(f"Validation images: {len(val_subset)}")

        # 2. Build Model
        log_message("Creating fresh ResNet-50 model...")
        model = create_model(pretrained=True)
        model.to(device)
        log_message("✅ Fresh model created.")

        # 3. Set up optimizer (matching MATLAB SGD with momentum)
        # Higher LR for the new FC layer (10x multiplier as in MATLAB)
        fc_params = list(model.fc.parameters())
        other_params = [p for name, p in model.named_parameters()
                        if "fc" not in name]

        optimizer = optim.SGD([
            {"params": other_params, "lr": TRAINING_CONFIG["initial_lr"]},
            {"params": fc_params, "lr": TRAINING_CONFIG["initial_lr"] * TRAINING_CONFIG["fc_lr_multiplier"]},
        ], momentum=TRAINING_CONFIG["momentum"], weight_decay=TRAINING_CONFIG["weight_decay"])

        # LR scheduler: piecewise drop (matching MATLAB)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=TRAINING_CONFIG["lr_drop_period"],
            gamma=TRAINING_CONFIG["lr_drop_factor"]
        )

        criterion = nn.CrossEntropyLoss()

        # 4. Training Loop
        total_batches = len(train_loader)
        training_state["total_batches"] = total_batches
        training_state["total_epochs"] = TRAINING_CONFIG["max_epochs"]

        log_message(f"🚀 Starting training for {TRAINING_CONFIG['max_epochs']} epochs...")
        log_message(f"Batch size: {TRAINING_CONFIG['batch_size']}, Total batches per epoch: {total_batches}")

        best_accuracy = 0.0

        for epoch in range(TRAINING_CONFIG["max_epochs"]):
            training_state["epoch"] = epoch + 1
            training_state["status"] = f"training_epoch_{epoch + 1}"

            model.train()
            running_loss = 0.0
            correct = 0
            total_samples = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total_samples += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update progress
                batch_progress = (epoch * total_batches + batch_idx + 1) / (TRAINING_CONFIG["max_epochs"] * total_batches) * 100
                training_state["batch"] = batch_idx + 1
                training_state["progress"] = round(batch_progress, 1)
                training_state["loss"] = round(running_loss / (batch_idx + 1), 4)
                training_state["accuracy"] = round(100. * correct / total_samples, 2)

                if (batch_idx + 1) % 50 == 0:
                    log_message(
                        f"Epoch {epoch+1}/{TRAINING_CONFIG['max_epochs']} | "
                        f"Batch {batch_idx+1}/{total_batches} | "
                        f"Loss: {training_state['loss']:.4f} | "
                        f"Acc: {training_state['accuracy']:.2f}%"
                    )

            epoch_acc = 100. * correct / total_samples
            epoch_loss = running_loss / total_batches

            # Validation
            log_message(f"Running validation for epoch {epoch + 1}...")
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_accuracy = 100. * val_correct / val_total if val_total > 0 else 0
            training_state["val_accuracy"] = round(val_accuracy, 2)

            log_message(
                f"Epoch {epoch+1} Complete | "
                f"Train Acc: {epoch_acc:.2f}% | "
                f"Val Acc: {val_accuracy:.2f}% | "
                f"Loss: {epoch_loss:.4f}"
            )

            # Save checkpoint
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": val_accuracy,
                "loss": epoch_loss,
            }, checkpoint_path)
            log_message(f"✅ Checkpoint saved: epoch {epoch + 1}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            scheduler.step()

        # 5. Save Final Model
        training_state["status"] = "saving"
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "accuracy": best_accuracy,
            "epoch": TRAINING_CONFIG["max_epochs"],
            "class_names": CLASS_NAMES,
            "trained_date": datetime.datetime.now().isoformat(),
        }, MODEL_PATH)

        log_message(f"🎉 Training completed successfully!")
        log_message(f"Final model saved. Best Validation Accuracy: {best_accuracy:.2f}%")

        training_state["status"] = "completed"
        training_state["progress"] = 100
        training_state["val_accuracy"] = best_accuracy

        return {
            "success": True,
            "accuracy": best_accuracy,
            "epochs": TRAINING_CONFIG["max_epochs"],
        }

    except Exception as e:
        log_message(f"❌ Training failed: {str(e)}")
        training_state["status"] = "error"
        training_state["is_training"] = False
        return {"error": str(e)}

    finally:
        training_state["is_training"] = False


if __name__ == "__main__":
    train_model()
