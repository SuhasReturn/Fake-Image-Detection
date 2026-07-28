"""
Configuration for Fake Image Detection Backend
Mirrors the MATLAB training parameters exactly.
"""
import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Model settings
MODEL_PATH = os.path.join(MODEL_DIR, "fake_detector_resnet50.pth")
NUM_CLASSES = 2
CLASS_NAMES = ["FAKE", "REAL"]
INPUT_SIZE = 224

# Training hyperparameters (matching MATLAB FreshTraining_4Epochs.m)
TRAINING_CONFIG = {
    "max_epochs": 4,
    "batch_size": 8,
    "initial_lr": 0.001,
    "lr_schedule": "piecewise",
    "lr_drop_factor": 0.1,
    "lr_drop_period": 2,
    "optimizer": "sgdm",
    "momentum": 0.9,
    "weight_decay": 1e-4,
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "fc_lr_multiplier": 10,
}

# Security
TRAINING_PASSWORD = "sk7411"

# Server
HOST = "0.0.0.0"
PORT = 5000
DEBUG = True

# ELA settings (matching MATLAB showExtraFeatures.m)
ELA_QUALITY = 90
ELA_SCALE = 12
