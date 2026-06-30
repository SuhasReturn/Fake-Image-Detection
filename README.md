# 🖼️ Fake Image Detection using Deep Learning (ResNet-50)

An AI-powered fake image detection system that classifies images as **Real** or **AI-Generated/Fake** using a fine-tuned **ResNet-50** convolutional neural network. The project also provides **Explainable AI (XAI)** visualizations including **Error Level Analysis (ELA)**, **Frequency Analysis (FFT)**, and **Grad-CAM** to help users understand why the model made its prediction.

---

## 📌 Features

* Detects whether an image is **Real** or **Fake (AI-Generated)**
* Fine-tuned **ResNet-50** deep learning model
* Achieves **97.17% validation accuracy**
* Image preprocessing and augmentation
* Explainable AI visualization:

  * Error Level Analysis (ELA)
  * Frequency Domain Analysis (FFT)
  * Grad-CAM Attention Maps
* Interactive MATLAB demo for testing custom images

---

## 🛠️ Technologies Used

* MATLAB
* Deep Learning Toolbox
* ResNet-50 (Transfer Learning)
* Image Processing Toolbox
* Computer Vision Toolbox

---

## 📂 Project Structure

```
Fake-Image-Detection/
│
├── dataset/
│   ├── REAL/
│   └── FAKE/
│
├── FreshTraining_4Epochs.m
├── DemoTest.m
├── showExtraFeatures.m
├── myFakeDetector_4Epochs_Final.mat
├── README.md
└── LICENSE
```

---

## 🧠 Model Architecture

* Base Model: **ResNet-50**
* Transfer Learning
* Final Fully Connected Layer modified for **2 classes**

  * REAL
  * FAKE
* Optimizer: SGD with Momentum
* Learning Rate Scheduling
* Data Augmentation
* Automatic Checkpoint Saving

---

## 📊 Dataset

The dataset should follow this directory structure:

```
dataset/
│
├── REAL/
│     image1.jpg
│     image2.jpg
│
└── FAKE/
      image1.jpg
      image2.jpg
```

The dataset is automatically divided into:

* 70% Training
* 15% Validation
* 15% Testing

---

## 🚀 Training

Run:

```matlab
FreshTraining_4Epochs
```

The training process:

* Loads dataset
* Applies image augmentation
* Fine-tunes ResNet-50
* Saves checkpoints
* Stores the final trained model as:

```
myFakeDetector_4Epochs_Final.mat
```

---

## ▶️ Running the Demo

Run:

```matlab
DemoTest
```

You can choose:

* Test Image from Dataset
* Personal Image

The system displays:

* Prediction (Real/Fake)
* Confidence Score
* Overall Detection Accuracy

---

## 🔍 Explainable AI

After prediction, the application can display:

### Error Level Analysis (ELA)

Highlights compression artifacts that often indicate manipulated images.

### Frequency Analysis (FFT)

Shows frequency-domain inconsistencies in images.

### Grad-CAM

Visualizes the regions of the image that influenced the model's prediction.

---

## 📈 Performance

| Metric              |      Value |
| ------------------- | ---------: |
| Model               |  ResNet-50 |
| Classes             |          2 |
| Validation Accuracy | **97.17%** |
| Image Size          |  224 × 224 |
| Epochs              |          4 |

---

## 💡 Future Improvements

* Support for multiple AI image generators
* Web-based interface
* Mobile application
* Real-time image detection
* Detection of deepfakes and manipulated videos
* Multi-class fake image classification

---

## 👨‍💻 Author

**Suhas S Kattimani**

B.Tech – Information Science and Engineering

Presidency University

---

## ⭐ Acknowledgements

* MATLAB Deep Learning Toolbox
* ResNet-50 Pre-trained Network
* MATLAB Image Processing Toolbox

---

## 📜 License

This project is intended for educational and research purposes.

---

### ⭐ If you found this project useful, consider giving it a star on GitHub!
