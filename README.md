<<<<<<< HEAD
# 🛡️ DeepGuard AI — Fake Image Detection System & Web Application

An AI-powered fake image detection system that classifies images as **REAL** or **FAKE (AI-Generated)** using a fine-tuned ResNet-50 deep neural network. The project features a modern **React web application**, a **Python Flask REST API**, and Explainable AI (XAI) forensic modules including **Error Level Analysis (ELA)**, **Frequency Analysis (FFT)**, and **Grad-CAM Attention Maps**.
=======
# 🖼️ Fake Image Detection using Deep Learning (ResNet-50)

An AI-powered fake image detection system that classifies images as **Real** or **AI-Generated/Fake** using a fine-tuned **ResNet-50** convolutional neural network. The project also provides **Explainable AI (XAI)** visualizations including **Error Level Analysis (ELA)**, **Frequency Analysis (FFT)**, and **Grad-CAM** to help users understand why the model made its prediction.
>>>>>>> 658e5e7ea7de9bb64267c7bfe4117b63dbaaf8ed

---

## 📌 Features

<<<<<<< HEAD
- 🎯 **High Accuracy**: 97.17% validation accuracy with fine-tuned ResNet-50 architecture.
- ⚡ **Interactive React Frontend**: Modern dark theme web app with glassmorphism aesthetics and micro-animations.
- 🔍 **Explainable AI (XAI) Suite**:
  - **Error Level Analysis (ELA)**: Exposes digital splices and compression anomalies.
  - **Frequency Spectrum (FFT)**: Highlights high-frequency periodicities common in synthetic renders.
  - **Grad-CAM Attention Maps**: Highlights exact visual regions influencing predictions.
- ⚙️ **Password-Protected Training**: Admin settings panel protected by password (`sk7411`) allowing on-demand model retraining.
- 🔬 **Legacy MATLAB Support**: Preserved MATLAB scripts (`DemoTest.m`, `FreshTraining_4Epochs.m`, `showExtraFeatures.m`) and trained `.mat` model.

---

## 🛠️ Tech Stack

- **Frontend**: React (Vite), JavaScript, Vanilla CSS (Glassmorphism design tokens)
- **Backend API**: Python, Flask, PyTorch, Torchvision, Pillow, NumPy, SciPy, Matplotlib
- **Legacy Engine**: MATLAB, Deep Learning Toolbox, Image Processing Toolbox
=======
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
>>>>>>> 658e5e7ea7de9bb64267c7bfe4117b63dbaaf8ed

---

## 📂 Project Structure

```
<<<<<<< HEAD
fake image project/
├── backend/                    # Python Flask API & PyTorch Model
│   ├── app.py                  # API Endpoints (/predict, /train, /status)
│   ├── model.py                # ResNet-50 PyTorch architecture & loader
│   ├── analysis.py             # ELA, FFT, Grad-CAM XAI functions
│   ├── train.py                # Training engine matching MATLAB parameters
│   ├── config.py               # Admin password (sk7411) & hyperparameters
│   ├── requirements.txt        # Python dependencies
│   └── Procfile                # Deployment configuration
├── frontend/                   # React Frontend (Vite)
│   ├── src/
│   │   ├── App.jsx             # Main React Single Page App
│   │   ├── index.css           # Custom CSS Design System
│   │   └── main.jsx
│   ├── package.json
│   └── vercel.json             # Vercel deployment setup
├── dataset/                    # Image Dataset (REAL & FAKE)
│   ├── train/
│   └── test/
├── DemoTest.m                  # Legacy MATLAB Demo Script
├── FreshTraining_4Epochs.m     # Legacy MATLAB Training Script
├── showExtraFeatures.m         # Legacy MATLAB Explainable AI Script
└── myFakeDetector_4Epochs_Final.mat # Legacy Trained MATLAB Model
=======
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
>>>>>>> 658e5e7ea7de9bb64267c7bfe4117b63dbaaf8ed
```

---

<<<<<<< HEAD
## 🚀 Running Locally

### 1. Start Python Backend API

```bash
cd backend
pip install -r requirements.txt
python app.py
```
*The server will run on `http://localhost:5000`.*

### 2. Start React Frontend

```bash
cd frontend
npm install
npm run dev
```
*Open your browser at `http://localhost:5173`.*

---

## 🔐 Admin Model Training

To retrain the model directly from the web interface:
1. Click the **⚙️ Settings** icon in the navbar.
2. Enter the admin password: `sk7411`.
3. Click **Start 4-Epoch Model Training**.
4. Monitor live progress, loss, validation accuracy, and execution logs in real time.

---

## 🌐 Deployment Instructions

### Backend (Render / Railway / Heroku)
1. Deploy the `backend/` directory to Render or Railway.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app`

### Frontend (Vercel / Netlify)
1. Deploy the `frontend/` directory to Vercel.
2. Framework Preset: **Vite**.
3. Environment Variables / Rewrites: Configure `API_BASE` to point to your deployed backend URL.
=======
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
>>>>>>> 658e5e7ea7de9bb64267c7bfe4117b63dbaaf8ed

---

## 👨‍💻 Author

<<<<<<< HEAD
**Suhas S Kattimani**  
B.Tech – Information Science and Engineering  
Presidency University  
GitHub: [@SuhasReturn](https://github.com/SuhasReturn)
=======
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
>>>>>>> 658e5e7ea7de9bb64267c7bfe4117b63dbaaf8ed
