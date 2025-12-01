# CAF Hybrid Light – Image Forgery Detection

CAF Hybrid Light is a lightweight hybrid deep-learning model for image forgery detection.
It integrates both spatial and frequency-domain features and includes a Flask web interface and REST API for predicting whether an image is Real or Fake.

# ✨ Features

Hybrid CNN + Frequency Tokenizer architecture
Fast PyTorch inference
Simple Flask-based web UI
JSON API for automation
Supports JPG, PNG, BMP, TIFF, WebP
Clean and modular codebase

# 📁 Project Structure

CAF_Hybrid_Light/
│
├── app.py                   # Flask app and prediction API
├── model_definition.py      # CAFHybridLight model architecture
├── CAFHybridLight_best.pth  # Trained model weights (not included)
├── templates/               # HTML templates for UI
├── static/                  # Static assets
└── README.md                # Documentation

# 🧠 Model Overview

The CAFHybridLight model combines:

## Frequency Tokenizer
Extracts high-frequency components using a Laplacian kernel.

## Timm Backbone
Used for spatial feature extraction.

## Channel-Attentive Fusion (CAF)
Fuses spatial + frequency features.

## Lightweight Classifier Head

# Output Classes:
0 → Fake
1 → Real

# ⚙️ Installation
## 1. Clone the repository
git clone https://github.com/yourusername/CAF_Hybrid_Light.git
cd CAF_Hybrid_Light

## 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows

## 3. Install dependencies
pip install -r requirements.txt

If a requirements file is not included:
pip install flask pillow torch torchvision timm

## 4. Add your trained model weights

Place your file at the project root:
CAFHybridLight_best.pth

## ▶️ Running the Application
python app.py


## Visit:
http://127.0.0.1:5000

This opens the web interface where you can upload an image for prediction.

## 🌐 API Usage
POST /predict

Send an image file and receive prediction results.

Example (curl):
curl -X POST http://127.0.0.1:5000/predict \
  -F "image=@test.jpg"

Example response:
{
  "prediction": "Fake",
  "confidence": 0.9921
}

# 🖼️ Supported Image Formats

JPG / JPEG
PNG
BMP
TIFF
WebP

# 🛠️ Key Files
app.py
Loads the CAFHybridLight model
Defines / and /predict routes
Handles image preprocessing and inference

model_definition.py
Defines the Frequency Tokenizer
Implements the CAFHybridLight architecture
Builds the final classifier layer

# 🤝 Contributing

Contributions, pull requests, and improvements are welcome.

# 📜 License

This project is licensed under the MIT License.
