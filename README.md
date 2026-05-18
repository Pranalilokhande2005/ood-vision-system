# Intelligent Vision System with OOD Detection

A production-grade Out-of-Distribution detection system built on ResNet-50.

## Results
| Detector | OOD Set | AUROC |
|---|---|---|
| KNN (k=50) | SVHN | **97.95%** |
| KNN (k=50) | CIFAR-100 | **87.38%** |

## Model
- Backbone: ResNet-50 fine-tuned on CIFAR-10
- OOD detectors: Energy + KNN + MC Dropout
- Calibration: Temperature scaling

## Project Structure
- model/ — trained weights and config
- frontend/ — HTML web UI
- api/ — FastAPI server

## Run locally
Install dependencies then run:
    pip install -r requirements.txt
    python api/app.py

Open frontend/index.html in browser and set API URL to http://localhost:8000

## Model Weights
Model is too large for GitHub. Download from Google Drive and place in model/ folder.

## Classes
airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck