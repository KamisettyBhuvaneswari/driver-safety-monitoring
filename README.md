# driver-safety-monitoring
Real-time driver safety monitoring using facial emotion recognition and drowsiness detection

# Real-Time Driver Safety Monitoring Using Facial Emotion Recognition

MSc Applied Artificial Intelligence — Computing Research Project

## Overview
Real-time driver safety monitoring system combining facial emotion recognition
and eye-closure-based drowsiness detection using MobileNetV2 transfer learning
and MediaPipe-based real-time inference.

## Models
- **Emotion Classifier** (FER-2013): 53.08% test accuracy, 7 emotion classes
- **Drowsiness Detector** (MRL Eye Dataset): 79.12% test accuracy, binary classification

## Files
- `demo.py` — real-time webcam inference script
- `emotion_model_best.h5` / `drowsiness_model_best.h5` — trained models
- `emotion_class_indices.json` / `eye_class_indices.json` — label mappings
- `*_confusion_matrix.png`, `*_training_curves.png` — evaluation results
- `driver_safety_monitoring.ipynb` — full training notebook (Kaggle, P100 GPU)

## Setup
```bash
pip install tensorflow opencv-python mediapipe==0.10.9
python demo.py
```

## Datasets
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [MRL Eye Dataset](https://www.kaggle.com/datasets/tauilabdelilah/mrl-eye-dataset)
