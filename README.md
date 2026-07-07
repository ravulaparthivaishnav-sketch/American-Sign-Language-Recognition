# Static ASL Letter Recognition

A real-time American Sign Language (ASL) static letter recognition system built with Python, OpenCV, MediaPipe, and a machine learning classifier.

## Features

- Real-time hand tracking using MediaPipe
- Detects and classifies static ASL alphabet letters
- Live webcam predictions
- Trained on custom hand landmark data
- Lightweight and runs locally

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Joblib

## How It Works

1. Capture webcam frames.
2. Detect hand landmarks with MediaPipe.
3. Convert landmarks into feature vectors.
4. Feed the features into the trained machine learning model.
5. Display the predicted ASL letter in real time.

## Files

- `decoderAi.py` — Runs real-time ASL prediction
- `model.py` — Trains the classifier
- `dataset.csv` — Training dataset
- `asl_model.pkl` — Trained model
- `label_encoder.pkl` — Label encoder
- `hand_landmarker.task` — MediaPipe hand tracking model

## Future Improvements

- Dynamic gesture recognition (J, Z, words)
- Larger training dataset
- Confidence scores
- Sentence generation
