# Student Exam Score Predictor

## Mission

This project investigates how student wellbeing (sleep, mental health, exercise, diet) and study habits relate to academic performance. The goal is to predict a student's exam score from their daily habits, to help identify students who may benefit from additional academic or wellbeing support.

## Dataset

**Student Habits vs Academic Performance** — 1,000 student records with 16 features covering study habits (study hours, attendance), wellbeing (sleep, mental health, exercise, diet), and background (gender, parental education). Sourced from Kaggle: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance

## Live Links

- **API (Swagger UI):** https://student-exam-predictor-ae37.onrender.com/docs
- **Video Demo:** _[YouTube link to be added]_

> Note: the API is hosted on Render's free tier, which spins down after inactivity. The first request after idle time may take 30-60 seconds to respond while the server wakes up.

## Repository Structure

```
linear_regression_model/
├── summative/
│   ├── linear_regression/
│   │   ├── multivariate.ipynb
│   │   ├── student_habits_performance.csv
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_order.json
│   ├── API/
│   │   ├── prediction.py
│   │   ├── requirements.txt
│   │   ├── runtime.txt
│   │   ├── best_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_order.json
│   └── FlutterApp/
│       ├── pubspec.yaml
│       ├── pubspec.lock
│       ├── analysis_options.yaml
│       ├── lib/
│       │   └── main.dart
│       ├── test/
│       │   └── widget_test.dart
│       ├── android/
│       ├── ios/
│       ├── linux/
│       ├── macos/
│       ├── windows/
│       └── web/
├── pyproject.toml
└── uv.lock
```

## Models Compared

Four regression approaches were trained and compared on the same 80/20 train/test split:

| Model | MSE | R² |
|---|---|---|
| **Linear Regression — Batch Gradient Descent** (best) | 26.00 | 0.899 |
| Linear Regression — Stochastic Gradient Descent | 26.14 | 0.898 |
| Random Forest (ensemble) | 37.78 | 0.853 |
| Decision Tree | 85.66 | 0.666 |

Both gradient-descent-based linear regression models substantially outperform the tree-based models, since `study_hours_per_day` has a strong linear relationship with `exam_score` (correlation ≈ 0.83). Batch Gradient Descent was selected as the best-performing model and is the one deployed in the API.

## API

The FastAPI app (`summative/API/prediction.py`) exposes:

- **`POST /predict`** — accepts 13 student attributes (validated with Pydantic — enforced types and range constraints) and returns a predicted exam score.
- **`POST /retrain`** — accepts an uploaded CSV of new student data and retrains the model on the spot, replacing the deployed model with the freshly trained one.

**CORS configuration:** origins, methods, and headers are explicitly restricted rather than using a wildcard (`*`). Only known client origins (localhost for local testing, the deployed Render URL for Swagger UI) are allowed; only `GET`/`POST` methods are permitted since those are the only ones the API uses; only `Content-Type`/`Authorization` headers are allowed; credentials are disabled since the API is stateless and does not use cookies or sessions.

## How to Run

### Set up the environment (uv)
```bash
uv sync
```

### Run the notebook
```bash
uv run jupyter notebook summative/linear_regression/multivariate.ipynb
```

### Run the API locally
```bash
cd summative/API
uv run uvicorn prediction:app --reload
```
Then visit `http://127.0.0.1:8000/docs` for the local Swagger UI.

### Run the mobile app
1. Install the [Flutter SDK](https://docs.flutter.dev/get-started/install) and the Flutter/Dart extensions for your editor.
2. Navigate to the app folder and install dependencies:
   ```bash
   cd summative/FlutterApp
   flutter pub get
   ```
3. Connect a device or start an emulator, then run:
   ```bash
   flutter run
   ```
4. The app is a single page with 13 input fields (study habits, wellbeing, and background), a **Predict Score** button, and a result area that displays the predicted exam score or a validation error message. It is already configured to call the live Render API — no additional setup is required to get a prediction.
