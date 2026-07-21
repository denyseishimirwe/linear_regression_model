"""
FastAPI app for predicting student exam scores from study habits and
wellbeing indicators, using the best model trained in Task 1
(Linear Regression via Gradient Descent).
"""

import json
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Student Exam Score Prediction API",
    description=(
        "Predicts a student's exam score from study habits and wellbeing "
        "indicators (sleep, mental health, exercise, diet, etc.), trained "
        "with a Gradient-Descent Linear Regression model."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# Reasoning:
# - allow_origins: we do NOT use "*" (wildcard). Only the Flutter app's
#   known origins are allowed. In a real Flutter mobile app the request
#   does not carry a browser "Origin" header, so this mainly protects the
#   Swagger UI / any web client testing this API from arbitrary websites
#   embedding or calling it. We explicitly list localhost (for local
#   Flutter web testing) and the deployed Render URL of this API itself
#   (for Swagger UI "Try it out" calls). This satisfies the requirement
#   to not generically allow all origins.
# - allow_methods: restricted to only the HTTP verbs this API actually
#   uses (GET for docs/health, POST for prediction and retraining) rather
#   than allowing all methods.
# - allow_headers: restricted to Content-Type and Authorization, the only
#   headers this API expects, rather than allowing all headers.
# - allow_credentials: False, since this API does not use cookies or
#   session-based authentication - it is a stateless prediction endpoint.
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8080",
    "https://student-exam-predictor-ae37.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
MODEL_PATH = "best_model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_ORDER_PATH = "feature_order.json"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(FEATURE_ORDER_PATH) as f:
    FEATURE_ORDER = json.load(f)

DIET_ORDER = {"Poor": 0, "Fair": 1, "Good": 2}
INTERNET_ORDER = {"Poor": 0, "Average": 1, "Good": 2}

# Maps for case-insensitive matching: lowercase input -> correctly-cased value
_DIET_CHOICES = ["Poor", "Fair", "Good"]
_INTERNET_CHOICES = ["Poor", "Average", "Good"]
_GENDER_CHOICES = ["Male", "Female", "Other"]
_YES_NO_CHOICES = ["Yes", "No"]
_PARENTAL_ED_CHOICES = ["High School", "Bachelor", "Master", "Unknown"]


def _normalize_choice(value, choices: list[str], field_name: str) -> str:
    """Matches the input against allowed choices regardless of case
    (e.g. 'yes', 'YES', 'Yes' all resolve to 'Yes'), so the API is
    forgiving of how the client (Flutter app, Swagger UI, curl, etc.)
    capitalizes its input."""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    for choice in choices:
        if value.strip().lower() == choice.lower():
            return choice
    raise ValueError(f"{field_name} must be one of {choices} (case-insensitive), got '{value}'")


# ---------------------------------------------------------------------------
# Request schema (data types + range constraints via Pydantic)
# ---------------------------------------------------------------------------
class StudentInput(BaseModel):
    study_hours_per_day: float = Field(..., ge=0, le=12, description="Hours studied per day")
    social_media_hours: float = Field(..., ge=0, le=12, description="Hours on social media per day")
    netflix_hours: float = Field(..., ge=0, le=12, description="Hours watching Netflix per day")
    attendance_percentage: float = Field(..., ge=0, le=100, description="Class attendance percentage")
    sleep_hours: float = Field(..., ge=0, le=24, description="Hours slept per night")
    exercise_frequency: int = Field(..., ge=0, le=14, description="Times exercised per week")
    mental_health_rating: int = Field(..., ge=1, le=10, description="Self-rated mental health (1-10)")
    diet_quality: Literal["Poor", "Fair", "Good"]
    internet_quality: Literal["Poor", "Average", "Good"]
    gender: Literal["Male", "Female", "Other"]
    part_time_job: Literal["Yes", "No"]
    extracurricular_participation: Literal["Yes", "No"]
    parental_education_level: Literal["High School", "Bachelor", "Master", "Unknown"]

    @field_validator("diet_quality", mode="before")
    @classmethod
    def _v_diet(cls, v):
        return _normalize_choice(v, _DIET_CHOICES, "diet_quality")

    @field_validator("internet_quality", mode="before")
    @classmethod
    def _v_internet(cls, v):
        return _normalize_choice(v, _INTERNET_CHOICES, "internet_quality")

    @field_validator("gender", mode="before")
    @classmethod
    def _v_gender(cls, v):
        return _normalize_choice(v, _GENDER_CHOICES, "gender")

    @field_validator("part_time_job", mode="before")
    @classmethod
    def _v_job(cls, v):
        return _normalize_choice(v, _YES_NO_CHOICES, "part_time_job")

    @field_validator("extracurricular_participation", mode="before")
    @classmethod
    def _v_extra(cls, v):
        return _normalize_choice(v, _YES_NO_CHOICES, "extracurricular_participation")

    @field_validator("parental_education_level", mode="before")
    @classmethod
    def _v_parent_ed(cls, v):
        return _normalize_choice(v, _PARENTAL_ED_CHOICES, "parental_education_level")

    class Config:
        json_schema_extra = {
            "example": {
                "study_hours_per_day": 4.5,
                "social_media_hours": 2.0,
                "netflix_hours": 1.0,
                "attendance_percentage": 85.0,
                "sleep_hours": 7.0,
                "exercise_frequency": 3,
                "mental_health_rating": 7,
                "diet_quality": "Good",
                "internet_quality": "Good",
                "gender": "Female",
                "part_time_job": "No",
                "extracurricular_participation": "Yes",
                "parental_education_level": "Bachelor",
            }
        }


class PredictionResponse(BaseModel):
    predicted_exam_score: float


def build_feature_vector(payload: StudentInput) -> pd.DataFrame:
    """Recreate the same encoding used during training, and return a single
    row DataFrame with columns matching FEATURE_ORDER exactly."""
    row = {
        "study_hours_per_day": payload.study_hours_per_day,
        "social_media_hours": payload.social_media_hours,
        "netflix_hours": payload.netflix_hours,
        "attendance_percentage": payload.attendance_percentage,
        "sleep_hours": payload.sleep_hours,
        "diet_quality": DIET_ORDER[payload.diet_quality],
        "exercise_frequency": payload.exercise_frequency,
        "internet_quality": INTERNET_ORDER[payload.internet_quality],
        "mental_health_rating": payload.mental_health_rating,
        "gender_Male": 1 if payload.gender == "Male" else 0,
        "gender_Other": 1 if payload.gender == "Other" else 0,
        "part_time_job_Yes": 1 if payload.part_time_job == "Yes" else 0,
        "extracurricular_participation_Yes": 1 if payload.extracurricular_participation == "Yes" else 0,
        "parental_education_level_High School": 1 if payload.parental_education_level == "High School" else 0,
        "parental_education_level_Master": 1 if payload.parental_education_level == "Master" else 0,
        "parental_education_level_Unknown": 1 if payload.parental_education_level == "Unknown" else 0,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Student Exam Score Prediction API. Visit /docs for Swagger UI."}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: StudentInput):
    try:
        X = build_feature_vector(payload)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        # exam scores are bounded 0-100 in the real world
        prediction = float(np.clip(prediction, 0, 100))
        return {"predicted_exam_score": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """
    Triggers a retraining of the model using newly uploaded data.
    Expects a CSV with the same raw columns as the original training
    dataset (including 'exam_score' as the target column).
    This is a manual trigger (called via this endpoint / Swagger UI)
    rather than a fully automated pipeline.
    """
    global model, scaler

    try:
        new_df = pd.read_csv(file.file)

        required_cols = {
            "study_hours_per_day", "social_media_hours", "netflix_hours",
            "attendance_percentage", "sleep_hours", "diet_quality",
            "exercise_frequency", "internet_quality", "mental_health_rating",
            "gender", "part_time_job", "extracurricular_participation",
            "parental_education_level", "exam_score",
        }
        missing = required_cols - set(new_df.columns)
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

        # Same preprocessing pipeline as training
        new_df["parental_education_level"] = new_df["parental_education_level"].fillna("Unknown")
        new_df["diet_quality"] = new_df["diet_quality"].map(DIET_ORDER)
        new_df["internet_quality"] = new_df["internet_quality"].map(INTERNET_ORDER)
        new_df = pd.get_dummies(
            new_df,
            columns=["gender", "part_time_job", "extracurricular_participation", "parental_education_level"],
            drop_first=True,
        )

        # Ensure all expected columns exist (some categories may be absent in new data)
        for col in FEATURE_ORDER:
            if col not in new_df.columns:
                new_df[col] = 0

        X_new = new_df[FEATURE_ORDER]
        y_new = new_df["exam_score"]

        X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

        new_scaler = StandardScaler()
        X_train_scaled = new_scaler.fit_transform(X_train)
        X_test_scaled = new_scaler.transform(X_test)

        new_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
        new_model.fit(X_train_scaled, y_train)

        preds = new_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Persist the retrained model so it's used for future /predict calls
        joblib.dump(new_model, MODEL_PATH)
        joblib.dump(new_scaler, SCALER_PATH)
        model = new_model
        scaler = new_scaler

        return {
            "message": "Model retrained successfully.",
            "rows_used": len(new_df),
            "test_mse": mse,
            "test_r2": r2,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
