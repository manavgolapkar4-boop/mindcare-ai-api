
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI(
    title="MindCare AI Recommendation API",
    description="Prototype for personalized cognitive game difficulty",
    version="1.0.0",
)

# Synthetic example data for a student project prototype.
# Features: score percentage, accuracy percentage,
# average response time in seconds.
training_data = np.array([
    [20, 25, 15],
    [30, 35, 12],
    [40, 40, 10],
    [45, 50, 9],
    [50, 55, 8],
    [55, 60, 7],
    [60, 65, 6],
    [65, 70, 5],
    [70, 75, 5],
    [75, 80, 4],
    [80, 85, 3],
    [90, 95, 2],
    [35, 30, 14],
    [48, 45, 11],
    [58, 55, 9],
    [68, 65, 7],
    [78, 75, 6],
    [88, 90, 4],
], dtype=float)

# Prototype labels only; these are not clinical judgments.
training_labels = [
    "Easy", "Easy", "Easy", "Easy", "Easy", "Easy",
    "Medium", "Medium", "Medium", "Medium",
    "Hard", "Hard", "Easy", "Easy", "Medium",
    "Medium", "Medium", "Hard",
]

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
model.fit(training_data, training_labels)


class PerformanceInput(BaseModel):
    score_percent: float = Field(ge=0, le=100)
    accuracy_percent: float = Field(ge=0, le=100)
    avg_response_time_seconds: float = Field(gt=0, le=600)


@app.get("/")
def home():
    return {
        "message": "MindCare AI recommendation API is running"
    }


@app.post("/recommend")
def recommend(data: PerformanceInput):
    features = np.array([[
        data.score_percent,
        data.accuracy_percent,
        data.avg_response_time_seconds,
    ]])

    difficulty = model.predict(features)[0]

    explanations = {
        "Easy": "Start with a simpler game challenge.",
        "Medium": "Try a moderate game challenge.",
        "Hard": "Try a more challenging game level.",
    }

    return {
        "recommended_difficulty": difficulty,
        "explanation": explanations[difficulty],
        "model": "Decision Tree",
        "purpose": "Game personalization prototype only",
    }
