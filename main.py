
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sklearn.tree import DecisionTreeClassifier
import numpy as np

app = FastAPI(
    title="MindCare AI Recommendation API",
    description="Prototype for personalized cognitive game difficulty",
    version="1.1.0",
)

training_data = np.array([
    [20, 25, 15], [30, 35, 12], [40, 40, 10],
    [45, 50, 9], [50, 55, 8], [55, 60, 7],
    [60, 65, 6], [65, 70, 5], [70, 75, 5],
    [75, 80, 4], [80, 85, 3], [90, 95, 2],
    [35, 30, 14], [48, 45, 11], [58, 55, 9],
    [68, 65, 7], [78, 75, 6], [88, 90, 4],
], dtype=float)

training_labels = [
    "Easy", "Easy", "Easy", "Easy", "Easy", "Easy",
    "Medium", "Medium", "Medium", "Medium", "Hard",
    "Hard", "Easy", "Easy", "Medium", "Medium",
    "Medium", "Hard",
]

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(training_data, training_labels)


class PerformanceInput(BaseModel):
    score_percent: float = Field(ge=0, le=100)
    accuracy_percent: float = Field(ge=0, le=100)
    avg_response_time_seconds: float = Field(gt=0, le=600)


class HistoryInput(BaseModel):
    recent_games: list[PerformanceInput] = Field(
        min_length=2, max_length=10
    )


def predict_difficulty(score, accuracy, response_time):
    features = np.array([[score, accuracy, response_time]], dtype=float)
    return str(model.predict(features)[0])


def get_explanation(difficulty):
    return {
        "Easy": "Try a simpler game challenge and build confidence.",
        "Medium": "Try a moderate challenge at a comfortable pace.",
        "Hard": "Your results suggest trying a more challenging game.",
    }[difficulty]


@app.get("/")
def home():
    return {
        "message": "MindCare AI recommendation API is running",
        "version": "1.1.0",
    }


# Existing Android endpoint — keep compatible.
@app.post("/recommend")
def recommend(data: PerformanceInput):
    difficulty = predict_difficulty(
        data.score_percent,
        data.accuracy_percent,
        data.avg_response_time_seconds,
    )
    return {
        "recommended_difficulty": difficulty,
        "explanation": get_explanation(difficulty),
        "model": "Decision Tree",
        "purpose": "Game personalization prototype only",
    }


# New endpoint — analyzes recent game history.
@app.post("/recommend/personalized")
def recommend_personalized(data: HistoryInput):
    games = data.recent_games

    average_score = float(np.mean([g.score_percent for g in games]))
    average_accuracy = float(np.mean([g.accuracy_percent for g in games]))
    average_time = float(np.mean([
        g.avg_response_time_seconds for g in games
    ]))

    difficulty = predict_difficulty(
        average_score, average_accuracy, average_time
    )

    midpoint = len(games) // 2
    earlier = games[:midpoint]
    recent = games[midpoint:]

    earlier_avg = (
        np.mean([g.score_percent for g in earlier])
        + np.mean([g.accuracy_percent for g in earlier])
    ) / 2

    recent_avg = (
        np.mean([g.score_percent for g in recent])
        + np.mean([g.accuracy_percent for g in recent])
    ) / 2

    change = float(recent_avg - earlier_avg)

    if change >= 5:
        trend = "Improving"
        trend_message = "Your recent game results are improving."
    elif change <= -5:
        trend = "Declining"
        trend_message = (
            "Your recent results are lower. Try a comfortable "
            "challenge and play at your own pace."
        )
    else:
        trend = "Stable"
        trend_message = "Your recent game results are fairly stable."

    return {
        "recommended_difficulty": difficulty,
        "explanation": get_explanation(difficulty),
        "performance_trend": trend,
        "trend_message": trend_message,
        "games_analyzed": len(games),
        "average_score_percent": round(average_score, 2),
        "average_accuracy_percent": round(average_accuracy, 2),
        "average_response_time_seconds": round(average_time, 2),
        "model": "Decision Tree with recent-history analysis",
        "purpose": "Game personalization prototype only",
        "notice": "Experimental game guidance, not a medical assessment.",
    }
