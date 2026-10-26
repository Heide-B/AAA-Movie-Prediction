from models import recommender
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def predict_info():
    recommender.load_model("movie_recommender.joblib")
    results = recommender.run_recommendation_pipeline(
        category="AAA",
        method="SVD",
    )

    return {
        "genres": (results["genres"].tolist()),
        "people": (results["people"].tolist()),
        "movies": (results["movies"].tolist()),

    }
