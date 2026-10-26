
from models import recommender

recommender.load_model("movie_recommender.joblib")
results = recommender.run_recommendation_pipeline(
    category="AAA",
    method="SVD",
)

print(results["genres"])
print(results["people"])
print(results["movies"])
