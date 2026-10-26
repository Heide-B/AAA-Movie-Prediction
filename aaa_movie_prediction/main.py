
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from run_recommender import router as recommender_router

app = FastAPI(
    title="AAA Movie Prediction API",
    description="Movie recommendation API",
    version="0.1.0",
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


app.include_router(recommender_router, prefix="/predict", tags=["Prediction"])
