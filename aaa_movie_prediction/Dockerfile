
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install project dependencies from pyproject.toml.
COPY pyproject.toml ./
COPY models/ ./models/
COPY run_recommender.py ./

RUN python -m pip install --upgrade pip \
    && python -m pip install .

# The model artifact is needed at runtime.
COPY models/movie_recommender.joblib ./models/movie_recommender.joblib

EXPOSE 8000

CMD ["uvicorn", "run_recommender:app", "--host", "0.0.0.0", "--port", "8000"]
