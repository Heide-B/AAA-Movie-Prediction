
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer, normalize

from models import recommender as mr

@pytest.fixture
def fitted_model(monkeypatch):
    """Create a small synthetic dataset and fitted model state."""

    data = pd.DataFrame({
        "tconst": ["m1", "m2", "m3", "m4"],
        "primaryTitle": [
            "Action One",
            "Action Two",
            "Comedy One",
            "Drama One",
        ],
        "genres": [
            "Action Adventure",
            "Action",
            "Comedy",
            "Drama",
        ],
        "actor": [
            "Alice Smith|Bob Jones",
            "Alice Smith",
            "Carol White",
            "David Green",
        ],
        "director": [
            "Director A",
            "Director A",
            "Director B",
            "Director C",
        ],
        "movie_category": ["AAA", "AAA", "A", "AA"],
        "weightedRating": [8.5, 7.5, 9.0, 6.5],
    })

    data["genre_list"] = data["genres"].apply(mr.parse_genres)
    data["actor_list"] = data["actor"].apply(mr.parse_entities)
    data["director_list"] = data["director"].apply(mr.parse_entities)

    data["rating_score"] = (
        data["weightedRating"].rank(pct=True)
    )

    data["tokens"] = data.apply(
        lambda row: (
            [f"category:{row['movie_category'].lower()}"]
            + [f"genre:{g}" for g in row["genre_list"]]
            + [f"actor:{a}" for a in row["actor_list"]]
            + [f"director:{d}" for d in row["director_list"]]
        ),
        axis=1,
    )

    encoder = MultiLabelBinarizer(sparse_output=True)
    X = encoder.fit_transform(data["tokens"]).tocsr()

    feature_index = {
        token: i
        for i, token in enumerate(encoder.classes_)
    }

    X_knn = normalize(X, norm="l2", axis=1)

    n_components = min(2, X.shape[0] - 1, X.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_svd = normalize(svd.fit_transform(X), norm="l2", axis=1)

    genre_to_indices = {}
    actor_to_indices = {}
    director_to_indices = {}

    for i, row in data.iterrows():
        for genre in row["genre_list"]:
            genre_to_indices.setdefault(genre, []).append(i)

        for actor in row["actor_list"]:
            actor_to_indices.setdefault(actor, []).append(i)

        for director in row["director_list"]:
            director_to_indices.setdefault(director, []).append(i)

    artifacts = {
        "df": data,
        "encoder": encoder,
        "feature_index": feature_index,
        "X": X,
        "X_knn": X_knn,
        "svd": svd,
        "X_svd": X_svd,
        "genre_to_indices": genre_to_indices,
        "actor_to_indices": actor_to_indices,
        "director_to_indices": director_to_indices,
    }

    for name, value in artifacts.items():
        monkeypatch.setattr(mr, name, value, raising=False)

    return artifacts


# --------------------------------------------------
# Parsing tests
# --------------------------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("Action Comedy", ["action", "comedy"]),
        (" Drama  Action Drama ", ["action", "drama"]),
        ("", []),
        (None, []),
        (np.nan, []),
    ],
)
def test_parse_genres(value, expected):
    assert mr.parse_genres(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (
            "Alice Smith|Bob Jones",
            ["alice smith", "bob jones"],
        ),
        (
            "Alice Smith; Bob Jones|Alice Smith",
            ["alice smith", "bob jones"],
        ),
        ("Walton Goggins", ["walton goggins"]),
        ("", []),
        (None, []),
        (np.nan, []),
    ],
)
def test_parse_entities(value, expected):
    assert mr.parse_entities(value) == expected


# --------------------------------------------------
# Core helper tests
# --------------------------------------------------

def test_get_movie_vectors_knn(fitted_model):
    assert mr.get_movie_vectors("KNN") is fitted_model["X_knn"]


def test_get_movie_vectors_svd(fitted_model):
    assert mr.get_movie_vectors("SVD") is fitted_model["X_svd"]


def test_get_movie_vectors_invalid_method(fitted_model):
    with pytest.raises(ValueError, match="method"):
        mr.get_movie_vectors("INVALID")


def test_profile_vector_returns_normalized_vector(fitted_model):
    profile = mr.profile_vector([0, 1], "KNN")

    assert profile is not None
    assert profile.shape == (1, fitted_model["X_knn"].shape[1])
    assert np.linalg.norm(profile) == pytest.approx(1.0)


def test_profile_vector_empty_indices(fitted_model):
    assert mr.profile_vector([], "KNN") is None


def test_similarity_to_profile_returns_valid_similarity(fitted_model):
    score = mr.similarity_to_profile([0], [0, 1], "KNN")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_similarity_to_profile_empty_indices(fitted_model):
    assert mr.similarity_to_profile([], [0], "KNN") == 0.0
    assert mr.similarity_to_profile([0], [], "KNN") == 0.0


def test_blended_score():
    result = mr.blended_score(0.8, 0.5)

    expected = (
        mr.SIMILARITY_WEIGHT * 0.8
        + mr.RATING_WEIGHT * 0.5
    )
    assert result == pytest.approx(expected)


def test_mean_rating_score(fitted_model):
    expected = fitted_model["df"].iloc[[0, 1]]["rating_score"].mean()

    assert mr.mean_rating_score([0, 1]) == pytest.approx(expected)
    assert mr.mean_rating_score([]) == 0.0


def test_mean_weighted_rating(fitted_model):
    assert mr.mean_weighted_rating([0, 1]) == pytest.approx(8.0)
    assert np.isnan(mr.mean_weighted_rating([]))


# --------------------------------------------------
# Category and query similarity tests
# --------------------------------------------------

def test_get_category_indices_is_case_insensitive(fitted_model):
    assert mr.get_category_indices(" aaa ") == [0, 1]


def test_get_category_indices_unknown_category(fitted_model):
    assert mr.get_category_indices("UNKNOWN") == []


def test_query_similarity_returns_one_score_per_movie(fitted_model):
    scores = mr.query_similarity(["genre:action"], method="KNN")

    assert scores.shape == (len(fitted_model["df"]),)
    assert np.all(np.isfinite(scores))
    assert np.all((scores >= 0) & (scores <= 1))


def test_query_similarity_unknown_tokens_returns_zeros(fitted_model):
    scores = mr.query_similarity(["genre:not_a_real_genre"])

    assert np.all(scores == 0)


def test_query_similarity_invalid_method(fitted_model):
    with pytest.raises(ValueError, match="method"):
        mr.query_similarity(["genre:action"], method="INVALID")


# --------------------------------------------------
# Recommendation tests
# --------------------------------------------------

def test_recommend_genres_returns_expected_columns(fitted_model):
    result = mr.recommend_genres("AAA", method="KNN", n=2)

    assert len(result) <= 2
    assert {
        "genre",
        "movie_count",
        "mean_weighted_rating",
        "similarity",
        "rating_score",
        "score",
    }.issubset(result.columns)


def test_recommend_genres_unknown_category_raises_error(fitted_model):
    with pytest.raises(ValueError, match="No movies found"):
        mr.recommend_genres("UNKNOWN")


def test_recommend_people_returns_actors(fitted_model):
    result = mr.recommend_people(
        genre="action",
        category="AAA",
        person_type="actor",
        method="KNN",
        n=3,
    )

    assert not result.empty
    assert "alice smith" in result["person"].tolist()
    assert set(result["person_type"]) == {"actor"}


def test_recommend_people_returns_directors(fitted_model):
    result = mr.recommend_people(
        genre="action",
        category="AAA",
        person_type="director",
        method="KNN",
        n=3,
    )

    assert not result.empty
    assert "director a" in result["person"].tolist()
    assert set(result["person_type"]) == {"director"}


def test_recommend_people_invalid_type_raises_error(fitted_model):
    with pytest.raises(ValueError, match="person_type"):
        mr.recommend_people(
            "action",
            "AAA",
            person_type="producer",
        )


def test_recommend_people_unknown_genre_returns_empty(fitted_model):
    result = mr.recommend_people(
        "unknown",
        "AAA",
        person_type="actor",
    )

    assert result.empty


def test_recommend_movies_returns_expected_columns(fitted_model):
    result = mr.recommend_movies(
        genres=["action"],
        actors=["alice smith"],
        directors=["director a"],
        method="KNN",
        n=3,
    )

    assert not result.empty
    assert len(result) <= 3
    assert {
        "tconst",
        "primaryTitle",
        "movie_category",
        "genres",
        "weightedRating",
        "feature_similarity",
        "final_score",
    }.issubset(result.columns)

    assert result["final_score"].is_monotonic_decreasing


def test_recommend_movies_unknown_features_raise_error(fitted_model):
    with pytest.raises(ValueError, match="None of the selected"):
        mr.recommend_movies(
            genres=["unknown_genre"],
            actors=[],
            directors=[],
        )


def test_recommendation_pipeline_invalid_method(fitted_model):
    with pytest.raises(ValueError, match="method"):
        mr.run_recommendation_pipeline(
            category="AAA",
            method="INVALID",
            verbose=False,
        )


# --------------------------------------------------
# Model loading test
# --------------------------------------------------

def test_load_model_restores_artifacts(tmp_path, monkeypatch):
    model_path = tmp_path / "test_model.joblib"

    artifacts = {
        "df": pd.DataFrame({"primaryTitle": ["Test Movie"]}),
        "feature_index": {"genre:action": 0},
        "SIMILARITY_WEIGHT": 0.6,
        "RATING_WEIGHT": 0.4,
    }
    joblib.dump(artifacts, model_path)

    loaded = mr.load_model(model_path)

    assert loaded is True
    pd.testing.assert_frame_equal(mr.df, artifacts["df"])
    assert mr.feature_index == artifacts["feature_index"]
    assert mr.SIMILARITY_WEIGHT == pytest.approx(0.6)
    assert mr.RATING_WEIGHT == pytest.approx(0.4)
