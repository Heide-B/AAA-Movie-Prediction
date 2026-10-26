import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix


RANDOM_STATE = 42
TEST_SIZE = 0.20
N_COMPONENTS = 50
K = 10
RELEVANCE_THRESHOLD = 0.10

SIMILARITY_WEIGHT = 0.60
RATING_WEIGHT = 0.40

GENRE_WEIGHT = 0.50
ACTOR_WEIGHT = 0.25
DIRECTOR_WEIGHT = 0.25

TOP_N_GENRES = 3
TOP_N_PEOPLE = 3
TOP_N_MOVIES = 3
GROUPS = {
    "genre": (GENRE_WEIGHT, "genre_set"),
    "actor": (ACTOR_WEIGHT, "actor_set"),
    "director": (DIRECTOR_WEIGHT, "director_set"),
}

MODEL_PATH = Path("movie_recommender.joblib")

required_columns = [
    "tconst",
    "primaryTitle",
    "genres",
    "actor",
    "director",
    "movie_category",
    "weightedRating",
]

def parse_genres(value):
    """Parse space-separated genre names."""
    if pd.isna(value):
        return []

    return sorted({
        item.strip().lower()
        for item in str(value).split()
        if item.strip()
    })


def parse_entities(value):
    """
    Parse actor/director names separated by | or ;.

    Names are kept intact: for example, 'walton goggins'
    remains one person rather than two separate tokens.
    """
    if pd.isna(value):
        return []

    return sorted({
        item.strip().lower()
        for item in re.split(r"[|;]", str(value))
        if item.strip()
    })


def run_preprocessing(df):
    agg_dict = {
        'weightedRating': 'max',
        'numVotes': 'max',
        'genres': 'max',
        'movie_category': 'max',
        'primaryTitle': 'max',
        'director': 'max',
        'actor': '|'.join
    }

    final_df = df.groupby(['tconst', 'startYear'], as_index=False).agg(agg_dict)
    missing_columns = [
        col for col in required_columns
        if col not in final_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = final_df[required_columns].copy()
    df["movie_category"] = (
        df["movie_category"]
        .astype("string")
        .str.strip()
    )
    df = df.dropna(
        subset=["tconst", "primaryTitle", "movie_category"]
    ).copy()
    df = df.drop_duplicates(subset="tconst").reset_index(drop=True)
    df["weightedRating"] = pd.to_numeric(
        df["weightedRating"],
        errors="coerce"
    )
    df["genre_list"] = df["genres"].apply(parse_genres)
    df["actor_list"] = df["actor"].apply(parse_entities)
    df["director_list"] = df["director"].apply(parse_entities)
    if df["weightedRating"].notna().any():
        df["rating_score"] = (
            df["weightedRating"]
            .rank(method="average", pct=True)
        )
    else:
        df["rating_score"] = 0.5

    df["rating_score"] = df["rating_score"].fillna(0.5)

    df["tokens"] = df.apply(
        lambda row: (
                [f"category:{row['movie_category'].lower()}"]
                + [f"genre:{g}" for g in row["genre_list"]]
                + [f"actor:{a}" for a in row["actor_list"]]
                + [f"director:{d}" for d in row["director_list"]]
        ),
        axis=1,
    )
    encoder = MultiLabelBinarizer(sparse_output=True)

    X = encoder.fit_transform(df["tokens"]).tocsr()

    feature_index = {
        token: i
        for i, token in enumerate(encoder.classes_)
    }

    # KNN operates on the original sparse feature matrix.
    X_knn = normalize(X, norm="l2", axis=1)

    # SVD learns a lower-dimensional representation of the same data.
    n_components = min(
        N_COMPONENTS,
        X.shape[0] - 1,
        X.shape[1] - 1
    )

    if n_components < 1:
        raise ValueError(
            "Not enough movies or features to fit SVD."
        )

    svd = TruncatedSVD(
        n_components=n_components,
        random_state=RANDOM_STATE
    )

    X_svd = svd.fit_transform(X)

    # Normalize movie vectors so cosine similarity is comparable.
    X_svd = normalize(X_svd, norm="l2", axis=1)

    # Cache movie indices for each genre, actor, and director.
    genre_to_indices = {}
    actor_to_indices = {}
    director_to_indices = {}

    for i, row in df.iterrows():
        for genre in row["genre_list"]:
            genre_to_indices.setdefault(genre, []).append(i)

        for actor in row["actor_list"]:
            actor_to_indices.setdefault(actor, []).append(i)

        for director in row["director_list"]:
            director_to_indices.setdefault(director, []).append(i)

def get_movie_vectors(method):
    """Return the movie vectors for the selected method."""
    method = method.upper()

    if method == "KNN":
        return X_knn

    if method == "SVD":
        return X_svd

    raise ValueError("method must be 'KNN' or 'SVD'")


def profile_vector(movie_indices, method):
    """
    Represent a group of movies by its mean feature vector.

    KNN: mean vector in the original feature space.
    SVD: mean vector in the latent space.
    """
    if len(movie_indices) == 0:
        return None

    vectors = get_movie_vectors(method)
    profile = np.asarray(
        vectors[movie_indices].mean(axis=0)
    ).reshape(1, -1)

    return normalize(profile, norm="l2", axis=1)


def similarity_to_profile(candidate_indices, target_indices, method):
    """
    Compute candidate-profile similarity to a target profile.

    A candidate's profile is the mean representation of its
    associated movies. The target profile is also a mean
    representation, calculated using the same method.
    """
    if not candidate_indices or not target_indices:
        return 0.0

    target = profile_vector(target_indices, method)
    candidate = profile_vector(candidate_indices, method)

    if target is None or candidate is None:
        return 0.0

    return float(cosine_similarity(target, candidate)[0, 0])


def blended_score(similarity, rating_score):
    """Apply identical similarity/rating weights to both methods."""
    return (
        SIMILARITY_WEIGHT * similarity
        + RATING_WEIGHT * rating_score
    )


def mean_rating_score(indices):
    """Mean normalized rating score for a group of movies."""
    if not indices:
        return 0.0

    return float(df.iloc[indices]["rating_score"].mean())


def mean_weighted_rating(indices):
    """Mean original weighted rating for a group of movies."""
    if not indices:
        return np.nan

    return float(df.iloc[indices]["weightedRating"].mean())


def get_category_indices(category):
    """Find movies matching the requested category."""
    category = str(category).strip().lower()

    return df.index[
        df["movie_category"].str.lower() == category
    ].tolist()


def query_similarity(
    query_tokens,
    method="KNN"
):
    """
    Calculate each movie's cosine similarity to a query
    containing selected genres, actors, and directors.

    KNN uses the original feature space.
    SVD projects the query into the learned latent space.
    """
    method = method.upper()

    valid_tokens = [
        token for token in query_tokens
        if token in feature_index
    ]

    scores = np.zeros(len(df), dtype=float)

    if not valid_tokens:
        return scores

    query_columns = [
        feature_index[token]
        for token in valid_tokens
    ]

    # Build a binary query vector in the original feature space.
    query = csr_matrix(
        (
            np.ones(len(query_columns)),
            (
                np.zeros(len(query_columns), dtype=int),
                query_columns
            )
        ),
        shape=(1, X.shape[1])
    )

    query = normalize(query, norm="l2", axis=1)

    if method == "KNN":
        scores = cosine_similarity(
            query,
            X_knn
        ).ravel()

    elif method == "SVD":
        query_latent = svd.transform(query)
        query_latent = normalize(
            query_latent,
            norm="l2",
            axis=1
        )

        scores = cosine_similarity(
            query_latent,
            X_svd
        ).ravel()

    else:
        raise ValueError("method must be 'KNN' or 'SVD'")

    # Negative cosine values are not useful as positive
    # recommendation relevance scores.
    return np.clip(scores, 0.0, 1.0)

def recommend_genres(
    category,
    method="KNN",
    n=TOP_N_GENRES
):
    """
    Rank genres by similarity to the selected category's
    movie profile, combined with their associated ratings.
    """
    category_indices = get_category_indices(category)

    if not category_indices:
        raise ValueError(
            f"No movies found for category '{category}'. "
            f"Available categories: "
            f"{sorted(df['movie_category'].dropna().unique())}"
        )

    results = []

    for genre, movie_indices in genre_to_indices.items():
        similarity = similarity_to_profile(
            movie_indices,
            category_indices,
            method
        )

        rating = mean_rating_score(movie_indices)

        score = blended_score(similarity, rating)

        results.append({
            "genre": genre,
            "movie_count": len(movie_indices),
            "mean_weighted_rating": mean_weighted_rating(
                movie_indices
            ),
            "similarity": similarity,
            "rating_score": rating,
            "score": score,
        })

    result_df = pd.DataFrame(results)

    return (
        result_df
        .sort_values(
            ["score", "movie_count"],
            ascending=[False, False]
        )
        .head(n)
        .reset_index(drop=True)
    )


def recommend_people(
    genre,
    category,
    person_type="actor",
    method="KNN",
    n=TOP_N_PEOPLE
):
    """
    Rank actors or directors based on:
      1. Similarity to the selected category + genre profile.
      2. Mean rating score of their associated movies.

    If the selected category has no movies in the genre,
    fall back to all movies belonging to that genre.
    """
    genre = str(genre).strip().lower()
    person_type = person_type.strip().lower()

    if person_type not in {"actor", "director"}:
        raise ValueError(
            "person_type must be 'actor' or 'director'"
        )

    genre_indices = genre_to_indices.get(genre, [])

    if not genre_indices:
        return pd.DataFrame()

    category_indices = set(get_category_indices(category))

    # Prefer movies that match BOTH category and genre.
    target_indices = [
        i for i in genre_indices
        if i in category_indices
    ]

    # Fallback if the category/genre intersection is empty.
    if not target_indices:
        target_indices = genre_indices

    if person_type == "actor":
        person_map = actor_to_indices
    else:
        person_map = director_to_indices

    results = []

    for person, person_indices in person_map.items():
        # Only consider people who have at least one movie
        # associated with the genre being recommended.
        relevant_indices = [
            i for i in person_indices
            if i in set(genre_indices)
        ]

        if not relevant_indices:
            continue

        similarity = similarity_to_profile(
            relevant_indices,
            target_indices,
            method
        )

        rating = mean_rating_score(relevant_indices)

        score = blended_score(similarity, rating)

        results.append({
            "person": person,
            "person_type": person_type,
            "genre": genre,
            "movie_count": len(relevant_indices),
            "mean_weighted_rating": mean_weighted_rating(
                relevant_indices
            ),
            "similarity": similarity,
            "rating_score": rating,
            "score": score,
        })

    if not results:
        return pd.DataFrame()

    return (
        pd.DataFrame(results)
        .sort_values(
            ["score", "movie_count"],
            ascending=[False, False]
        )
        .head(n)
        .reset_index(drop=True)
    )


def recommend_movies(
    genres,
    actors,
    directors,
    method="KNN",
    n=TOP_N_MOVIES
):
    """
    Rank movies by similarity to selected genres, actors,
    and directors, combined with normalized weighted ratings.

    Movies may belong to any category.
    """
    genre_tokens = [
        f"genre:{str(g).strip().lower()}"
        for g in genres
    ]

    actor_tokens = [
        f"actor:{str(a).strip().lower()}"
        for a in actors
    ]

    director_tokens = [
        f"director:{str(d).strip().lower()}"
        for d in directors
    ]

    groups = [
        (genre_tokens, GENRE_WEIGHT),
        (actor_tokens, ACTOR_WEIGHT),
        (director_tokens, DIRECTOR_WEIGHT),
    ]

    # Calculate each feature group's similarities separately.
    group_scores = []
    active_weights = []

    for tokens, weight in groups:
        if any(token in feature_index for token in tokens):
            scores = query_similarity(tokens, method)
            group_scores.append(scores * weight)
            active_weights.append(weight)

    if not active_weights:
        raise ValueError(
            "None of the selected genres, actors, or directors "
            "were found in the fitted feature vocabulary."
        )

    total_weight = sum(active_weights)

    combined_similarity = (
        np.sum(group_scores, axis=0) / total_weight
    )

    rating_scores = df["rating_score"].to_numpy()

    final_scores = (
        SIMILARITY_WEIGHT * combined_similarity
        + RATING_WEIGHT * rating_scores
    )

    # Require at least one exact selected-feature match.
    # This prevents unrelated movies from entering solely
    # because their rating is high.
    matched_indices = set()

    for token in (
        genre_tokens + actor_tokens + director_tokens
    ):
        if token not in feature_index:
            continue

        column = feature_index[token]
        matching_movies = X[:, column].nonzero()[0]
        matched_indices.update(matching_movies.tolist())

    if not matched_indices:
        return pd.DataFrame()

    candidate_indices = np.array(
        sorted(matched_indices),
        dtype=int
    )

    result = df.iloc[candidate_indices][
        [
            "tconst",
            "primaryTitle",
            "movie_category",
            "genres",
            "weightedRating",
        ]
    ].copy()

    result["feature_similarity"] = combined_similarity[
        candidate_indices
    ]

    result["final_score"] = final_scores[candidate_indices]

    result = result.sort_values(
        ["final_score", "feature_similarity", "weightedRating"],
        ascending=[False, False, False]
    )

    return result.head(n).reset_index(drop=True)


def run_recommendation_pipeline(
    category,
    method="KNN",
    n_genres=TOP_N_GENRES,
    n_people=TOP_N_PEOPLE,
    n_movies=TOP_N_MOVIES,
    verbose=True
):
    """
    Run genre, actor, director, and movie recommendations
    using either KNN or SVD.
    """
    method = method.upper()

    if method not in {"KNN", "SVD"}:
        raise ValueError("method must be 'KNN' or 'SVD'")

    # Stage 1: genres.
    genres_df = recommend_genres(
        category=category,
        method=method,
        n=n_genres
    )

    # Stage 2: actors and directors for each genre.
    people_results = {}

    for genre in genres_df["genre"]:
        people_results[genre] = {
            "actors": recommend_people(
                genre=genre,
                category=category,
                person_type="actor",
                method=method,
                n=n_people
            ),
            "directors": recommend_people(
                genre=genre,
                category=category,
                person_type="director",
                method=method,
                n=n_people
            ),
        }

    # Use all recommended entities as the movie query.
    selected_genres = genres_df["genre"].tolist()

    selected_actors = []
    selected_directors = []

    for genre, people in people_results.items():
        selected_actors.extend(
            people["actors"]["person"].tolist()
            if not people["actors"].empty else []
        )

        selected_directors.extend(
            people["directors"]["person"].tolist()
            if not people["directors"].empty else []
        )

    # Remove duplicates while preserving order.
    selected_actors = list(dict.fromkeys(selected_actors))
    selected_directors = list(dict.fromkeys(selected_directors))

    # Stage 3: movies.
    movies_df = recommend_movies(
        genres=selected_genres,
        actors=selected_actors,
        directors=selected_directors,
        method=method,
        n=n_movies
    )

    output = {
        "method": method,
        "category": category,
        "genres": genres_df,
        "people": people_results,
        "movies": movies_df,
    }

    if verbose:
        print(f"\n{'=' * 65}")
        print(f"{method} RECOMMENDATIONS | CATEGORY: {category}")
        print(f"{'=' * 65}")

        print("\nTOP GENRES")
        print(
            genres_df[
                [
                    "genre",
                    "movie_count",
                    "mean_weighted_rating",
                    "similarity",
                    "score",
                ]
            ].round(3).to_string(index=False)
        )

        for genre, people in people_results.items():
            print(f"\nGENRE: {genre.upper()}")

            print("\n  Actors")
            if people["actors"].empty:
                print("  No actors found.")
            else:
                print(
                    people["actors"][
                        [
                            "person",
                            "movie_count",
                            "mean_weighted_rating",
                            "similarity",
                            "score",
                        ]
                    ].round(3).to_string(index=False)
                )

            print("\n  Directors")
            if people["directors"].empty:
                print("  No directors found.")
            else:
                print(
                    people["directors"][
                        [
                            "person",
                            "movie_count",
                            "mean_weighted_rating",
                            "similarity",
                            "score",
                        ]
                    ].round(3).to_string(index=False)
                )

        print("\nTOP MOVIES")
        if movies_df.empty:
            print("No matching movies found.")
        else:
            print(
                movies_df[
                    [
                        "primaryTitle",
                        "movie_category",
                        "genres",
                        "weightedRating",
                        "feature_similarity",
                        "final_score",
                    ]
                ].round(3).to_string(index=False)
            )

    return output

def load_model(path=MODEL_PATH):
    """
    Load saved artifacts into this module's namespace.

    Call this before run_recommendation_pipeline().
    """
    global df, encoder, feature_index, X, X_knn, svd, X_svd
    global genre_to_indices, actor_to_indices, director_to_indices
    global SIMILARITY_WEIGHT, RATING_WEIGHT
    global GENRE_WEIGHT, ACTOR_WEIGHT, DIRECTOR_WEIGHT
    global TOP_N_GENRES, TOP_N_PEOPLE, TOP_N_MOVIES

    artifacts = joblib.load(path)

    globals().update(artifacts)

    print(f"Model loaded from: {path}")
    return True

if __name__ == "__main__":
    df = pd.read_csv("data/movies.csv")
    run_preprocessing(df)
    load_model()