import os

from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

movies = pd.read_csv("movies.csv")

required_columns = {"id", "title", "tags"}
missing_columns = required_columns.difference(movies.columns)
if missing_columns:
    raise RuntimeError(f"movies.csv is missing columns: {', '.join(sorted(missing_columns))}")

movies["title"] = movies["title"].astype(str)
movies["tags"] = movies["tags"].fillna("").astype(str)

cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

# Keep this sparse Calling .toarray() can use too much memory on Render to heavy to do the work takes full day on this small project what the fuck 
vectors = cv.fit_transform(movies["tags"])


def movie_payload(row):
    return {
        "id": int(row.id),
        "title": row.title
    }


def find_movie_index(movie_title):
    normalized_title = movie_title.strip().lower()

    exact_match = movies[movies["title"].str.lower() == normalized_title]
    if not exact_match.empty:
        return exact_match.index[0]

    partial_match = movies[
        movies["title"].str.contains(
            movie_title,
            case=False,
            na=False,
            regex=False
        )
    ]
    if not partial_match.empty:
        return partial_match.index[0]

    return None


@app.route("/")
def home():
    return jsonify({
        "message": "Movie Recommendation API Running",
        "movies_count": int(len(movies))
    })


@app.route("/movies")
def get_movies():
    result = movies[["id", "title"]]
    return jsonify(result.to_dict("records"))


@app.route("/search/<query>")
def search(query):
    result = movies[
        movies["title"].str.contains(
            query,
            case=False,
            na=False,
            regex=False
        )
    ][["id", "title"]].head(10)

    return jsonify(result.to_dict("records"))


@app.route("/recommend/<movie>")
def recommend(movie):
    movie_index = find_movie_index(movie)

    if movie_index is None:
        return jsonify({
            "error": "Movie not found",
            "movie": movie
        }), 404

    distances = cosine_similarity(
        vectors[movie_index],
        vectors
    ).flatten()

    movie_list = sorted(
        enumerate(distances),
        reverse=True,
        key=lambda item: item[1]
    )[1:41]

    recommendations = [
        movie_payload(movies.iloc[index])
        for index, score in movie_list
        if score > 0
    ]

    return jsonify(recommendations)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
