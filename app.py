from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

# Load movies
movies = pd.read_csv("movies.csv")

# Create vectors when server starts
cv = CountVectorizer(
    max_features=5000,
    stop_words='english'
)

vectors = cv.fit_transform(
    movies['tags'].astype(str)
).toarray()


@app.route('/')
def home():
    return jsonify({
        "message": "Movie Recommendation API Running"
    })


@app.route('/movies')
def get_movies():

    result = movies[['id', 'title']]

    return jsonify(
        result.to_dict('records')
    )


@app.route('/search/<query>')
def search(query):

    result = movies[
        movies['title'].str.contains(
            query,
            case=False,
            na=False
        )
    ][['id', 'title']].head(10)

    return jsonify(
        result.to_dict('records')
    )


@app.route('/recommend/<movie>')
def recommend(movie):

    try:

        movie_index = movies[
            movies['title'] == movie
        ].index[0]

        distances = cosine_similarity(
            vectors[movie_index].reshape(1, -1),
            vectors
        ).flatten()

        movie_list = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )[1:40]

        recommendations = []

        for i in movie_list:

            recommendations.append({
                "id": int(movies.iloc[i[0]].id),
                "title": movies.iloc[i[0]].title
            })

        return jsonify(recommendations)

    except Exception as e:

        return jsonify({
            "error": "Movie not found",
            "message": str(e)
        }), 404


if __name__ == "__main__":
    app.run()