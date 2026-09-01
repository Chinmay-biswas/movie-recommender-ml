# Movie Recommender API

A Flask JSON API that returns content-based movie recommendations from the metadata in the bundled `movies.csv` dataset. At startup, the app converts each movie's `tags` text into a sparse bag-of-words representation and uses cosine similarity to find related titles.

## Features

- Content-based recommendations derived from movie metadata tags
- Case-insensitive title search
- JSON endpoints for API status, the movie catalogue, search, and recommendations
- CORS middleware for browser-based clients
- Bundled dataset with 9,998 movie records

## How it works

The application expects `movies.csv` to be in the project root with these columns:

- `id`
- `title`
- `tags`

It normalizes the title and tag fields, then builds a `CountVectorizer` with up to 5,000 features and English stop-word removal. Recommendation results are ranked with cosine similarity. No pre-trained model download is required; the vector representation is built when the application starts.

## Tech stack

- Python
- Flask and Flask-CORS
- pandas
- scikit-learn
- Gunicorn (included for WSGI hosting)

## Getting started

```bash
git clone https://github.com/Chinmay-biswas/movie-recommender-ml.git
cd movie-recommender-ml

python -m venv .venv
```

Activate the virtual environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Install the dependencies and start the API:

```bash
python -m pip install -r requirements.txt
python app.py
```

The server listens on port `5000` by default. Set the `PORT` environment variable before starting the app to use a different port.

## API reference

| Endpoint | Description |
| --- | --- |
| `GET /` | Returns a status message and the number of loaded movies. |
| `GET /movies` | Returns the full catalogue as `id` and `title` records. |
| `GET /search/<query>` | Returns up to 10 case-insensitive literal title matches. |
| `GET /recommend/<movie>` | Returns up to 41 positively similar movie records for a matched title. |

Example requests:

```bash
curl http://127.0.0.1:5000/
curl "http://127.0.0.1:5000/search/Godfather"
curl "http://127.0.0.1:5000/recommend/The%20Godfather"
```

When a title cannot be matched, `/recommend/<movie>` returns HTTP `404`:

```json
{
  "error": "Movie not found",
  "movie": "Unknown title"
}
```

## Project structure

```text
app.py                 Flask API and recommendation logic
movies.csv             Movie metadata used by the recommender
requirements.txt       Python dependencies
movieRecomander.png    Development-process screenshot
```

## Current limitations

- Recommendations use only the supplied `tags` text; there are no user profiles, ratings, or collaborative-filtering features.
- Responses include only `id` and `title`, not similarity scores.
- The recommendation response can include the queried movie itself and may contain fewer than 41 records.
- If multiple rows share a title, the first exact or partial title match is used.
- `/movies` returns the whole catalogue without pagination.

## Notes

This repository contains an API only; it does not include a frontend client, deployment configuration, automated tests, or model-quality evaluation.
