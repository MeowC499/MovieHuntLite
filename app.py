from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import torch
import torch.nn as nn
import os
import logging
import pickle

from model.ncf_model import NCF_Hybrid

# Logging setup
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)

# Global state
data = None
model = None
user_encoder = None
movie_encoder = None


def initialize_app():
    global data, model, user_encoder, movie_encoder

    if data is None or model is None:
        logging.info("Initializing application with preloaded dataset and model...")

        # Load ratings and movies from MovieLens .dat files
        logging.info("Loading MovieLens .dat files from 'data/'")
        ratings = pd.read_csv(
    "data/ratings.dat",
    sep="::",
    engine="python",
    names=["userId", "movieId", "rating", "timestamp"],
    encoding="latin-1"
)

        movies = pd.read_csv(
    "data/movies.dat",
    sep="::",
    engine="python",
    names=["movieId", "title", "genres"],
    encoding="latin-1"  # 👈 This is the magic that stops pandas from crying
)


        # Load encoders from pickle files
        with open("model/user_encoder.pkl", "rb") as f:
            user_encoder = pickle.load(f)
        with open("model/item_encoder.pkl", "rb") as f:
            movie_encoder = pickle.load(f)

        # Store all data
        data = {
            "ratings": ratings,
            "all_movies": movies,
            "user_encoder": user_encoder,
            "movie_encoder": movie_encoder
        }

        # Load trained hybrid model
        model_path = "model/ncf_hybrid_model.pth"
        model = NCF_Hybrid(num_users=len(user_encoder), num_items=len(movie_encoder))
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
            model.eval()
            logging.info("✅ Hybrid model loaded successfully.")
        else:
            logging.warning("⚠️ Model file not found. Running in dummy mode.")
            model = None


@app.route("/", methods=["GET"])
def index():
    initialize_app()
    movie_titles = data["all_movies"]["title"].dropna().tolist()
    return render_template("index.html", titles=movie_titles)


@app.route("/recommend", methods=["POST"])
def recommend():
    initialize_app()
    username = request.form.get("username")
    selected_titles = request.form.getlist("favorites")
    logging.info(f"Received input - username: {username}, favorites: {selected_titles}")

    if not username or not selected_titles:
        return redirect(url_for("index"))

    predictions = []
    if model:
        user_tensor = torch.tensor([abs(hash(username)) % len(user_encoder)], dtype=torch.long)
        all_movies = data["all_movies"]
        movie_encoder = data["movie_encoder"]

        for _, row in all_movies.iterrows():
            title = row["title"]
            movie_id = row["movieId"]

            if title in selected_titles or movie_id not in movie_encoder:
                continue

            movie_tensor = torch.tensor([movie_encoder[movie_id]], dtype=torch.long)

            with torch.no_grad():
                pred_rating = model(user_tensor, movie_tensor).item()
                predictions.append((title, round(pred_rating, 2)))

        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:20]
    else:
        predictions = [(f"Dummy Movie {i+1}", round(5 - i * 0.2, 2)) for i in range(20)]

    return render_template("results.html", username=username, recommendations=predictions, favorites=selected_titles)


@app.route("/feedback", methods=["POST"])
def feedback():
    username = request.form.get("username")
    selected_movies = request.form.get("favorites")
    feedback_text = request.form.get("feedback")

    logging.info(f"Feedback received from {username}: {feedback_text}")
    os.makedirs("data", exist_ok=True)
    with open("data/user_feedback.csv", "a", encoding="utf-8") as f:
        f.write(f"{username},{selected_movies},{feedback_text}\n")

    return render_template("thankyou.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    logging.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port)
