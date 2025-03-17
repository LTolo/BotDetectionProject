# Beschreibung: Flask-App, die eine einfache Benutzeroberfläche für die Klassifizierung von Posts als Bot oder Nicht-Bot bereitstellt.

import os
import sys
import json
import random
import datetime
import csv
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import pandas as pd

# Füge den übergeordneten Ordner zum PYTHONPATH hinzu, damit der Ordner "api" gefunden wird.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
try:
    from api.reddit_api import get_reddit_data
    print("Reddit API erfolgreich importiert.")
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None

# Konfiguriere Flask (app.py liegt im Ordner "demos")
app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Modell und Tokenizer laden (ersetze model_name ggf. durch den Pfad deines feinabgestimmten Modells)
model_name = "bert-base-uncased"
print("Lade Tokenizer und Modell ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
print("Modell und Tokenizer geladen.")

def predict_post(text, max_length=128):
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf"
    )
    logits = model.predict(encodings).logits
    prob = tf.nn.sigmoid(logits).numpy()[0][0]
    return prob

# Pfad zur CSV-Datei, in der die lokal erstellten Posts gespeichert werden
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "demodata.csv")
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "post", "community_note", "predicted_probability", "predicted_label"])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        post_text = request.form.get("post_text", "").strip()
        community_note = request.form.get("community_note", "").strip()
        # Falls eine Note eingegeben wird, hänge sie an
        if community_note:
            augmented_text = f"{post_text} Note: {community_note}"
        else:
            augmented_text = post_text

        probability = predict_post(augmented_text)
        predicted_label = "Bot" if probability > 0.5 else "Nicht Bot"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, post_text, community_note, f"{probability:.4f}", predicted_label])
        return redirect(url_for("posts"))
    return render_template("index.html")

@app.route("/posts")
def posts():
    # Bestimme, ob Details angezeigt werden sollen. Standard: True.
    details = request.args.get("details", "true").lower() == "true"

    posts_local = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["source"] = "local"
            posts_local.append(row)
    
    posts_reddit = []
    if get_reddit_data is not None:
        reddit_df = get_reddit_data(subreddit_name="learnpython", limit=10)
        if isinstance(reddit_df, pd.DataFrame) and "Tweet" in reddit_df.columns:
            for idx, row in reddit_df.iterrows():
                tweet = row["Tweet"]
                predicted_prob = predict_post(tweet)
                predicted_label = "Bot" if predicted_prob > 0.5 else "Nicht Bot"
                posts_reddit.append({
                    "timestamp": "1900-01-01 00:00:00",  # Platzhalter, damit Reddit-Posts am Ende erscheinen
                    "post": tweet,
                    "community_note": "",
                    "predicted_probability": f"{predicted_prob:.4f}",
                    "predicted_label": predicted_label,
                    "source": "reddit"
                })
        else:
            print("Reddit API lieferte unerwartete Daten.")
    
    combined_posts = posts_local + posts_reddit

    # Sortiere die lokalen Posts nach Zeitstempel (neueste oben)
    import datetime
    def sort_key(post):
        try:
            return datetime.datetime.strptime(post["timestamp"], "%Y-%m-%d %H:%M:%S")
        except Exception as e:
            return datetime.datetime.min
    combined_posts = sorted(combined_posts, key=sort_key, reverse=True)

    return render_template("posts.html", posts=combined_posts, show_details=details)

    
@app.route("/add_note/<timestamp>", methods=["GET", "POST"])
def add_note(timestamp):
    # Lese alle Posts aus der CSV-Datei ein
    posts = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append(row)
    
    # Finde den Post anhand des Zeitstempels
    post_found = None
    for post in posts:
        if post["timestamp"] == timestamp:
            post_found = post
            break
    if not post_found:
        return "Post nicht gefunden.", 404

    if request.method == "POST":
        new_note = request.form.get("community_note", "").strip()
        # Füge neue Note zu vorhandenen hinzu (mit " || " als Trenner)
        if post_found["community_note"]:
            updated_notes = post_found["community_note"] + " || " + new_note
        else:
            updated_notes = new_note
        post_found["community_note"] = updated_notes

        # Kombiniere den Originaltext und alle Community Notes, um den augmentierten Text zu bilden
        augmented_text = f"{post_found['post']} Note: {updated_notes}"
        # Neu-Klassifizierung basierend auf dem augmentierten Text
        new_prob = predict_post(augmented_text)
        new_label = "Bot" if new_prob > 0.5 else "Nicht Bot"
        post_found["predicted_probability"] = f"{new_prob:.4f}"
        post_found["predicted_label"] = new_label

        # Schreibe alle Posts zurück in die CSV
        with open(DATA_FILE, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["timestamp", "post", "community_note", "predicted_probability", "predicted_label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for p in posts:
                writer.writerow(p)
        return redirect(url_for("posts"))
    
    return render_template("add_note.html", post=post_found)

if __name__ == "__main__":
    app.run(debug=True)
