import os
import sys
import json
import random
import datetime
import csv
from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

# Konfiguriere Flask-App und gebe den Pfad zu Templates und Static an.
# Da app.py im Ordner 'demos' liegt, müssen wir die übergeordneten Ordner angeben.
app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Modell und Tokenizer laden (ersetze model_name ggf. durch den Pfad deines feinabgestimmten Modells)
model_name = "bert-base-uncased"  # oder "path/to/your/fine_tuned_model"
print("Lade Tokenizer und Modell ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
print("Modell und Tokenizer geladen.")

# Hilfsfunktion zur Vorhersage
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

# Pfad zur CSV-Datei, in der die Posts gespeichert werden (relativ zum demos-Ordner)
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
        if community_note:
            augmented_text = f"{post_text} Note: {community_note}"
        else:
            augmented_text = post_text

        # Vorhersage berechnen
        probability = predict_post(augmented_text)
        predicted_label = "Bot" if probability > 0.5 else "Nicht Bot"

        # Speichere den Post in der CSV-Datei
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, post_text, community_note, f"{probability:.4f}", predicted_label])

        return redirect(url_for("posts"))  # Direkt zur posts.html umleiten
    return render_template("index.html")

@app.route("/posts")
def posts():
    posts = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            posts.append(row)
    posts = sorted(posts, key=lambda x: x["timestamp"], reverse=True)
    return render_template("posts.html", posts=posts)

if __name__ == "__main__":
    # Hinweis: Das Laden des BERT-Modells kann einige Zeit in Anspruch nehmen.
    # Sobald das Modell geladen ist, wird die App gestartet.
    print("Starte Flask-App. Bitte gedulde dich bis zum Laden des Modells...")
    print("App starten: http://127.0.0.1:5000/")
    app.run(debug=True)
    