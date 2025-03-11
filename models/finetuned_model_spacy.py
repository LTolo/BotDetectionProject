# Dieses Skript führt ein Fine-Tuning eines vortrainierten spaCy Transformer-Modells (en_core_web_trf) zur Erkennung von Bot-Accounts in Twitter-Daten durch.
# Es lädt einen CSV-Datensatz, konvertiert die Daten in das spaCy-Trainingsformat, teilt die Daten auf,
# trainiert das Modell und bewertet es.
# Optional testet es das Modell auf Daten, die über die Reddit API abgerufen wurden.
# Es verwendet spaCy für das Modelltraining und scikit-learn für die Datenaufteilung.


import os
import random
import spacy
from spacy.util import minibatch, compounding
import pandas as pd
from sklearn.model_selection import train_test_split

# Falls vorhanden, versuche die Reddit API zu importieren
try:
    from api.reddit_api import get_reddit_data
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. "
          "Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None

def load_data(file_path):
    """
    Lädt den CSV-Datensatz und konvertiert ihn in das spaCy-Trainingsformat.
    Jeder Eintrag ist ein Tupel (Text, {"cats": {"BOT": bool, "REAL": bool}}).
    """
    df = pd.read_csv(file_path)
    df["Tweet"] = df["Tweet"].astype(str)
    df["Bot Label"] = df["Bot Label"].astype(int)
    
    data = []
    for tweet, label in zip(df["Tweet"], df["Bot Label"]):
        # Wir definieren die Kategorien: True für BOT, False für REAL
        cats = {"BOT": label == 1, "REAL": label == 0}
        data.append((tweet, {"cats": cats}))
    print(f"{len(data)} Trainingsbeispiele geladen.")
    return data

def evaluate(nlp, data):
    """
    Führt eine einfache Evaluation durch, indem für jeden Text die Kategorie mit
    dem höheren Score als Vorhersage verwendet wird.
    """
    correct = 0
    total = 0
    for text, annotations in data:
        doc = nlp(text)
        predicted = doc.cats
        pred_label = "BOT" if predicted["BOT"] >= predicted["REAL"] else "REAL"
        true_label = "BOT" if annotations["cats"]["BOT"] else "REAL"
        if pred_label == true_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def test_on_reddit_data(nlp):
    """
    Holt Daten über die Reddit API, falls verfügbar, und wendet das feinabgestimmte Modell an.
    """
    if get_reddit_data is None:
        print("Reddit API nicht verfügbar. Überspringe Reddit-Test.")
        return

    print("\nHole Reddit-Daten...")
    reddit_data = get_reddit_data(subreddit_name="learnpython", limit=10)

    if not isinstance(reddit_data, pd.DataFrame):
        reddit_data = pd.DataFrame(reddit_data)

    if "Tweet" not in reddit_data.columns:
        print("Reddit-Daten enthalten nicht die erforderliche Spalte ('Tweet').")
        return

    print("\nMapping der Vorhersagen: 0 = Real Account, 1 = Bot Account")
    for tweet in reddit_data["Tweet"].fillna(""):
        doc = nlp(tweet)
        pred_label = "BOT" if doc.cats["BOT"] >= doc.cats["REAL"] else "REAL"
        print(f"Text: {tweet[:50]}... -> Vorhersage: {pred_label}")

def main():
    # Basisverzeichnis (anpassen, falls erforderlich)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "twitter_dataset.csv")
    print(f"Lade Kaggle-Daten von: {data_path}")
    
    # Daten laden und in spaCy-Format konvertieren
    data = load_data(data_path)
    random.shuffle(data)
    
    # Aufteilen in Trainings- und Testdaten (z. B. 80/20-Aufteilung)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Lade ein vortrainiertes spaCy Transformer-Modell (für feine Abstimmung)
    # Hinweis: Für deutsche Tweets z. B. "de_core_news_trf" verwenden
    nlp = spacy.load("en_core_web_trf")
    
    # Falls noch nicht vorhanden, Textklassifikator hinzufügen
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    
    # Füge die Labels hinzu
    textcat.add_label("BOT")
    textcat.add_label("REAL")
    
    # Beginne mit dem Feintuning (das Training)
    optimizer = nlp.resume_training()
    n_iter = 10  # Anzahl der Trainingsiterationen – je nach Datensatz anpassen
    print("Beginne das Fine-Tuning...")
    for i in range(n_iter):
        losses = {}
        # Batch-Größe wird progressiv erhöht
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        print(f"Iteration {i+1}/{n_iter}, Losses: {losses}")
    
    # Evaluation auf Testdaten
    accuracy = evaluate(nlp, test_data)
    print(f"\nTest Accuracy: {accuracy:.2f}")
    
    # Test auf Reddit-Daten (sofern verfügbar)
    test_on_reddit_data(nlp)
    
    # Optional: Speichern des feinabgestimmten spaCy-Modells
    # output_dir = os.path.join(base_dir, "models", "spacy_finetuned_model")
    # nlp.to_disk(output_dir)
    # print(f"Modell wurde unter {output_dir} gespeichert.")

if __name__ == "__main__":
    main()
