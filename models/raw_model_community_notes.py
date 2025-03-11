import os
import sys
import json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)

# Versuch, die Reddit API zu importieren
try:
    from api.reddit_api import get_reddit_data
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. "
          "Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None


def load_kaggle_data(file_path, community_notes_path=None):
    """
    Lädt den Kaggle CSV-Datensatz und optional die Community Notes aus einer JSON-Datei.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV-Daten geladen: {df.shape[0]} Zeilen.")

        if community_notes_path:
            try:
                with open(community_notes_path, 'r', encoding='utf-8') as f:
                    community_notes = json.load(f)

                # Füge Community Notes als neue Spalte hinzu
                df['Community Notes'] = df['Tweet ID'].map(
                    lambda tweet_id: community_notes.get(str(tweet_id), "")
                )
                
                # Kombiniere Tweets und Community Notes
                df['Combined Text'] = df['Tweet'].fillna("") + " " + df['Community Notes'].fillna("")
                print("Community Notes erfolgreich hinzugefügt.")
            except FileNotFoundError:
                print(f"Warnung: Community Notes-Datei nicht gefunden: {community_notes_path}")
            except json.JSONDecodeError:
                print(f"Warnung: Fehler beim Lesen der Community Notes-Datei: {community_notes_path}")

        return df
    except Exception as e:
        print(f"Fehler beim Laden der CSV-Datei: {e}")
        raise


def preprocess_data(df, use_combined_text=True):
    """
    Verwendet die 'Tweet'-Spalte oder 'Combined Text' als Feature und die 'Bot Label'-Spalte als Ziel.
    Zusätzliche Vorverarbeitung (wie Textbereinigung) kann hier ergänzt werden.
    """
    if use_combined_text and 'Combined Text' in df.columns:
        X = df["Combined Text"].fillna("")
    else:
        X = df["Tweet"].fillna("")
    y = df["Bot Label"]
    return X, y


def train_model(X_train, y_train):
    """
    Erstellt und trainiert eine Pipeline, die den TfidfVectorizer und
    eine Logistic Regression kombiniert.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("clf", LogisticRegression(solver="liblinear", max_iter=1000))
    ])

    print("Trainiere das Modell ...")
    pipeline.fit(X_train, y_train)
    print("Training abgeschlossen.")
    return pipeline


def evaluate_model(model, X_test, y_test, dataset_name="Testdaten"):
    """
    Bewertet das Modell und gibt Metriken wie Accuracy, F1, Precision und Recall aus.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(f"\n--- Evaluation auf {dataset_name} ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("\nKlassifikationsbericht:\n", report)


def test_on_reddit_data(model):
    """
    Holt Daten über die Reddit API und wendet das Modell an.
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

    if "source" in reddit_data.columns:
        for source, group in reddit_data.groupby("source"):
            if source == 'new':
                header = "Die 10 neuesten Beiträge (.new)"
            elif source == 'hot':
                header = "Die 10 heißesten Beiträge (.hot)"
            else:
                header = f"Beiträge aus '{source}'"
            print(f"\n--- {header} ---")
            X_group = group["Tweet"].fillna("")
            predictions = model.predict(X_group)
            for tweet, pred in zip(X_group, predictions):
                print(f"Text: {tweet[:50]}... -> Vorhersage: {pred}")
    else:
        X_reddit = reddit_data["Tweet"].fillna("")
        predictions = model.predict(X_reddit)
        for tweet, pred in zip(X_reddit, predictions):
            print(f"Text: {tweet[:50]}... -> Vorhersage: {pred}")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_data_path = os.path.join(base_dir, "data", "twitter_dataset.csv")
    community_notes_path = os.path.join(base_dir, "data", "community_notes.json")
    print(f"Lade Kaggle-Daten von: {kaggle_data_path}")

    df = load_kaggle_data(kaggle_data_path, community_notes_path)
    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test, dataset_name="Kaggle Testdaten")

    test_on_reddit_data(model)

    # Optional: Speichern des Modells (z. B. mit joblib)
    # from joblib import dump
    # dump(model, os.path.join(base_dir, "models", "raw_model.joblib"))
    # print("Modell wurde gespeichert.")


if __name__ == "__main__":
    main()