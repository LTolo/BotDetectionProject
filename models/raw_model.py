# Dieses Skript trainiert ein neuronales Netzwerk von Grund auf (raw model) zur Erkennung von Bot-Accounts in Twitter-Daten.
# Es lädt einen CSV-Datensatz, tokenisiert die Textdaten, trainiert ein Bidirectional LSTM-Modell und bewertet es.
# Optional testet es das Modell auf Daten, die über die Reddit API abgerufen wurden.
# Es verwendet TensorFlow/Keras für das Modelltraining und scikit-learn für die Evaluation.


import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
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

# Basisverzeichnis manuell festlegen (für Jupyter Notebook)
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))  # Geht ein Verzeichnis hoch.

def load_kaggle_data(file_path):
    """
    Lädt den Kaggle CSV-Datensatz.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"CSV-Daten geladen: {df.shape[0]} Zeilen.")
        return df
    except Exception as e:
        print(f"Fehler beim Laden der CSV-Datei: {e}")
        raise

def preprocess_data(df):
    """
    Verwendet die 'Tweet'-Spalte als Feature und die 'Bot Label'-Spalte als Ziel.
    """
    X = df["Tweet"].fillna("")
    y = df["Bot Label"]
    return X, y

def tokenize_and_pad(X_train, X_test, max_words=10000, max_len=100):
    """
    Tokenisiert den Text und wandelt ihn in gepaddete Sequenzen um.
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post", truncating="post")

    return X_train_pad, X_test_pad, tokenizer

def train_model(X_train_pad, y_train, max_words=10000, embedding_dim=16, max_len=100, epochs=10):
    """
    Erstellt und trainiert ein neuronales Netz, das sich in der Architektur an feingetunte Modelle anlehnt,
    aber als Raw-Modell von Grund auf trainiert wird.
    """
    model = Sequential([
        # Eingabe: Sequenzen mit Längen max_len
        Embedding(max_words, embedding_dim, input_length=max_len),
        # Hinzufügen eines Bidirectional LSTM für mehr Kontext-Erfassung (raw, aber komplexer)
        Bidirectional(LSTM(64, return_sequences=True)),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Trainiere das Modell ...")
    history = model.fit(X_train_pad, y_train, epochs=epochs, validation_split=0.2, verbose=2)
    print("Training abgeschlossen.")
    return model, history

def evaluate_model(model, X_test_pad, y_test, dataset_name="Testdaten"):
    """
    Bewertet das Modell und gibt Metriken aus.
    """
    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)

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

def test_on_reddit_data(model, tokenizer, max_len=100):
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
            X_group_seq = tokenizer.texts_to_sequences(X_group)
            X_group_pad = pad_sequences(X_group_seq, maxlen=max_len, padding="post", truncating="post")
            predictions_prob = model.predict(X_group_pad)
            predictions = (predictions_prob > 0.5).astype(int)
            for tweet, pred in zip(X_group, predictions):
                print(f"Text: {tweet[:50]}... -> Vorhersage: {pred[0]}")
    else:
        X_reddit = reddit_data["Tweet"].fillna("")
        X_reddit_seq = tokenizer.texts_to_sequences(X_reddit)
        X_reddit_pad = pad_sequences(X_reddit_seq, maxlen=max_len, padding="post", truncating="post")
        predictions_prob = model.predict(X_reddit_pad)
        predictions = (predictions_prob > 0.5).astype(int)
        for tweet, pred in zip(X_reddit, predictions):
            print(f"Text: {tweet[:50]}... -> Vorhersage: {pred[0]}")

def main():
    # Pfad zur Kaggle CSV-Datei (ausgehend vom Standort dieses Scripts)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kaggle_data_path = os.path.join(base_dir, "data", "twitter_dataset.csv")
    print(f"Lade Kaggle-Daten von: {kaggle_data_path}")

    df = load_kaggle_data(kaggle_data_path)
    X, y = preprocess_data(df)

    # Aufteilen in Trainings- und Testdaten (80/20-Aufteilung)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Texttokenisierung und Padding
    max_words = 10000
    max_len = 100
    X_train_pad, X_test_pad, tokenizer = tokenize_and_pad(X_train, X_test, max_words, max_len)

    # Modelltraining (raw Modell, nicht fine-getuned)
    model, history = train_model(X_train_pad, y_train, max_words, embedding_dim=16, max_len=max_len, epochs=10)

    # Evaluation auf Kaggle-Testdaten
    evaluate_model(model, X_test_pad, y_test, dataset_name="Kaggle Testdaten")

    # Test auf Reddit-Daten
    test_on_reddit_data(model, tokenizer, max_len)

    # Optional: Speichern des Modells (z. B. mit joblib oder model.save())
    # model.save(os.path.join(base_dir, "models", "raw_model.h5"))
    # print("Modell wurde gespeichert.")

if __name__ == "__main__":
    main()
