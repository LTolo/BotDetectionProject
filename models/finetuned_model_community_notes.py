# Dieses Skript führt ein Fine-Tuning eines vortrainierten BERT-Modells (bert-base-uncased) zur Erkennung von Bot-Accounts in Twitter-Daten durch,
# wobei Community Notes (zusätzliche Textinformationen) basierend auf dem Bot-Label hinzugefügt werden.
# Es lädt einen CSV-Datensatz und eine JSON-Datei mit Community Notes, teilt die Daten auf, tokenisiert die Texte,
# trainiert das Modell und bewertet es.
# Optional testet es das Modell auf Daten, die über die Reddit API abgerufen wurden.
# Es verwendet die Transformers-Bibliothek von Hugging Face für das Modell und TensorFlow für das Training.


import os
# Legacy-Modus aktivieren – muss vor jeglichem Import von TensorFlow/Keras erfolgen!
os.environ["TF_USE_LEGACY_KERAS"] = "True"

import sys
# Füge den übergeordneten Ordner zum PYTHONPATH hinzu, damit der Ordner "api" gefunden wird
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
import random
import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# Versuch, die Reddit API zu importieren
try:
    from api.reddit_api import get_reddit_data
    print("Reddit API erfolgreich importiert.")
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. "
          "Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None

# Basisverzeichnis 
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def load_community_notes(file_path):
    """
    Lädt die Community Notes aus einer JSON-Datei.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Fehler beim Laden der JSON-Datei: {e}")
        return None

def preprocess_data(df, community_notes):
    """
    Fügt Community Notes basierend auf dem Label ('Bot Label') hinzu.
    """
    X = df["Tweet"].fillna("")
    y = df["Bot Label"].values

    bot_notes = community_notes.get("bot_notes", [])
    non_bot_notes = community_notes.get("non_bot_notes", [])

    augmented_texts = []
    for text, label in zip(X, y):
        note = random.choice(bot_notes if label == 1 else non_bot_notes)
        augmented_texts.append(f"{text} Note: {note}")
    return augmented_texts, y

def encode_data(texts, tokenizer, max_length=128):
    """
    Encodiert die Texte mithilfe des Tokenizers und gibt ein TensorFlow-kompatibles Dictionary zurück.
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf"
    )
    return encodings

def create_tf_dataset(encodings, labels, batch_size=32, shuffle=True):
    """
    Erzeugt ein tf.data.Dataset aus den encodierten Texten und Labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(encodings), labels))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    return dataset

class TrainingAccuracyCallback(tf.keras.callbacks.Callback):
    def __init__(self, train_dataset, y_train):
        super().__init__()
        self.train_dataset = train_dataset
        self.y_train = y_train

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.train_dataset).logits
        # Flache beide Arrays, um sicherzugehen, dass die Formen übereinstimmen
        y_pred = (tf.nn.sigmoid(predictions).numpy().flatten() > 0.5).astype(int)
        y_true = self.y_train.flatten()
        acc = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch+1} Training Accuracy (Callback): {acc:.4f}")

def train_model(model, train_dataset, val_dataset, epochs=3, train_dataset_raw=None, y_train=None):
    """
    Feintunet das vortrainierte Modell und gibt die Trainingsgenauigkeit pro Epoche aus.
    """
    callbacks = []
    if train_dataset_raw is not None and y_train is not None:
        callbacks.append(TrainingAccuracyCallback(train_dataset_raw, y_train))
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
    return history

def evaluate_model(model, dataset, y_true):
    """
    Bewertet das Modell und gibt Metriken aus.
    """
    predictions = model.predict(dataset).logits
    y_pred = (tf.nn.sigmoid(predictions).numpy() > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred)
    
    print(f"\n--- Evaluation ---")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print("\nKlassifikationsbericht:\n", report)

def test_on_reddit_data(model, tokenizer, max_length=128):
    """
    Holt Daten über die Reddit API, encodiert diese und wendet das Modell an.
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
    texts = reddit_data["Tweet"].fillna("")
    encodings = encode_data(texts, tokenizer, max_length)
    dataset = tf.data.Dataset.from_tensor_slices(dict(encodings)).batch(32)
    predictions = model.predict(dataset).logits
    preds = (tf.nn.sigmoid(predictions).numpy() > 0.5).astype(int)
    for tweet, pred in zip(texts, preds):
        print(f"Text: {tweet[:50]}... -> Vorhersage: {pred[0]}")

def main():
    # Lade Kaggle-Daten
    kaggle_data_path = os.path.join(base_dir, "data", "twitter_dataset.csv")
    df = load_kaggle_data(kaggle_data_path)

    # Lade Community Notes
    community_notes_path = os.path.join(base_dir, "data", "community_notes.json")
    community_notes = load_community_notes(community_notes_path)

    # Preprocessing: Community Notes werden an die Tweets angehängt
    X, y = preprocess_data(df, community_notes)
    
    # Aufteilen in Trainings- und Testdaten (80/20-Aufteilung)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Verwende einen vortrainierten Tokenizer und das zugehörige Modell
    model_name = "bert-base-uncased"  # ggf. anpassen, z.B. "bert-base-german-cased" für Deutsch
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)  # Binary classification
    
    # Kompiliere das Modell mit dem Legacy-Optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    # Encodiere die Texte
    max_length = 128
    train_encodings = encode_data(X_train, tokenizer, max_length)
    test_encodings = encode_data(X_test, tokenizer, max_length)
    
    # Erstelle tf.data.Datasets
    batch_size = 32
    train_dataset = create_tf_dataset(train_encodings, y_train, batch_size=batch_size, shuffle=True)
    val_dataset = create_tf_dataset(test_encodings, y_test, batch_size=batch_size, shuffle=False)
    
    # Fine-Tuning des vortrainierten Modells inkl. Ausgabe der Trainingsgenauigkeit pro Epoche
    epochs = 3  # Für ein schnelles Fine-Tuning-Beispiel – ggf. erhöhen
    history = train_model(model, train_dataset, val_dataset, epochs=epochs,
                          train_dataset_raw=train_dataset, y_train=y_train)
    
    # Evaluation auf Testdaten
    evaluate_model(model, val_dataset, y_test)
    
    # Test auf Reddit-Daten
    test_on_reddit_data(model, tokenizer, max_length)

    # Optional: Modell speichern
    # model.save_pretrained(os.path.join(base_dir, "models", "finetuned_model"))
    # tokenizer.save_pretrained(os.path.join(base_dir, "models", "finetuned_model"))
    # print("Modell wurde gespeichert.")

if __name__ == "__main__":
    main()