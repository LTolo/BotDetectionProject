# Dieses Skript führt ein Fine-Tuning eines vortrainierten spaCy Transformer-Modells (en_core_web_trf) zur Erkennung von Bot-Accounts in Twitter-Daten durch.
# Es lädt einen CSV-Datensatz, konvertiert die Daten in das spaCy-Trainingsformat, teilt die Daten auf,
# trainiert das Modell und bewertet es.
# Optional testet es das Modell auf Daten, die über die Reddit API abgerufen wurden.
# Es verwendet spaCy für das Modelltraining und scikit-learn für die Datenaufteilung.


import os
import sys
# Füge den übergeordneten Ordner zum PYTHONPATH hinzu, damit der Ordner "api" gefunden wird
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import random
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import time

# Falls vorhanden, versuche die Reddit API zu importieren
try:
    from api.reddit_api import get_reddit_data
    print("Reddit API erfolgreich importiert.")
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. "
          "Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None

# --- Optional: Monkey-Patch für BPE-Merges, falls erforderlich ---
import spacy_curated_transformers.tokenization.hf_loader as hf_loader
original_convert = hf_loader._convert_byte_bpe_encoder

def fixed_convert_byte_bpe_encoder(model, tokenizer):
    vocab_merges = tokenizer.vocab_merges if hasattr(tokenizer, "vocab_merges") else {}
    if "merges" in vocab_merges:
        merges = [
            tuple(merge) if isinstance(merge, list) else tuple(merge.split(" "))
            for merge in vocab_merges["merges"]
        ]
        tokenizer.vocab_merges["merges"] = merges
    return original_convert(model, tokenizer)

hf_loader._convert_byte_bpe_encoder = fixed_convert_byte_bpe_encoder
# --- Ende Monkey-Patch ---

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
        # Definiere die Kategorien: True für BOT, False für REAL
        cats = {"BOT": label == 1, "REAL": label == 0}
        data.append((tweet, {"cats": cats}))
    print(f"{len(data)} Trainingsbeispiele geladen.")
    return data

def evaluate(nlp, data):
    """
    Berechnet die Accuracy, indem für jeden Text die Kategorie mit dem höheren Score als Vorhersage genutzt wird.
    """
    correct = 0
    total = 0
    for text, annotations in data:
        doc = nlp(text)
        pred_label = "BOT" if doc.cats["BOT"] >= doc.cats["REAL"] else "REAL"
        true_label = "BOT" if annotations["cats"]["BOT"] else "REAL"
        if pred_label == true_label:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def test_on_reddit_data(nlp):
    """
    Holt Daten über die Reddit API (sofern verfügbar) und wendet das feinabgestimmte Modell an.
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
    
    # Option: Verwende nur einen Teil der Trainingsdaten (z.B. 10%)
    subset_ratio = 0.1
    data = data[: int(len(data) * subset_ratio)]
    print(f"Verwende {len(data)} Trainingsbeispiele (Subset) für schnelles Training.")
    
    random.shuffle(data)
    
    # Aufteilen in Trainings- und Testdaten (z.B. 80/20-Aufteilung)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Lade das vortrainierte spaCy Transformer-Modell (z.B. en_core_web_trf)
    print("Lade Modell en_core_web_trf ...")
    nlp = spacy.load("en_core_web_trf")
    
    # Falls noch nicht vorhanden, füge den Textklassifikator hinzu
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat", last=True)
    else:
        textcat = nlp.get_pipe("textcat")
    
    # Füge die Labels hinzu
    textcat.add_label("BOT")
    textcat.add_label("REAL")
    
    # Initialisiere das Modell neu
    print("Initialisiere Modell ...")
    nlp.initialize()
    
    # Option: Transformer einfrieren, damit nur der Textklassifikator trainiert wird.
    # Das kann das Training deutlich beschleunigen, wenn du bereits gute Sprachrepräsentationen hast.
    try:
        transformer = nlp.get_pipe("transformer")
        transformer.model.freeze()
        print("Transformer-Komponente wurde eingefroren.")
    except Exception as e:
        print("Fehler beim Einfrieren des Transformers:", e)
    
    # Optimizer einmal initialisieren
    optimizer = nlp.resume_training()
    
    # Trainingsschleife mit Fortschrittsanzeige
    print("Beginne Training ...")
    n_iter = 3  # Reduziere die Anzahl der Iterationen/Epochen für schnelles Training
    batch_size = 64  # Falls der Arbeitsspeicher es zulässt, erhöhe die Batchgröße
    for i in range(n_iter):
        losses = {}
        start_iter = time.time()
        batches = minibatch(train_data, size=compounding(4.0, batch_size, 1.5))
        batch_count = 0
        for batch in batches:
            examples = []
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
            batch_count += 1
            sys.stdout.write(f"\rIteration {i+1}, Batch {batch_count} processed...")
            sys.stdout.flush()
        end_iter = time.time()
        print(f"\nIteration {i+1} abgeschlossen in {end_iter - start_iter:.2f} Sekunden, Losses: {losses}")
        train_accuracy = evaluate(nlp, train_data)
        print(f"Iteration {i+1} Training Accuracy: {train_accuracy:.2f}")
    
    # Evaluation auf den Testdaten
    test_acc = evaluate(nlp, test_data)
    print(f"\nTest Accuracy: {test_acc:.2f}")
    
    # Test auf Reddit-Daten (sofern verfügbar)
    test_on_reddit_data(nlp)
    
    # Optional: Speichern des feinabgestimmten Modells
    # output_dir = os.path.join(base_dir, "models", "spacy_finetuned_model")
    # nlp.to_disk(output_dir)
    # print(f"Modell wurde unter {output_dir} gespeichert.")

if __name__ == "__main__":
    main()