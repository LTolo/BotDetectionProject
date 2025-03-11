import os
import sys
import random
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Concatenate, Embedding, LSTM,
                                    GlobalAveragePooling1D, Dropout, Bidirectional)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import TextVectorization, Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Pfad-Anpassung
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, "api"))

# Reddit API Import
try:
    from api.reddit_api import get_reddit_data
except ImportError:
    print("Warnung: reddit_api konnte nicht importiert werden. Stelle sicher, dass der Ordner 'api' im PYTHONPATH ist.")
    get_reddit_data = None


def load_community_notes(file_path):
    """Lädt die Community Notes aus der JSON-Datei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notes = json.load(f)
        return notes['bot_notes'], notes['non_bot_notes']
    except Exception as e:
        print(f"Fehler beim Laden der Community Notes: {e}")
        return [], []


def assign_community_note(label, bot_notes, non_bot_notes):
    """Weist eine Community Note basierend auf dem Label zu."""
    if random.random() < 0.05:
        all_notes = bot_notes + non_bot_notes
        return random.choice(all_notes)
    else:
        return random.choice(bot_notes) if label == 1 else random.choice(non_bot_notes)


def load_and_preprocess_data(data_path, notes_file_path):
    """Lädt und verarbeitet die Daten aus der CSV-Datei."""
    df = pd.read_csv(data_path)
    df["Tweet"] = df["Tweet"].astype(str)
    df["Bot Label"] = df["Bot Label"].astype(int)

    # Community Notes laden
    bot_notes, non_bot_notes = load_community_notes(notes_file_path)
    df["community_notes"] = df["Bot Label"].apply(lambda x: assign_community_note(x, bot_notes, non_bot_notes))

    df["cleaned_Tweet"] = df["Tweet"]
    numeric_features = ["Retweet Count", "Mention Count", "Follower Count", "Verified"]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_text = train_df["cleaned_Tweet"].values
    test_text = test_df["cleaned_Tweet"].values

    train_comm = train_df["community_notes"].values
    test_comm = test_df["community_notes"].values

    train_numeric = train_df[numeric_features].values.astype("float32")
    test_numeric = test_df[numeric_features].values.astype("float32")

    y_train = train_df["Bot Label"].values
    y_test = test_df["Bot Label"].values

    return train_text, test_text, train_comm, test_comm, train_numeric, test_numeric, y_train, y_test, numeric_features


def build_and_train_model(train_text, train_comm, train_numeric, y_train, numeric_features):
    """Erstellt und trainiert das TensorFlow-Modell."""
    max_tokens = 10000
    output_sequence_length = 150
    text_vectorizer = TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=output_sequence_length)
    text_vectorizer.adapt(train_text)

    normalizer = Normalization(axis=-1)
    normalizer.adapt(train_numeric)

    community_vectorizer = TextVectorization(max_tokens=5000, output_mode="int", output_sequence_length=50)
    community_vectorizer.adapt(train_comm)

    text_input = Input(shape=(1,), dtype=tf.string, name="text_input")
    x = text_vectorizer(text_input)
    x = Embedding(input_dim=max_tokens, output_dim=128, mask_zero=True)(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.4)(x)

    numeric_input = Input(shape=(len(numeric_features),), name="numeric_input")
    norm_numeric = normalizer(numeric_input)

    community_input = Input(shape=(1,), dtype=tf.string, name="community_input")
    comm = community_vectorizer(community_input)
    comm = Embedding(input_dim=5000, output_dim=32, mask_zero=True)(comm)
    comm = GlobalAveragePooling1D()(comm)
    comm = Dense(32, activation="relu")(comm)

    combined = Concatenate()([x, norm_numeric, comm])
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.4)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.4)(z)
    output = Dense(1, activation="sigmoid")(z)

    model = Model(inputs=[text_input, numeric_input, community_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    batch_size = 32
    epochs = 100
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        {"text_input": train_text, "numeric_input": train_numeric, "community_input": train_comm},
        y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )
    return model


def evaluate_model(model, test_text, test_comm, test_numeric, y_test):
    """Bewertet das Modell auf den Testdaten."""
    loss, test_acc = model.evaluate(
        {"text_input": test_text, "numeric_input": test_numeric, "community_input": test_comm},
        y_test
    )
    print(f"Test Accuracy: {test_acc:.4f}")

    predictions = model.predict(
        {"text_input": test_text, "numeric_input": test_numeric, "community_input": test_comm})
    predicted_labels = (predictions > 0.5).astype(int)

    for i in range(5):
        print("Cleaned Tweet:", test_text[i])
        print("Community Note:", test_comm[i])
        print("Predicted Label:", predicted_labels[i][0], "Actual Label:", y_test[i])
        print("---")


def main():
    # Pfade anpassen
    data_path = os.path.join(base_dir, "data", "twitter_dataset.csv")
    notes_file_path = os.path.join(base_dir, "data", "community_notes.json")

    # Daten laden und verarbeiten
    train_text, test_text, train_comm, test_comm, train_numeric, test_numeric, y_train, y_test, numeric_features = load_and_preprocess_data(data_path, notes_file_path)

    # Modell erstellen und trainieren
    model = build_and_train_model(train_text, train_comm, train_numeric, y_train, numeric_features)

    # Modell bewerten
    evaluate_model(model, test_text, test_comm, test_numeric, y_test)


if __name__ == "__main__":
    main()