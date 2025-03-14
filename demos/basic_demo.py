import os
import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import numpy as np

def predict_tweet(model, tokenizer, text, max_length=128):
    """
    Encodiert den Text und gibt die Bot-Wahrscheinlichkeit zurück.
    """
    # Tokenisiere den Text
    encodings = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="tf"
    )
    # Erhalte die Logits vom Modell und wende Sigmoid an, um eine Wahrscheinlichkeit zu berechnen
    logits = model.predict(encodings).logits
    prob = tf.nn.sigmoid(logits).numpy()[0][0]
    return prob

def main():
    # Modell und Tokenizer laden (hier verwenden wir den gleichen Modellnamen, 
    # ersetze ihn ggf. durch den Pfad deines feinabgestimmten Modells)
    model_name = "bert-base-uncased"  # oder z.B. "path/to/your/fine_tuned_model"
    print("Lade Tokenizer und Modell ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    print("Modell und Tokenizer geladen.\n")

    print("Gib 'exit' ein, um das Programm zu beenden.\n")
    while True:
        tweet = input("Simulierter Tweet: ")
        if tweet.strip().lower() == 'exit':
            break
        note = input("Optional: Community Note (drücke Enter, wenn keine): ")
        if note.strip():
            augmented_text = f"{tweet} Note: {note}"
        else:
            augmented_text = tweet

        # Vorhersage berechnen
        probability = predict_tweet(model, tokenizer, augmented_text)
        predicted_class = "Bot" if probability > 0.5 else "Nicht Bot"
        print(f"Vorhersage: {predicted_class} (Wahrscheinlichkeit: {probability:.4f})\n")

if __name__ == "__main__":
    main()
