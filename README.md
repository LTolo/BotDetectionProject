# Bot-Erkennung in sozialen Netzwerken

## Projektbeschreibung

Dieses Projekt konzentriert sich auf die Entwicklung eines Modells zur Erkennung von Bot-Accounts in sozialen Netzwerken. Ziel ist es, ein Modell zu trainieren, das zwischen echten Nutzern und Bot-Accounts unterscheidet, indem es öffentlich zugängliche Daten (Reddit-Posts) und einen Trainingsdatensatz (Kaggle Twitter-Datensatz) verwendet.

## Ziele

*   **Kaggle Twitter-Datensatz als Trainingsdatensatz:** Verwendung des Kaggle Twitter-Datensatzes als Grundlage für das Modelltraining.
*   **Reddit API als Testdatensatz:** Nutzung der Reddit API zum Abrufen von Daten für die Validierung des Modells in einer realen Umgebung.
*   **SpaCy für Datenvorverarbeitung:** Einsatz von SpaCy für Named Entity Recognition (NER) und Tokenisierung zur Verbesserung der Textdaten.
*   **Entwicklung verschiedener Modelle:**
    *   **Raw Modell:** Ein einfaches Modell ohne spezielle Vorverarbeitung.
    *   **Feingetuntes Modell:** Ein verfeinertes Modell mit optimierten Hyperparametern.
    *   **Feingetuntes Modell mit SpaCy:** Ein Modell, das SpaCy für die Textvorverarbeitung nutzt.

## Modellentwicklung

### Raw Modell

Das Raw Modell dient als Basismodell und wird ohne besondere Vorverarbeitung trainiert. Es verwendet grundlegende Textmerkmale wie Wortfrequenzen.

### Feingetuntes Modell

Dieses Modell basiert auf dem Raw Modell und wird durch Hyperparameter-Optimierung und Feature Engineering verfeinert.

### Feingetuntes Modell mit SpaCy

Dieses Modell verwendet SpaCy für die Textvorverarbeitung und Feature-Extraktion, um die Leistung des Modells zu verbessern.

## Projektschritte

1.  **Datenextraktion:**
    *   **Twitter-Datensatz:** Verwendung des Kaggle Twitter-Datensatzes für das Training.
    *   **Reddit-Daten:** Extraktion von Posts und Kommentaren über die Reddit API.

2.  **Vorverarbeitung:**
    *   Normalisierung, Tokenisierung und NER mit SpaCy.
    *   Umwandlung von Textdaten in ein für das Modell geeignetes Format (z.B. TF-IDF, Word2Vec).

3.  **Modelltraining und Feintuning:**
    *   Training des Raw Modells und des feingetunten Modells.
    *   Feintuning mit und ohne SpaCy.

4.  **Evaluation und Test:**
    *   Bewertung der Modelle anhand von Metriken wie Präzision, Recall und F1-Score.
    *   Vergleich der Leistung von Modellen mit und ohne SpaCy.

5.  **Dashboard:**
    *   Erstellung eines Dashboards (z.B. mit Streamlit) zur Visualisierung der Ergebnisse.
    *   Anzeige der wichtigsten Merkmale erkannter Bots.

## Zusammenfassung der Modelle

*   **Raw Modell:** Basismodell ohne SpaCy.
*   **Feingetuntes Modell:** Verbessertes Modell mit optimierten Parametern.
*   **Feingetuntes Modell mit SpaCy:** Modell mit zusätzlicher SpaCy-Verarbeitung.

## Fazit

Dieses Projekt zielt darauf ab, eine effektive Methode zur Bot-Erkennung in sozialen Netzwerken zu entwickeln. Durch den Vergleich verschiedener Modelle wird der Einfluss von Vorverarbeitungstechniken und Modell-Tuning untersucht. Das Ziel ist ein robustes Modell, das Bot-Interaktionen von echten Nutzern unterscheiden kann.

## Installation und Nutzung

1.  **Repository klonen:** `git clone <repository_url>`
2.  **Abhängigkeiten installieren:** `pip install -r requirements.txt`
3.  **Kaggle-Datensatz herunterladen:** (https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset?resource=download)
4.  **Reddit API-Zugangsdaten konfigurieren:**
5.  **Modelle trainieren:** (Zuerst alle Modelle vollständig treinieren bevor das Dashboard gestartet wird)
6.  **Dashboard starten:**

