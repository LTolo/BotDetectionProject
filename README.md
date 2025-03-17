# Bot-Erkennung in sozialen Netzwerken

## Projektbeschreibung

Dieses Projekt konzentriert sich auf die Entwicklung everschiedener Modelle zur Erkennung von Bot-Accounts in sozialen Netzwerken. Ziel ist es, Modelle zu trainieren und sie zu vergleichen, die zwischen echten Nutzern und Bot-Accounts unterscheidet. Dies passiert indem dei Modelle öffentlich zugängliche Daten als Testdatensatz (Reddit-Posts) und einen Trainingsdatensatz (Kaggle Twitter-Datensatz) verwenden.
Zusätzlich wird eine Web-Demo bereitgestellt, die den Einsatz der trainierten Modelle in einer Webanwendung demonstriert.

## Ziele

*   **Kaggle Twitter-Datensatz als Trainingsdatensatz:** Verwendung des Kaggle Twitter-Datensatzes als Grundlage für das Modelltraining.
*   **Reddit API als Testdatensatz:** Nutzung der Reddit API zum Abrufen von Daten für die Validierung des Modells in einer realen Umgebung.
*   **Entwicklung verschiedener Modelle:**
    *   **Raw Modell:** Ein einfaches Modell ohne spezielle Vorverarbeitung.
    *   **Feingetuntes Modell mit BERT:** Ein verfeinertes Modell mit optimierten Hyperparametern.
    *   **Feingetuntes Modell mit BERT und Community Notes:** Ein verfeinertes Modell mit optimierten Hyperparametern.
    *   **Feingetuntes Modell mit SpaCy:** Ein Modell, das SpaCy für die Textvorverarbeitung nutzt.

## Modellentwicklung

### Raw Modell

Das Raw Modell dient als Basismodell und wird ohne besondere Vorverarbeitung trainiert. Es verwendet grundlegende Textmerkmale wie Wortfrequenzen.

### Feingetuntes Modell mit BERT

Dieses Modell basiert auf einem BERT Modell und wird durch Hyperparameter-Optimierung und Feature Engineering verfeinert.

### Feingetuntes Modell mit BERT und Community Notes

Dieses Modell basiert auf einem BERT Modell und wird durch Hyperparameter-Optimierung und Feature Engineering verfeinert. Zusätzlich nutzt es Community Notes als zusätzliche Textinformation.

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
    *   Training des Raw Modells.
    *   Feintuning mit den beiden BERT sowie eines SpaCy Modells.

4.  **Evaluation und Test:**
    *   Bewertung der Modelle anhand von Metriken wie Präzision, Recall und F1-Score.
    *   Vergleich der Leistung von den Modellen.

5. **Bereitstellung von Demos:**
    *   Implementierung einer Basic Demo, bei der man das Prinzip der Anwendung im Terminal testet.
    *   Eine Flask-Demo, die den Einsatz der trainierten Modelle visualisiert und interaktiv demonstriert.



## Zusammenfassung der Modelle

*   **Raw Modell:** Basismodell ohne SpaCy.
*   **Feingetuntes Modell mit BERT:** Verbessertes Modell mit optimierten Parametern.
*   **Feingetuntes Modell mit BERT und Community Notes:** Verbessertes Modell mit optimierten Parametern.
*   **Feingetuntes Modell mit SpaCy:** Modell mit zusätzlicher SpaCy-Verarbeitung.

## Demos und Integration
**Neben der reinen Modellentwicklung enthält dieses Repository auch Demo-Anwendungen die zeigen:**
*   **Wie die Modelle in einer Webanwendung (basierend auf Flask) eingebunden werden können.**
*   **Wie Reddit-Daten über die API abgerufen und zur Vorhersage genutzt werden.**

## Fazit

Dieses Projekt zielt darauf ab, eine effektive Methode zur Bot-Erkennung in sozialen Netzwerken zu entwickeln. Durch den Vergleich verschiedener Modelle wird der Einfluss von Vorverarbeitungstechniken und Modell-Tuning untersucht. Das Ziel ist ein möglichst robustes Modell sowie eine Demo, welche Bot-Interaktionen von echten Nutzern unterscheiden kann und demonstriert.

## Installation und Nutzung

1.  **Repository klonen:** `git clone <repository_url>`
2.  **Abhängigkeiten installieren:** `pip install -r requirements.txt`
3.  **Kaggle-Datensatz herunterladen:** (https://www.kaggle.com/datasets/goyaladi/twitter-bot-detection-dataset?resource=download)
4.  **Reddit API-Zugangsdaten konfigurieren**
5.  **Modelle trainieren**
6.  **Demo Starten:** `python demo/app.py`

