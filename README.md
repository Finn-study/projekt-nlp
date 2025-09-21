# Aufgabenstellung 1: Sentimentanalyse von Produktrezensionen

Dieses Projekt (Modul „Projekt: NLP“) klassifiziert Amazon-Produktrezensionen automatisiert und sagt eine numerische Bewertung von 1–5 Sternen vorher. Zusätzlich wird eine 3-Klassen-Sentimentanalyse (negativ/neutral/positiv) durchgeführt.

## Features & Modelle

-   **Datensammlung & Sampling:** Erstellung einer Stichprobe aus drei Kategorien ("Automotive", "Books", "Video_Games") des [McAuley Lab Datensatzes (Amazon Review Data 2023)](https://amazon-reviews-2023.github.io/).
-   **Datenaufbereitung:** Vereinheitlichung, Filterung und Zusammenführung der Daten in eine einzige CSV-Datei (`final_dataset.csv`).
-   **Feature-Engineering:** Vorverarbeitung der Texte durch Tokenisierung, Entfernung von englischen Stoppwörtern, Lemmatisierung und anschließende Vektorisierung mittels TF-IDF.
-   **Modelle & Evaluation:**
    -   Multinomial Naive Bayes (5 Klassen & 3 Klassen)
    -   Neuronales Netz (DNN) (5 Klassen & 3 Klassen)
-   **Auswertung:** Berechnung der Genauigkeit (Accuracy) und Erstellung von Konfusionsmatrizen für jedes Modell.

## Projektstruktur

```
├── 01_prepare_data.py      # Skript zum Sampeln und Aufbereiten der Rohdaten
├── 02_models_train.py      # Hauptskript für Textvorverarbeitung, Modelltraining und Evaluation
├── requirements.txt        # Liste der benötigten Python-Bibliotheken
└── README.md               # Dieses Dokument
```

## Installation

**1. Repository klonen:**
```bash
git clone https://github.com/Finn-study/projekt-nlp
cd projekt-nlp
```

**2. Abhängigkeiten installieren:**
Es wird dringend empfohlen, eine virtuelle Umgebung zu verwenden.
```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Umgebung aktivieren
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Bibliotheken installieren
pip install -r requirements.txt
```

**3. NLTK-Ressourcen:**
Das Skript `02_models_train.py` lädt beim ersten Ausführen automatisch die notwendigen NLTK-Ressourcen (`punkt`, `wordnet`, `stopwords`) herunter.

## Ausführung

Die Ausführung erfolgt in zwei Schritten:

**Schritt 1: Daten vorbereiten**

1.  **Rohdaten herunterladen:** Lade die `.jsonl.gz`-Dateien für die Kategorien `Automotive`, `Books` und `Video_Games` von der [Amazon Review Data (2023)](https://amazon-reviews-2023.github.io/) Webseite herunter.
2.  **Pfade anpassen:** Öffne die Datei `01_prepare_data.py` und **passe die hartcodierten Pfade** zu den heruntergeladenen Dateien an deine lokale Ordnerstruktur an.
3.  **Skript ausführen:**
    ```bash
    python 01_prepare_data.py
    ```
    Dieses Skript erzeugt die Datei `final_dataset.csv`, die als Input für den nächsten Schritt dient.

### Schritt 2: Modelle trainieren & evaluieren

1.  **Pfad überprüfen:** Stelle sicher, dass der `file_path` in `02_models_train.py` auf deine erstellte `final_dataset.csv` verweist.
2.  **Skript ausführen:**
    ```bash
    python 02_models_train.py
    ```
    Das Skript führt den gesamten Analyseprozess aus, inklusive Training, Evaluation und der Anzeige der Konfusionsmatrizen.

## Technische Details
-   **Reproduzierbarkeit:** Die zufällige Stichprobe in `01_prepare_data.py` ist durch `random.seed(42)` reproduzierbar.
-   **Performance:** Die Anzahl der Features für TF-IDF ist auf `max_features=5000` begrenzt, um die Rechenzeit und den Speicherbedarf zu kontrollieren. Dieser Wert kann bei Bedarf angepasst werden.