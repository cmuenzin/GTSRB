# GTSRB – German Traffic Sign Recognition Benchmark

Ein Deep-Learning-Projekt zur Klassifikation deutscher Verkehrsschilder mithilfe von Convolutional Neural Networks (CNNs) auf Basis des GTSRB-Datensatzes.

## 🔍 Projektüberblick

Dieses Projekt implementiert ein CNN-Modell zur Bildklassifikation mit dem Ziel, deutsche Verkehrsschilder automatisiert zu erkennen. Der verwendete Datensatz ist der **German Traffic Sign Recognition Benchmark (GTSRB)** – ein standardisierter Datensatz für Forschung im Bereich Computer Vision im Kontext autonomer Fahrzeuge.

## 📦 Inhalt

- `train_model.py` – Trainingspipeline für das CNN-Modell
- `predict.py` – Vorhersage-Skript für Einzelbilder
- `utils.py` – Hilfsfunktionen (z. B. Datenvorverarbeitung)
- `load_data.py` – Funktionen zum Laden und Vorverarbeiten des GTSRB-Datensatzes
- `model.py` – Definition des CNN-Modells
- `requirements.txt` – Liste benötigter Python-Bibliotheken

## 🚀 Setup & Installation

### 1. Voraussetzungen

- Python ≥ 3.6
- Virtuelle Umgebung empfohlen (z. B. `venv` oder `conda`)

### 2. Repository klonen

```bash
git clone https://github.com/cmuenzin/GTSRB.git
cd GTSRB
