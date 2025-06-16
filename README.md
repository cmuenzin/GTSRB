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

3. Abhängigkeiten installieren
bash
Kopieren
Bearbeiten
pip install -r requirements.txt
4. GTSRB-Datensatz herunterladen
Lade den Datensatz von GTSRB Download-Seite herunter und entpacke ihn in das Projektverzeichnis:

Kopieren
Bearbeiten
GTSRB/
└── Final_Training/
    └── Images/
🏋️‍♂️ Modell trainieren
bash
Kopieren
Bearbeiten
python train_model.py
Optional: Hyperparameter und Pfade können direkt im Skript angepasst werden.

📈 Modell bewerten
Während des Trainings werden Metriken wie Genauigkeit und Verlust ausgegeben. Zusätzlich können Validierungsdaten verwendet werden, um das Modellverhalten zu überwachen.

🔮 Einzelbild-Vorhersage
bash
Kopieren
Bearbeiten
python predict.py path_to_image.jpg
Das Skript gibt die erkannte Verkehrsschild-Klasse und den zugehörigen Labelnamen aus.

🧠 Modellarchitektur
Das CNN besteht aus mehreren Schichten:

Conv2D → ReLU → MaxPooling

Dropout zur Regularisierung

Dense-Schichten mit Softmax-Ausgabe für Klassifikation in 43 Klassen

📊 Ergebnisse
Das Modell erreicht auf dem GTSRB-Datensatz eine Genauigkeit von über 95 % (je nach Hyperparameterwahl und Datenaugmentierung).

✅ ToDo / Weiterentwicklung
 Integration von Datenaugmentation

 Export als .onnx oder .tflite für Edge Deployment

 Live-Feed-Integration mit Webcam / Dashcam

 Optimierung der Trainingsdauer durch Transfer Learning

🤝 Mitwirken
Pull Requests und Feature-Vorschläge sind willkommen. Bitte stelle sicher, dass du sauberen, dokumentierten Code einreichst und dich an bestehende Projektstandards hältst.

📄 Lizenz
Dieses Projekt steht unter der MIT License. Weitere Informationen siehe LICENSE.

📚 Quellen
GTSRB Dataset: https://benchmark.ini.rub.de/

TensorFlow/Keras Dokumentation

Paper: The German Traffic Sign Recognition Benchmark: A multi-class classification competition (Stallkamp et al., 2012)

