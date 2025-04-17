# cnn_train.py
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm  # Fortschrittsanzeige

# Parameter
data_dir = os.path.join("data", "recognition", "Train")  # Achte darauf: Ordner sollten "00", "01", ... etc. heißen!
model_save_path = os.path.join("models", "gtsrb_cnn.pt")
input_size = 64  # Bildgröße: 64x64
batch_size = 32
epochs = 20     # Training über 20 Epochen
learning_rate = 0.001
num_classes = 43

# Datenvorverarbeitung mit erweiterter, moderater Datenaugmentation
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Datensatz laden – Ordnerstruktur: data/recognition/Train/<class>/image.jpg
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
total_samples = len(dataset)
subset_size = int(0.5 * total_samples)  # Nutze 50% der Daten
print(f"Gesamtanzahl der Trainingsbilder: {total_samples}, nutze {subset_size} Bilder")

# Zufällige Auswahl von 50% der Daten
indices = list(range(total_samples))
random.shuffle(indices)
subset_indices = indices[:subset_size]
subset_dataset = Subset(dataset, subset_indices)

# Aufteilen in Trainings- und Validierungs-Sets (80% Training, 20% Validierung)
num_train = int(0.8 * len(subset_dataset))
num_val = len(subset_dataset) - num_train
train_dataset, val_dataset = random_split(subset_dataset, [num_train, num_val])
print(f"Trainingsbilder: {num_train}, Validierungsbilder: {num_val}")

# Optional: Auswertung der Klassenverteilung im Trainingsset
train_indices = train_dataset.indices
train_targets = [dataset.targets[i] for i in train_indices]
class_counts = np.bincount(train_targets, minlength=num_classes)
print("Klassenhäufigkeiten im Trainingsset:", class_counts)

# Erstelle Gewichte für einen WeightedRandomSampler (falls gewünscht)
class_weights = 1. / (class_counts + 1e-6)
samples_weight = [class_weights[t] for t in train_targets]
samples_weight = torch.DoubleTensor(samples_weight)
weighted_sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

# DataLoader mit WeightedRandomSampler
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=weighted_sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Definition des komplexeren CNN-Modells – BetterCNN
class BetterCNN(nn.Module):
    def __init__(self, num_classes=43):
        super(BetterCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128 * (input_size // 8) * (input_size // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BetterCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Starte Training ...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = running_loss / len(train_loader)
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch + 1}/{epochs} - Trainings-Loss: {avg_loss:.4f} - Dauer: {epoch_time:.2f}s")

print("Starte Evaluierung auf dem Validierungsset ...")
model.eval()
correct = 0
total = 0
val_loss = 0.0
with torch.no_grad():
    progress_bar = tqdm(val_loader, desc="Evaluierung", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
avg_val_loss = val_loss / len(val_loader)
accuracy = correct / total * 100
print(f"Validierungsdaten: Loss = {avg_val_loss:.4f}, Accuracy = {accuracy:.2f}%")

os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"Modell gespeichert als {model_save_path}")
