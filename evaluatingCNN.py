# cnn_train.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# 1. Konfiguration
DATA_DIR       = os.path.join("data", "recognition", "Train")  # Achte auf numerisch korrekt sortierte Ordner ("00","01",…)
INPUT_SIZE     = 64
BATCH_SIZE     = 32
EPOCHS         = 20
LR             = 0.001
NUM_CLASSES    = 43
TRAIN_FRACTION = 0.85
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Datenaugmentierung & Lade‐Pipeline
transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
total = len(dataset)
n_train = int(TRAIN_FRACTION * total)
n_val   = total - n_train
print(f"Gesamtbilder: {total}, Trainingsbilder: {n_train}, Validierungsbilder: {n_val}")

train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# 3. Modelldefinition (BetterCNN)
class BetterCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64,128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2)
        )
        feat_dim = 128 * (INPUT_SIZE // 8) * (INPUT_SIZE // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(feat_dim, 256), nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

model = BetterCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# 4. Training
print("Starte Training …")
for epoch in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0.0
    start = time.time()
    bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    for images, labels in bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    avg_train_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} – Train-Loss: {avg_train_loss:.4f} – Dauer: {time.time()-start:.1f}s")

# 5. Evaluation auf Validierungs-Set
print("Starte Evaluierung auf dem Validierungsset …")
model.eval()
val_loss = 0.0
correct  = 0
total    = 0
with torch.no_grad():
    bar = tqdm(val_loader, desc="Validierung", leave=False)
    for images, labels in bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
avg_val_loss = val_loss / len(val_loader)
accuracy     = correct / total * 100
print(f"Validation-Loss: {avg_val_loss:.4f} – Accuracy: {accuracy:.2f}%")
