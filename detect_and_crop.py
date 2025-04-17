# detect_and_crop.py
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO 
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Pfadkonfiguration
test_images_path = os.path.join("data", "recognition", "Test")
yolo_model_path = r"C:\Users\casam\Documents\06_Coding\Python\MachineLearning\01_roadSignsProject\runs\detect\train4\weights\best.pt"
cnn_model_path = os.path.join("models", "gtsrb_cnn.pt")

# Fest codiertes Mapping – gewünschte Zuordnung
classes_mapping = {
    0: 'Speed limit (20km/h)',
    1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)',
    5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)',
    7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)',
    9: 'No passing',
    10: 'No passing veh over 3.5 tons',
    11: 'Right-of-way at intersection',
    12: 'Priority road',
    13: 'Yield',
    14: 'Stop',
    15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited',
    17: 'No entry',
    18: 'General caution',
    19: 'Dangerous curve left',
    20: 'Dangerous curve right',
    21: 'Double curve',
    22: 'Bumpy road',
    23: 'Slippery road',
    24: 'Road narrows on the right',
    25: 'Road work',
    26: 'Traffic signals',
    27: 'Pedestrians',
    28: 'Children crossing',
    29: 'Bicycles crossing',
    30: 'Beware of ice/snow',
    31: 'Wild animals crossing',
    32: 'End speed + passing limits',
    33: 'Turn right ahead',
    34: 'Turn left ahead',
    35: 'Ahead only',
    36: 'Go straight or right',
    37: 'Go straight or left',
    38: 'Keep right',
    39: 'Keep left',
    40: 'Roundabout mandatory',
    41: 'End of no passing',
    42: 'End no passing veh > 3.5 tons'
}
print("Verwendetes Mapping:", classes_mapping)

input_size = 64  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Definition der gleichen BetterCNN-Architektur wie im Training
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

cnn_model = BetterCNN(num_classes=43).to(device)
cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))
cnn_model.eval()

# Transformation für den CNN-Eingang (keine Augmentation im Inferenzmodus)
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

yolo_model = YOLO(yolo_model_path)

def detect_and_classify(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler beim Laden von {image_path}")
        return None
    annotated_image = image.copy()
    results = yolo_model(image_path)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else []
    print(f"Anzahl erkannter Objekte: {len(detections)}")
    for box in detections:
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = cnn_model(crop_tensor)
            pred = output.argmax(dim=1).item()
            label = classes_mapping.get(pred, "Unbekannt")
            print(f"Voraussage: {pred} -> {label}")
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(annotated_image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated_image

if __name__ == "__main__":
    test_images = [os.path.join(test_images_path, f) for f in os.listdir(test_images_path) if f.lower().endswith(".jpg")]
    if not test_images:
        print("Keine Testbilder gefunden.")
    else:
        import random
        test_img_path = random.choice(test_images)
        print(f"Bearbeite {test_img_path}")
        output_img = detect_and_classify(test_img_path)
        if output_img is not None:
            out_path = os.path.join("models", "annotated_result.jpg")
            cv2.imwrite(out_path, output_img)
            print(f"Annotiertes Bild gespeichert unter {out_path}")
