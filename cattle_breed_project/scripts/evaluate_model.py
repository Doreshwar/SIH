import os
import random
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image

# === Custom Dataset Class for Hierarchical Structure ===
class HierarchicalCattleDataset(Dataset):
    def __init__(self, data_dir, transform=None, label2idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label2idx = label2idx or {}
        self.samples = []
        self.class_names = []
        
        # Load samples from hierarchical structure
        for species_dir in ['CATTLE', 'BUFFALO']:
            species_path = os.path.join(data_dir, species_dir)
            if not os.path.exists(species_path):
                continue
                
            for breed_folder in os.listdir(species_path):
                breed_path = os.path.join(species_path, breed_folder)
                if not os.path.isdir(breed_path):
                    continue
                    
                # Extract breed name from folder name (e.g., "INDIA_BUFFALO_0400_BANNI_01011" -> "banni")
                breed_name = self._extract_breed_name(breed_folder)
                if breed_name not in self.label2idx:
                    print(f"⚠️ Breed '{breed_name}' not found in model classes, skipping")
                    continue
                    
                # Add all images from this breed folder
                for img_file in os.listdir(breed_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(breed_path, img_file)
                        self.samples.append((img_path, self.label2idx[breed_name]))
                        
        print(f"Loaded {len(self.samples)} samples")
        
    def _extract_breed_name(self, folder_name):
        """Extract breed name from folder name like INDIA_BUFFALO_0400_BANNI_01011"""
        parts = folder_name.split('_')
        if len(parts) >= 4:
            return parts[3].lower()  # Extract breed name and convert to lowercase
        return folder_name.lower()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCattleClassifier, self).__init__()
        
        # Use EfficientNet-B2 for better efficiency
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "cleaned")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "..", "annotations")
MODEL_PATH = os.path.join(BASE_DIR, "Best_model_final.pth")  # Updated path
REPORT_PATH = os.path.join(ANNOTATIONS_DIR, "evaluation_report.csv")
PREDICTIONS_PATH = os.path.join(ANNOTATIONS_DIR, "sample_predictions.csv")
CM_COUNTS_PATH = os.path.join(ANNOTATIONS_DIR, "confusion_matrix_counts.png")
CM_PERCENT_PATH = os.path.join(ANNOTATIONS_DIR, "confusion_matrix_percent.png")

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

# === Data transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load model first to get class information ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found at: {MODEL_PATH}")
    print("Please train the model first using train_cleaned_model.py")
    exit(1)

# Load checkpoint
print(f"Loading model from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

# Extract information from checkpoint
num_classes = checkpoint['num_classes']
label2idx = checkpoint['label2idx']
breeds = checkpoint['breeds']

print(f"Model info: {num_classes} classes - {breeds}")

# Initialize model with correct architecture
model = ImprovedCattleClassifier(num_classes=num_classes, dropout_rate=0.3)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ Model loaded successfully! Best validation accuracy was: {checkpoint['best_val_acc']:.4f}")

# === Dataset with hierarchical structure ===
# Use custom dataset that handles CATTLE/BUFFALO hierarchy
dataset_hierarchical = HierarchicalCattleDataset(DATA_DIR, transform=transform, label2idx=label2idx)

if len(dataset_hierarchical) == 0:
    print("❌ No valid samples found in dataset. Check breed name matching.")
    exit(1)

print(f"Found {len(dataset_hierarchical)} valid samples across {len(set(label2idx.values()))} breeds")

# Create test split
num_test = int(0.1 * len(dataset_hierarchical))
num_train = len(dataset_hierarchical) - num_test
train_set, test_set = random_split(dataset_hierarchical, [num_train, num_test])

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# === Evaluation with hierarchical dataset ===
correct, total = 0, 0
all_labels, all_preds = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

accuracy = correct / total

# === Save evaluation summary ===
evaluation_summary = {
    "Test_Accuracy": accuracy, 
    "Test_Size": total,
    "Model_Best_Val_Acc": checkpoint['best_val_acc'],
    "Model_Best_Epoch": checkpoint['epoch'] + 1,
    "Num_Classes": num_classes,
    "Model_Architecture": "EfficientNet-B2 + Custom Classifier"
}

pd.DataFrame([evaluation_summary]).to_csv(REPORT_PATH, index=False)

print(f"✅ Evaluation complete. Test Accuracy: {accuracy:.2%}")
print(f"📊 Model's best validation accuracy during training: {checkpoint['best_val_acc']:.2%}")
print(f"📊 Report saved at: {REPORT_PATH}")

# === Confusion Matrix (Counts) ===
cm_counts = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
plt.figure(figsize=(12, 10))
sns.heatmap(cm_counts, annot=True, fmt="d", cmap="Blues",
            xticklabels=breeds,
            yticklabels=breeds)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Counts)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CM_COUNTS_PATH)
plt.close()
print(f"📉 Confusion matrix (counts) saved at: {CM_COUNTS_PATH}")

# === Confusion Matrix (Percent per class) ===
# Add small epsilon to avoid division by zero
row_sums = cm_counts.sum(axis=1, keepdims=True)
row_sums = np.where(row_sums == 0, 1, row_sums)  # Replace 0 with 1 to avoid division by zero
cm_percent = cm_counts.astype('float') / row_sums * 100

plt.figure(figsize=(12, 10))
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues",
            xticklabels=breeds,
            yticklabels=breeds)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Normalized %)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(CM_PERCENT_PATH)
plt.close()
print(f"📉 Confusion matrix (percent) saved at: {CM_PERCENT_PATH}")

# === Sample Predictions (10 random images, with top-3 classes) ===
random_indices = random.sample(range(len(test_set)), min(10, len(test_set)))

sample_data = []
for idx in random_indices:
    img, label = test_set[idx]
    
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    top3_idx = probs.argsort()[-3:][::-1]
    top3_labels = [breeds[i] for i in top3_idx]
    top3_probs = [round(probs[i], 4) for i in top3_idx]

    true_label = breeds[label]
    pred_label = top3_labels[0]
    confidence = top3_probs[0]

    # Get image path from the hierarchical dataset
    img_path = test_set.dataset.samples[test_set.indices[idx]][0]

    sample_data.append({
        "Image Path": img_path,
        "True Label": true_label,
        "Top-1 Prediction": pred_label,
        "Top-1 Confidence": confidence,
        "Top-3 Predictions": ", ".join(top3_labels),
        "Top-3 Confidences": ", ".join(str(p) for p in top3_probs)
    })

if sample_data:  # Only save if we have valid samples
    pd.DataFrame(sample_data).to_csv(PREDICTIONS_PATH, index=False)
    print(f"🔍 Sample predictions saved at: {PREDICTIONS_PATH}")
else:
    print("⚠️ No valid samples found for predictions")
