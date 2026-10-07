#!/usr/bin/env python3
"""
Training script for the raw (original) dataset.
This script uses the original unbalanced dataset for comparison with the cleaned version.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Suppress CUDA device prints
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

# ----------------------------
# Configurations
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "raw")  # Use raw dataset
CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 15
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ----------------------------
# Transforms
# ----------------------------
def get_transforms():
    # Training transforms with moderate augmentation for raw dataset
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform_train, transform_val

# ----------------------------
# Dataset Class
# ----------------------------
class CattleDataset(Dataset):
    def __init__(self, dataframe, transform=None, label2idx=None):
        self.data = dataframe
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image"]
        label_name = self.data.iloc[idx]["label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)

        label_idx = self.label2idx[label_name]
        return image, label_idx

# ----------------------------
# Load Data from Raw Dataset
# ----------------------------
def load_raw_data():
    """Load data from the original raw dataset directory"""
    df = pd.read_csv(CSV_FILE)
    image_paths, labels = [], []

    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(RAW_DATASET, species.upper(), folder_name)

        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder_path}")
            continue

        for img in os.listdir(folder_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder_path, img))
                labels.append(breed_name)

    data = pd.DataFrame({"image": image_paths, "label": labels})
    return data

# ----------------------------
# Model Architecture
# ----------------------------
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCattleClassifier, self).__init__()
        
        # Use EfficientNet-B2 for better efficiency
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head - simpler for cleaner data (BatchNorm removed)
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

# ----------------------------
# Training & Validation
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ----------------------------
# Main
# ----------------------------
def main():
    print("Starting training with RAW (original unbalanced) dataset...")
    
    # Check if raw dataset exists
    if not os.path.exists(RAW_DATASET):
        print(f"Raw dataset not found at {RAW_DATASET}")
        print("Please check the dataset path.")
        return
    
    data = load_raw_data()
    print(f"Found {len(data)} images across {data['label'].nunique()} breeds")
    
    # Label mapping
    breeds = sorted(data["label"].unique())
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    print("Label mapping:", label2idx)
    
    # Check class distribution
    breed_counts = data['label'].value_counts()
    print(f"\nClass distribution:")
    for breed, count in breed_counts.items():
        print(f"  {breed}: {count} images")
    
    min_count = breed_counts.min()
    max_count = breed_counts.max()
    imbalance_ratio = max_count / min_count
    print(f"\nClass balance: {min_count}-{max_count} images (ratio: {imbalance_ratio:.2f}x)")
    print(f"NOTE: This is the ORIGINAL unbalanced dataset!")
    
    # Split train/val with stratification
    train_df, val_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )
    
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    # Compute class weights for balanced training
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label']
    )
    class_weight_dict = {breed: weight for breed, weight in zip(breeds, class_weights)}
    class_weights_tensor = torch.tensor([class_weight_dict[breed] for breed in breeds], dtype=torch.float32).to(DEVICE)
    
    # Get transforms
    transform_train, transform_val = get_transforms()
    
    # Datasets & Loaders
    train_dataset = CattleDataset(train_df, transform=transform_train, label2idx=label2idx)
    val_dataset = CattleDataset(val_df, transform=transform_val, label2idx=label2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    
    # Model
    model = ImprovedCattleClassifier(num_classes=len(breeds), dropout_rate=0.3)
    model = model.to(DEVICE)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Tracking for plots
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'epoch': epoch,
                'label2idx': label2idx,
                'breeds': breeds
            }, "best_model_raw.pth")
            print("Saved best model!")
        else:
            patience_counter += 1
        
        # Live plot
        ax1.clear()
        ax1.plot(history["train_loss"], label="Train Loss", color='blue')
        ax1.plot(history["val_loss"], label="Val Loss", color='red')
        ax1.set_title("Loss over Epochs (Raw Dataset)")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.plot(history["train_acc"], label="Train Acc", color='blue')
        ax2.plot(history["val_acc"], label="Val Acc", color='red')
        ax2.set_title("Accuracy over Epochs (Raw Dataset)")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.pause(0.1)
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break
    
    plt.ioff()
    plt.show()
    print(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load("best_model_raw.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_val_loss, final_val_acc = validate_one_epoch(model, val_loader, criterion)
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    # Compare with original accuracy
    print(f"\nRAW DATASET RESULTS:")
    print(f"Dataset: Original unbalanced dataset")
    print(f"Class imbalance ratio: {imbalance_ratio:.2f}x")
    print(f"Final accuracy: {final_val_acc:.2%}")
    
    if final_val_acc > 0.70:
        print("Good performance achieved!")
    elif final_val_acc > 0.60:
        print("Moderate performance achieved!")
    else:
        print("Performance could be improved with dataset balancing.")
    
    print(f"\nCOMPARISON NOTE:")
    print(f"Compare this result with the cleaned dataset training to see the improvement!")
    print(f"Expected: Cleaned dataset should perform better due to balanced classes.")

if __name__ == "__main__":
    main()
