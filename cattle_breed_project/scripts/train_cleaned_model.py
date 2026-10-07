#!/usr/bin/env python3
"""
Training script for the cleaned and balanced dataset.
This script addresses the class imbalance issue that was causing low accuracy.
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Suppress CUDA device prints
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
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
RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "cleaned")  # Use cleaned dataset
CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 15  # Increased patience for cleaner dataset
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
    print("Device name:", torch.cuda.get_device_name(0))

# ----------------------------
# Transforms
# ----------------------------
def get_transforms():
    # Training transforms with moderate augmentation for cleaner dataset
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),  # Reduced rotation for cleaner data
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
# Load Data from Cleaned Dataset
# ----------------------------
def load_cleaned_data():
    """Load data from the cleaned dataset directory"""
    df = pd.read_csv(CSV_FILE)
    image_paths, labels = [], []

    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(RAW_DATASET, species.upper(), folder_name)

        if not os.path.exists(folder_path):
            print(f"⚠️ Missing cleaned folder: {folder_path}")
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
    def __init__(self, num_classes, dropout_rate=0.3):  # Reduced dropout for cleaner data
        super(ImprovedCattleClassifier, self).__init__()
        
        # Use EfficientNet-B2 for better efficiency
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head - simpler for cleaner data
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

def validate_one_epoch(model, loader, criterion, return_predictions=False):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds = []
    all_labels = []
    
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
            
            if return_predictions:
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    if return_predictions:
        return epoch_loss, epoch_acc, all_preds, all_labels
    return epoch_loss, epoch_acc

def calculate_metrics(y_true, y_pred, breeds, label2idx):
    """Calculate precision, recall, F1 score and other metrics"""
    
    # Calculate overall metrics
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Create reverse mapping for breed names
    idx2breed = {v: k for k, v in label2idx.items()}
    
    # Get detailed classification report
    target_names = [idx2breed[i] for i in range(len(breeds))]
    class_report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
    
    return {
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'classification_report': class_report
    }

def print_metrics(metrics, epoch_num=None):
    """Print formatted metrics"""
    if epoch_num:
        print(f"\n--- Epoch {epoch_num} Metrics ---")
    else:
        print(f"\n--- Final Model Metrics ---")
    
    print(f"Macro Average:")
    print(f"  Precision: {metrics['precision_macro']:.4f}")
    print(f"  Recall:    {metrics['recall_macro']:.4f}")
    print(f"  F1 Score:  {metrics['f1_macro']:.4f}")
    
    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1 Score:  {metrics['f1_weighted']:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(metrics['classification_report'])

# ----------------------------
# Main
# ----------------------------
def main():
    print("Starting training with cleaned and balanced dataset...")
    
    # Check if cleaned dataset exists
    if not os.path.exists(RAW_DATASET):
        print(f"Cleaned dataset not found at {RAW_DATASET}")
        print("Please run the dataset cleaning script first.")
        return
    
    data = load_cleaned_data()
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
    
    # Tracking for plots and metrics
    history = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
        "precision_macro": [], "recall_macro": [], "f1_macro": [],
        "precision_weighted": [], "recall_weighted": [], "f1_weighted": []
    }
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    print("Starting training...")
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        
        # Calculate validation metrics (including predictions for F1, precision, recall)
        if (epoch + 1) % 5 == 0 or epoch == 0:  # Calculate detailed metrics every 5 epochs
            val_loss, val_acc, val_preds, val_labels = validate_one_epoch(model, val_loader, criterion, return_predictions=True)
            metrics = calculate_metrics(val_labels, val_preds, breeds, label2idx)
            
            # Store metrics in history
            history["precision_macro"].append(metrics['precision_macro'])
            history["recall_macro"].append(metrics['recall_macro'])
            history["f1_macro"].append(metrics['f1_macro'])
            history["precision_weighted"].append(metrics['precision_weighted'])
            history["recall_weighted"].append(metrics['recall_weighted'])
            history["f1_weighted"].append(metrics['f1_weighted'])
        else:
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
        
        # Print detailed metrics every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Val F1 (Macro): {metrics['f1_macro']:.4f}, F1 (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"   Val Precision: {metrics['precision_weighted']:.4f}, Recall: {metrics['recall_weighted']:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model with comprehensive information
            best_model_data = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'best_val_loss': val_loss,
                'epoch': epoch,
                'label2idx': label2idx,
                'breeds': breeds,
                'num_classes': len(breeds),
                'train_acc': train_acc,
                'train_loss': train_loss,
                'current_lr': current_lr,
                'metrics': metrics if (epoch + 1) % 5 == 0 or epoch == 0 else None,
                'model_architecture': 'EfficientNet-B2',
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'label_smoothing': LABEL_SMOOTHING,
                'patience': PATIENCE,
                'total_epochs': NUM_EPOCHS,
                'dataset_path': RAW_DATASET,
                'training_samples': len(train_df),
                'validation_samples': len(val_df)
            }
            torch.save(best_model_data, "Best_model_final.pth")
            print("Saved best model!")
        else:
            patience_counter += 1
        
        # Live plot
        ax1.clear()
        ax1.plot(history["train_loss"], label="Train Loss", color='blue')
        ax1.plot(history["val_loss"], label="Val Loss", color='red')
        ax1.set_title("Loss over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)
        
        ax2.clear()
        ax2.plot(history["train_acc"], label="Train Acc", color='blue')
        ax2.plot(history["val_acc"], label="Val Acc", color='red')
        ax2.set_title("Accuracy over Epochs")
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
    checkpoint = torch.load("Best_model_final.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint['epoch']
    best_train_acc = checkpoint.get('train_acc', 0)
    best_train_loss = checkpoint.get('train_loss', 0)
    best_val_loss = checkpoint.get('best_val_loss', 0)
    
    # Final validation with comprehensive metrics
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    
    final_val_loss, final_val_acc, final_preds, final_labels = validate_one_epoch(model, val_loader, criterion, return_predictions=True)
    final_metrics = calculate_metrics(final_labels, final_preds, breeds, label2idx)
    
    # Print comprehensive metrics
    print_metrics(final_metrics)
    
    # Compare with original accuracy
    print(f"\nIMPROVEMENT SUMMARY:")
    print(f"Original accuracy: ~56%")
    print(f"New accuracy: {final_val_acc:.2%}")
    print(f"Improvement: {final_val_acc - 0.56:.2%}")
    
    print(f"\nKEY METRICS SUMMARY:")
    print(f"F1 Score (Macro): {final_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {final_metrics['f1_weighted']:.4f}")
    print(f"Precision (Weighted): {final_metrics['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {final_metrics['recall_weighted']:.4f}")
    
    if final_val_acc > 0.70:
        print("\nSignificant improvement achieved!")
    elif final_val_acc > 0.60:
        print("\nModerate improvement achieved!")
    else:
        print("\nFurther optimization may be needed.")
    
    # Create comprehensive stats file for the best model
    with open("Best_model_final_stats.txt", "w") as f:
        f.write("CATTLE BREED CLASSIFICATION PROJECT - BEST MODEL STATISTICS\n")
        f.write("="*80 + "\n\n")
        
        # Project Overview
        f.write("PROJECT OVERVIEW:\n")
        f.write("-" * 40 + "\n")
        f.write("Project: Cattle Breed Classification using Deep Learning\n")
        f.write("Dataset: Cleaned and Balanced Cattle Breed Images\n")
        f.write(f"Total Classes: {len(breeds)}\n")
        f.write(f"Classes: {', '.join(breeds)}\n")
        f.write(f"Dataset Path: {RAW_DATASET}\n")
        f.write(f"Training Samples: {len(train_df)}\n")
        f.write(f"Validation Samples: {len(val_df)}\n\n")
        
        # Class Distribution
        f.write("CLASS DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")
        train_class_counts = train_df['label'].value_counts().sort_index()
        val_class_counts = val_df['label'].value_counts().sort_index()
        for breed in breeds:
            train_count = train_class_counts.get(breed, 0)
            val_count = val_class_counts.get(breed, 0)
            f.write(f"{breed}: {train_count} train, {val_count} val images\n")
        f.write(f"\nClass Balance Ratio: {max_count/min_count:.2f}x (Range: {min_count}-{max_count} images)\n\n")
        
        # Model Architecture & Configuration
        f.write("MODEL ARCHITECTURE & CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Base Architecture: EfficientNet-B2 (ImageNet pretrained)\n")
        f.write(f"Custom Classifier: 2-layer MLP with dropout\n")
        f.write(f"Input Size: 224x224 RGB images\n")
        f.write(f"Dropout Rate: 0.3\n")
        f.write(f"Number of Classes: {len(breeds)}\n")
        f.write(f"Total Parameters: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")
        f.write(f"Trainable Parameters: ~{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M\n\n")
        
        # Training Configuration
        f.write("TRAINING CONFIGURATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Optimizer: AdamW\n")
        f.write(f"Initial Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Total Epochs: {NUM_EPOCHS}\n")
        f.write(f"Early Stopping Patience: {PATIENCE}\n")
        f.write(f"Label Smoothing: {LABEL_SMOOTHING}\n")
        f.write(f"Loss Function: CrossEntropyLoss with class weights\n")
        f.write(f"Scheduler: CosineAnnealingWarmRestarts\n")
        f.write(f"Mixed Precision: Enabled (AMP)\n")
        f.write(f"Device: {DEVICE}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write("\n")
        
        # Data Augmentation
        f.write("DATA AUGMENTATION:\n")
        f.write("-" * 40 + "\n")
        f.write("Training Augmentations:\n")
        f.write("  - Random Resized Crop (224x224, scale 0.8-1.0)\n")
        f.write("  - Random Horizontal Flip (50% probability)\n")
        f.write("  - Random Rotation (±10 degrees)\n")
        f.write("  - Color Jitter (brightness, contrast, saturation, hue)\n")
        f.write("  - ImageNet Normalization\n")
        f.write("Validation: Resize to 224x224 + ImageNet Normalization\n\n")
        
        # Best Model Performance
        f.write("BEST MODEL PERFORMANCE:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Epoch: {best_epoch + 1}\n")
        f.write(f"Training Accuracy: {best_train_acc:.4f} ({best_train_acc*100:.2f}%)\n")
        f.write(f"Training Loss: {best_train_loss:.4f}\n")
        f.write(f"Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)\n")
        f.write(f"Validation Loss: {best_val_loss:.4f}\n\n")
        
        # Detailed Metrics
        f.write("DETAILED PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write("Macro Average (unweighted):\n")
        f.write(f"  Precision: {final_metrics['precision_macro']:.4f} ({final_metrics['precision_macro']*100:.2f}%)\n")
        f.write(f"  Recall:    {final_metrics['recall_macro']:.4f} ({final_metrics['recall_macro']*100:.2f}%)\n")
        f.write(f"  F1 Score:  {final_metrics['f1_macro']:.4f} ({final_metrics['f1_macro']*100:.2f}%)\n")
        f.write("\nWeighted Average (class-balanced):\n")
        f.write(f"  Precision: {final_metrics['precision_weighted']:.4f} ({final_metrics['precision_weighted']*100:.2f}%)\n")
        f.write(f"  Recall:    {final_metrics['recall_weighted']:.4f} ({final_metrics['recall_weighted']*100:.2f}%)\n")
        f.write(f"  F1 Score:  {final_metrics['f1_weighted']:.4f} ({final_metrics['f1_weighted']*100:.2f}%)\n\n")
        
        # Performance Analysis
        f.write("PERFORMANCE ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Improvement over baseline (~56%): {(final_val_acc - 0.56)*100:.2f} percentage points\n")
        if final_val_acc > 0.70:
            f.write("Status: Significant improvement achieved!\n")
        elif final_val_acc > 0.60:
            f.write("Status: Moderate improvement achieved!\n")
        else:
            f.write("Status: Further optimization recommended.\n")
        f.write("\n")
        
        # Technical Implementation Details
        f.write("TECHNICAL IMPLEMENTATION DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write("Key Technical Features:\n")
        f.write("• Class-weighted loss function to handle imbalanced data\n")
        f.write("• Label smoothing for better generalization\n")
        f.write("• Mixed precision training for memory efficiency\n")
        f.write("• Cosine annealing with warm restarts for learning rate\n")
        f.write("• Early stopping to prevent overfitting\n")
        f.write("• Stratified train/validation split\n")
        f.write("• Data augmentation optimized for cleaned dataset\n")
        f.write("• Reproducible training with fixed random seeds\n\n")
        
        # File Information
        f.write("MODEL FILES:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Model Weights: Best_model_final.pth\n")
        f.write(f"Statistics File: Best_model_final_stats.txt\n")
        f.write(f"Training Script: train_cleaned_model.py\n\n")
        
        # Detailed Classification Report
        f.write("DETAILED CLASSIFICATION REPORT:\n")
        f.write("-" * 40 + "\n")
        f.write(final_metrics['classification_report'])
        f.write("\n")
        
        # Usage Instructions
        f.write("MODEL USAGE INSTRUCTIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("To load and use this model:\n")
        f.write("1. Load the model checkpoint: torch.load('Best_model_final.pth')\n")
        f.write("2. Initialize the ImprovedCattleClassifier with the correct num_classes\n")
        f.write("3. Load the state_dict into the model\n")
        f.write("4. Use the label2idx mapping for class predictions\n")
        f.write("5. Apply the same preprocessing transforms as validation\n\n")
        
        # Footer
        f.write("="*80 + "\n")
        f.write("Generated by: Cattle Breed Classification Training Script\n")
        f.write(f"Date: {epoch+1} epochs completed\n")
        f.write("This report contains all technical details needed for model deployment and analysis.\n")
    
    print(f"\nComprehensive model statistics saved to: Best_model_final_stats.txt")
    print(f"Best model saved as: Best_model_final.pth")

if __name__ == "__main__":
    main()
