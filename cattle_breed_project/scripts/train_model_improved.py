import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
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
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configurations
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "raw")
CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
BATCH_SIZE = 16  # Reduced for better gradient updates
NUM_EPOCHS = 50  # Increased epochs
LEARNING_RATE = 0.001  # Slightly higher initial LR
PATIENCE = 10  # Increased patience
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
LABEL_SMOOTHING = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# ----------------------------
# Advanced Augmentation
# ----------------------------
def get_transforms():
    # Training transforms with advanced augmentation
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # Reduced for cattle
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
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
# Mixup and CutMix
# ----------------------------
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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
# Load Data
# ----------------------------
def load_data():
    df = pd.read_csv(CSV_FILE)
    image_paths, labels = [], []

    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(RAW_DATASET, species.upper(), folder_name)

        if not os.path.exists(folder_path):
            continue

        for img in os.listdir(folder_path):
            if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(folder_path, img))
                labels.append(breed_name)

    data = pd.DataFrame({"image": image_paths, "label": labels})
    return data

# ----------------------------
# Advanced Model Architecture
# ----------------------------
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(ImprovedCattleClassifier, self).__init__()
        
        # Use EfficientNet-B3 for better performance
        self.backbone = models.efficientnet_b3(pretrained=True)
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ----------------------------
# Training & Validation
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, scaler, use_mixup=True, use_cutmix=True):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            # Apply mixup or cutmix
            if use_mixup and random.random() < 0.5:
                images, y_a, y_b, lam = mixup_data(images, labels, MIXUP_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            elif use_cutmix and random.random() < 0.5:
                images, y_a, y_b, lam = cutmix_data(images, labels, CUTMIX_ALPHA)
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
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
    data = load_data()
    print(f"✅ Found {len(data)} images across {data['label'].nunique()} breeds")
    
    # Label mapping
    breeds = sorted(data["label"].unique())
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    print("🔖 Label mapping:", label2idx)
    
    # Split train/val with stratification
    train_df, val_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )
    
    # Compute class weights for imbalanced data
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(train_df['label']), 
        y=train_df['label']
    )
    class_weight_dict = {breed: weight for breed, weight in zip(breeds, class_weights)}
    class_weights_tensor = torch.tensor([class_weight_dict[breed] for breed in breeds], dtype=torch.float32).to(DEVICE)
    
    # Create weighted sampler for training
    sample_weights = [class_weight_dict[label] for label in train_df['label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Get transforms
    transform_train, transform_val = get_transforms()
    
    # Datasets & Loaders
    train_dataset = CattleDataset(train_df, transform=transform_train, label2idx=label2idx)
    val_dataset = CattleDataset(val_df, transform=transform_val, label2idx=label2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)
    
    # Model
    model = ImprovedCattleClassifier(num_classes=len(breeds), dropout_rate=0.5)
    model = model.to(DEVICE)
    
    # Optimizer with different learning rates for backbone and classifier
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
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
    
    print("🚀 Starting training with advanced techniques...")
    
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
        
        print(f"📅 Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"   🏋️ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   🧪 Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
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
            }, "best_model_improved.pth")
            print("💾 Saved best model!")
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
            print("⏹️ Early stopping triggered")
            break
    
    plt.ioff()
    plt.show()
    print(f"✅ Training complete. Best Val Acc: {best_val_acc:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load("best_model_improved.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_val_loss, final_val_acc = validate_one_epoch(model, val_loader, criterion)
    print(f"🎯 Final Validation Accuracy: {final_val_acc:.4f}")

if __name__ == "__main__":
    main()
