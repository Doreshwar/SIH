import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
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

# Import the improved model class
from train_model_improved import ImprovedCattleClassifier, get_transforms, mixup_data, cutmix_data, mixup_criterion

# ----------------------------
# Configurations
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "raw")
CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 0.001
PATIENCE = 8
WEIGHT_DECAY = 1e-4
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
LABEL_SMOOTHING = 0.1
N_FOLDS = 5

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
# Training & Validation Functions
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
# Cross-Validation Training
# ----------------------------
def train_fold(train_df, val_df, fold, breeds, label2idx):
    """Train a single fold"""
    print(f"\n🔄 Training Fold {fold + 1}/{N_FOLDS}")
    
    # Compute class weights
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
    
    # Optimizer
    backbone_params = list(model.backbone.parameters())
    classifier_params = list(model.classifier.parameters())
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=8, T_mult=2, eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    fold_history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)
        
        fold_history["train_loss"].append(train_loss)
        fold_history["train_acc"].append(train_acc)
        fold_history["val_loss"].append(val_loss)
        fold_history["val_acc"].append(val_acc)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"   📅 Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"      🏋️ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"      🧪 Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        # Save best model for this fold
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
                'breeds': breeds,
                'fold': fold
            }, f"best_model_fold_{fold}.pth")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"   ⏹️ Early stopping triggered for fold {fold + 1}")
            break
    
    print(f"   ✅ Fold {fold + 1} complete. Best Val Acc: {best_val_acc:.4f}")
    return best_val_acc, fold_history

# ----------------------------
# Main Cross-Validation
# ----------------------------
def main():
    data = load_data()
    print(f"✅ Found {len(data)} images across {data['label'].nunique()} breeds")
    
    # Label mapping
    breeds = sorted(data["label"].unique())
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    print("🔖 Label mapping:", label2idx)
    
    # Prepare data for cross-validation
    X = data['image'].values
    y = data['label'].values
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    fold_accuracies = []
    all_histories = []
    
    print(f"🚀 Starting {N_FOLDS}-Fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Create fold data
        train_df = data.iloc[train_idx].reset_index(drop=True)
        val_df = data.iloc[val_idx].reset_index(drop=True)
        
        print(f"\n📊 Fold {fold + 1} - Train: {len(train_df)}, Val: {len(val_df)}")
        
        # Train fold
        best_acc, fold_history = train_fold(train_df, val_df, fold, breeds, label2idx)
        fold_accuracies.append(best_acc)
        all_histories.append(fold_history)
    
    # Calculate and display results
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n🎯 Cross-Validation Results:")
    print(f"   Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"   Individual Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves for all folds
    ax1 = axes[0, 0]
    for i, history in enumerate(all_histories):
        ax1.plot(history['train_loss'], label=f'Fold {i+1} Train', alpha=0.7)
        ax1.plot(history['val_loss'], label=f'Fold {i+1} Val', alpha=0.7, linestyle='--')
    ax1.set_title('Loss Curves - All Folds')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Accuracy curves for all folds
    ax2 = axes[0, 1]
    for i, history in enumerate(all_histories):
        ax2.plot(history['train_acc'], label=f'Fold {i+1} Train', alpha=0.7)
        ax2.plot(history['val_acc'], label=f'Fold {i+1} Val', alpha=0.7, linestyle='--')
    ax2.set_title('Accuracy Curves - All Folds')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # Plot 3: Fold accuracies
    ax3 = axes[1, 0]
    bars = ax3.bar(range(1, N_FOLDS + 1), fold_accuracies)
    ax3.set_title('Fold Accuracies')
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Best Validation Accuracy')
    ax3.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, fold_accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Add mean line
    ax3.axhline(y=mean_acc, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_acc:.3f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of accuracies
    ax4 = axes[1, 1]
    ax4.boxplot(fold_accuracies, labels=['All Folds'])
    ax4.set_title('Accuracy Distribution')
    ax4.set_ylabel('Validation Accuracy')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    results = {
        'fold_accuracies': fold_accuracies,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'all_histories': all_histories
    }
    
    import pickle
    with open('cv_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\n💾 Results saved to cv_results.pkl")
    print(f"📊 Plots saved to cross_validation_results.png")
    
    return mean_acc, std_acc, fold_accuracies

if __name__ == "__main__":
    main()
