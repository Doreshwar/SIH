import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

# ----------------------------
# Configurations
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "raw")
CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0005
PATIENCE = 7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {DEVICE}")

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

        image = Image.open(img_path).convert("RGB")
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
# Training & Validation
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

    # Split train/val
    train_df, val_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )

    # Transforms (with stronger augmentation)
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_dataset = CattleDataset(train_df, transform=transform_train, label2idx=label2idx)
    val_dataset = CattleDataset(val_df, transform=transform_val, label2idx=label2idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Compute class weights
    label_counts = Counter(train_df["label"])
    weights = [1.0 / label_counts[label] for label in train_df["label"]]
    class_weights = torch.tensor([1.0 / label_counts[breed] for breed in breeds], dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model (ResNet50 with dropout)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, len(breeds))
    )
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3, verbose=True)

    # Tracking for plots
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Training loop with early stopping + fine-tuning
    best_val_acc = 0.0
    patience_counter = 0

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for epoch in range(NUM_EPOCHS):
        # Fine-tuning: unfreeze backbone after 5 epochs
        if epoch == 5:
            for param in model.parameters():
                param.requires_grad = True
            print("🔓 Unfroze entire ResNet50 for fine-tuning")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"📅 Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"   🏋️ Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"   🧪 Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

        # LR scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("💾 Saved best model!")
        else:
            patience_counter += 1

        # Live plot
        ax1.clear()
        ax1.plot(history["train_loss"], label="Train Loss")
        ax1.plot(history["val_loss"], label="Val Loss")
        ax1.set_title("Loss over Epochs")
        ax1.legend()

        ax2.clear()
        ax2.plot(history["train_acc"], label="Train Acc")
        ax2.plot(history["val_acc"], label="Val Acc")
        ax2.set_title("Accuracy over Epochs")
        ax2.legend()

        plt.pause(0.1)

        # Early stopping
        if patience_counter >= PATIENCE:
            print("⏹️ Early stopping triggered")
            break

    plt.ioff()
    plt.show()
    print(f"✅ Training complete. Best Val Acc: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()
