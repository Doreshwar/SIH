import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Test Time Augmentation
# ----------------------------
class TTADataset(Dataset):
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
        return image, label_idx, img_path

def get_tta_transforms():
    """Get multiple augmented versions for test-time augmentation"""
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Different augmentation strategies
    tta_transforms = [
        # Original
        base_transform,
        
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        # Rotation
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        # Color jitter
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        # Random crop
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    ]
    
    return tta_transforms

def predict_with_tta(model, dataloader, tta_transforms, device, num_augmentations=5):
    """Make predictions using test-time augmentation"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in dataloader:
            batch_predictions = []
            
            # Get predictions for each TTA transform
            for transform in tta_transforms[:num_augmentations]:
                # Apply transform to batch
                transformed_images = []
                for img in images:
                    # Convert tensor back to PIL for transform
                    if isinstance(img, torch.Tensor):
                        # Denormalize
                        img_denorm = img.clone()
                        img_denorm[0] = img_denorm[0] * 0.229 + 0.485
                        img_denorm[1] = img_denorm[1] * 0.224 + 0.456
                        img_denorm[2] = img_denorm[2] * 0.225 + 0.406
                        img_denorm = torch.clamp(img_denorm, 0, 1)
                        
                        # Convert to PIL
                        img_pil = transforms.ToPILImage()(img_denorm)
                    else:
                        img_pil = img
                    
                    transformed_img = transform(img_pil)
                    transformed_images.append(transformed_img)
                
                transformed_images = torch.stack(transformed_images).to(device)
                
                # Get predictions
                outputs = model(transformed_images)
                probabilities = torch.softmax(outputs, dim=1)
                batch_predictions.append(probabilities.cpu().numpy())
            
            # Average predictions across all augmentations
            avg_predictions = np.mean(batch_predictions, axis=0)
            predictions = np.argmax(avg_predictions, axis=1)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
    
    return np.array(all_predictions), np.array(all_labels), all_paths

def evaluate_model_with_tta(model_path, test_data, label2idx, breeds, device):
    """Evaluate model with test-time augmentation"""
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ImprovedCattleClassifier(num_classes=len(breeds), dropout_rate=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get TTA transforms
    tta_transforms = get_tta_transforms()
    
    # Create dataset and dataloader
    test_dataset = TTADataset(test_data, transform=None, label2idx=label2idx)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Make predictions with TTA
    predictions, true_labels, paths = predict_with_tta(
        model, test_loader, tta_transforms, device, num_augmentations=5
    )
    
    # Calculate accuracy
    accuracy = np.mean(predictions == true_labels)
    
    # Generate detailed report
    print(f"🎯 Test-Time Augmentation Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Total samples: {len(predictions)}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=breeds))
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=breeds, yticklabels=breeds)
    plt.title('Confusion Matrix - Test Time Augmentation')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_tta.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Per-class accuracy
    class_accuracies = []
    for i, breed in enumerate(breeds):
        mask = true_labels == i
        if np.sum(mask) > 0:
            class_acc = np.mean(predictions[mask] == true_labels[mask])
            class_accuracies.append(class_acc)
            print(f"   {breed}: {class_acc:.4f} ({np.sum(mask)} samples)")
        else:
            class_accuracies.append(0.0)
    
    # Plot per-class accuracy
    plt.figure(figsize=(15, 6))
    bars = plt.bar(breeds, class_accuracies)
    plt.title('Per-Class Accuracy with Test-Time Augmentation')
    plt.xlabel('Breed')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy_tta.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, predictions, true_labels

# ----------------------------
# Model Ensemble
# ----------------------------
class ModelEnsemble:
    def __init__(self, model_paths, breeds, device):
        self.models = []
        self.breeds = breeds
        self.device = device
        
        for model_path in model_paths:
            checkpoint = torch.load(model_path, map_location=device)
            model = ImprovedCattleClassifier(num_classes=len(breeds), dropout_rate=0.5)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            self.models.append(model)
    
    def predict(self, dataloader, tta_transforms=None, num_augmentations=3):
        """Make ensemble predictions"""
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, paths in dataloader:
                batch_predictions = []
                
                for model in self.models:
                    model_predictions = []
                    
                    if tta_transforms:
                        # Use TTA for each model
                        for transform in tta_transforms[:num_augmentations]:
                            transformed_images = []
                            for img in images:
                                if isinstance(img, torch.Tensor):
                                    # Denormalize and convert to PIL
                                    img_denorm = img.clone()
                                    img_denorm[0] = img_denorm[0] * 0.229 + 0.485
                                    img_denorm[1] = img_denorm[1] * 0.224 + 0.456
                                    img_denorm[2] = img_denorm[2] * 0.225 + 0.406
                                    img_denorm = torch.clamp(img_denorm, 0, 1)
                                    img_pil = transforms.ToPILImage()(img_denorm)
                                else:
                                    img_pil = img
                                
                                transformed_img = transform(img_pil)
                                transformed_images.append(transformed_img)
                            
                            transformed_images = torch.stack(transformed_images).to(self.device)
                            outputs = model(transformed_images)
                            probabilities = torch.softmax(outputs, dim=1)
                            model_predictions.append(probabilities.cpu().numpy())
                        
                        # Average predictions across augmentations for this model
                        avg_predictions = np.mean(model_predictions, axis=0)
                    else:
                        # No TTA, direct prediction
                        images_tensor = images.to(self.device)
                        outputs = model(images_tensor)
                        avg_predictions = torch.softmax(outputs, dim=1).cpu().numpy()
                    
                    batch_predictions.append(avg_predictions)
                
                # Average predictions across all models
                ensemble_predictions = np.mean(batch_predictions, axis=0)
                predictions = np.argmax(ensemble_predictions, axis=1)
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)

def main():
    # Load test data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILE = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")
    RAW_DATASET = os.path.join(BASE_DIR, "..", "data", "raw")
    
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
    
    # Split data (use same split as training)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )
    
    # Label mapping
    breeds = sorted(data["label"].unique())
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Check if improved model exists
    model_path = "best_model_improved.pth"
    if os.path.exists(model_path):
        print("🔍 Evaluating with Test-Time Augmentation...")
        accuracy, predictions, true_labels = evaluate_model_with_tta(
            model_path, test_df, label2idx, breeds, device
        )
    else:
        print("❌ Improved model not found. Please run train_model_improved.py first.")
        return
    
    print(f"\n✅ Evaluation complete! Final accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
