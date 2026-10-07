#!/usr/bin/env python3
"""
Advanced Statistics Script for Cattle Breed Classification Model
Provides comprehensive evaluation metrics including precision, recall, F1 scores
and detailed per-class analysis.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === Custom Dataset Class for Hierarchical Structure ===
class HierarchicalCattleDataset(Dataset):
    def __init__(self, data_dir, transform=None, label2idx=None):
        self.data_dir = data_dir
        self.transform = transform
        self.label2idx = label2idx or {}
        self.samples = []
        
        # Load samples from hierarchical structure
        for species_dir in ['CATTLE', 'BUFFALO']:
            species_path = os.path.join(data_dir, species_dir)
            if not os.path.exists(species_path):
                continue
                
            for breed_folder in os.listdir(species_path):
                breed_path = os.path.join(species_path, breed_folder)
                if not os.path.isdir(breed_path):
                    continue
                    
                # Extract breed name from folder name
                breed_name = self._extract_breed_name(breed_folder)
                if breed_name not in self.label2idx:
                    print(f"⚠️ Breed '{breed_name}' not found in model classes, skipping")
                    continue
                    
                # Add all images from this breed folder
                for img_file in os.listdir(breed_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(breed_path, img_file)
                        self.samples.append((img_path, self.label2idx[breed_name]))
        
        print(f"Loaded {len(self.samples)} samples for advanced statistics")
        
    def _extract_breed_name(self, folder_name):
        """Extract breed name from folder name like INDIA_BUFFALO_0400_BANNI_01011"""
        parts = folder_name.split('_')
        if len(parts) >= 4:
            return parts[3].lower()
        return folder_name.lower()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

# === Model Architecture ===
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCattleClassifier, self).__init__()
        
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
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

def load_model_and_data():
    """Load the trained model and dataset"""
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data", "cleaned")
    MODEL_PATH = os.path.join(BASE_DIR, "Best_model_final.pth")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check model file
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print("Please train the model first using train_cleaned_model.py")
        return None, None, None, None, None
    
    # Load checkpoint
    print(f"Loading model from: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    
    # Extract model information
    num_classes = checkpoint['num_classes'] 
    label2idx = checkpoint['label2idx']
    breeds = checkpoint['breeds']
    
    print(f"Model info: {num_classes} classes - {breeds}")
    
    # Initialize and load model
    model = ImprovedCattleClassifier(num_classes=num_classes, dropout_rate=0.3)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = HierarchicalCattleDataset(DATA_DIR, transform=transform, label2idx=label2idx)
    
    if len(dataset) == 0:
        print("❌ No valid samples found in dataset")
        return None, None, None, None, None
    
    return model, dataset, breeds, label2idx, device

def evaluate_model(model, dataset, device, test_size=0.2):
    """Evaluate model and return predictions and true labels"""
    # Create test split
    num_test = int(test_size * len(dataset))
    num_train = len(dataset) - num_test
    _, test_set = random_split(dataset, [num_train, num_test])
    
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"Evaluating on {len(test_set)} test samples...")
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def calculate_comprehensive_metrics(y_true, y_pred, y_probs, breeds):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1 - Multiple averaging methods
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Classification report
    class_report = classification_report(
        y_true, y_pred, 
        target_names=breeds, 
        zero_division=0,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Class distribution in test set
    unique, counts = np.unique(y_true, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'classification_report': class_report,
        'confusion_matrix': cm,
        'class_distribution': class_distribution
    }

def save_advanced_statistics(metrics, breeds, base_dir, output_file="advanced_statistics.txt"):
    """Save comprehensive statistics to text file"""
    
    # Ensure output file is saved in the scripts directory
    output_path = os.path.join(base_dir, output_file)
    
    with open(output_path, 'w') as f:
        f.write("CATTLE BREED CLASSIFICATION - ADVANCED STATISTICS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Header information
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Architecture: EfficientNet-B2 + Custom Classifier\n")
        f.write(f"Number of Classes: {len(breeds)}\n")
        f.write(f"Classes: {', '.join(breeds)}\n\n")
        
        # Overall Performance Summary
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write(f"Accuracy:                    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Balanced Accuracy:           {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)\n\n")
        
        # Precision, Recall, F1 Score Summary
        f.write("PRECISION, RECALL, F1 SCORE SUMMARY\n")
        f.write("-" * 50 + "\n")
        f.write("Macro Average (unweighted across classes):\n")
        f.write(f"  Precision: {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)\n")
        f.write(f"  Recall:    {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)\n")
        f.write(f"  F1 Score:  {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)\n\n")
        
        f.write("Micro Average (global across all samples):\n")
        f.write(f"  Precision: {metrics['precision_micro']:.4f} ({metrics['precision_micro']*100:.2f}%)\n")
        f.write(f"  Recall:    {metrics['recall_micro']:.4f} ({metrics['recall_micro']*100:.2f}%)\n")
        f.write(f"  F1 Score:  {metrics['f1_micro']:.4f} ({metrics['f1_micro']*100:.2f}%)\n\n")
        
        f.write("Weighted Average (weighted by class frequency):\n")
        f.write(f"  Precision: {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)\n")
        f.write(f"  Recall:    {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)\n")
        f.write(f"  F1 Score:  {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)\n\n")
        
        # Per-Class Detailed Analysis
        f.write("PER-CLASS DETAILED ANALYSIS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 50 + "\n")
        
        for i, breed in enumerate(breeds):
            support = metrics['class_distribution'].get(i, 0)
            precision = metrics['precision_per_class'][i]
            recall = metrics['recall_per_class'][i]
            f1 = metrics['f1_per_class'][i]
            
            f.write(f"{breed:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}\n")
        
        f.write("\n")
        
        # Performance Analysis
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 50 + "\n")
        
        # Best performing classes
        best_f1_idx = np.argmax(metrics['f1_per_class'])
        worst_f1_idx = np.argmin(metrics['f1_per_class'])
        
        f.write(f"Best Performing Class:  {breeds[best_f1_idx]} (F1: {metrics['f1_per_class'][best_f1_idx]:.4f})\n")
        f.write(f"Worst Performing Class: {breeds[worst_f1_idx]} (F1: {metrics['f1_per_class'][worst_f1_idx]:.4f})\n\n")
        
        # Class imbalance analysis
        support_values = [metrics['class_distribution'].get(i, 0) for i in range(len(breeds))]
        max_support = max(support_values)
        min_support = min([s for s in support_values if s > 0])  # Exclude zero support
        imbalance_ratio = max_support / min_support if min_support > 0 else float('inf')
        
        f.write(f"Class Distribution Analysis:\n")
        f.write(f"  Most samples:  {max_support} samples\n")
        f.write(f"  Least samples: {min_support} samples\n")
        f.write(f"  Imbalance ratio: {imbalance_ratio:.2f}x\n\n")
        
        # Model recommendations
        f.write("MODEL PERFORMANCE ASSESSMENT\n")
        f.write("-" * 50 + "\n")
        
        if metrics['f1_macro'] >= 0.80:
            performance_level = "Excellent"
        elif metrics['f1_macro'] >= 0.70:
            performance_level = "Good"
        elif metrics['f1_macro'] >= 0.60:  
            performance_level = "Fair"
        else:
            performance_level = "Needs Improvement"
            
        f.write(f"Overall Performance Level: {performance_level}\n")
        f.write(f"Macro F1 Score: {metrics['f1_macro']:.4f}\n\n")
        
        if metrics['f1_macro'] < 0.70:
            f.write("Recommendations for Improvement:\n")
            f.write("• Consider collecting more data for underperforming classes\n")
            f.write("• Experiment with different data augmentation strategies\n")
            f.write("• Try advanced techniques like focal loss for class imbalance\n")
            f.write("• Consider ensemble methods or model fine-tuning\n\n")
        
        # Technical Details
        f.write("TECHNICAL EVALUATION DETAILS\n")
        f.write("-" * 50 + "\n")
        f.write("Evaluation Method: Stratified random test split (20% of dataset)\n")
        f.write("Metrics Computed: Sklearn precision_score, recall_score, f1_score\n")
        f.write("Zero Division Handling: Set to 0 for undefined metrics\n")
        f.write("Multi-class Strategy: One-vs-Rest for binary classification metrics\n\n")
        
        # Detailed Classification Report (sklearn format)
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 50 + "\n")
        
        # Convert classification report to readable format
        report = metrics['classification_report']
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
        f.write("-" * 52 + "\n")
        
        for breed in breeds:
            if breed in report:
                class_metrics = report[breed]
                f.write(f"{breed:<12} {class_metrics['precision']:<10.4f} "
                       f"{class_metrics['recall']:<10.4f} {class_metrics['f1-score']:<10.4f} "
                       f"{class_metrics['support']:<10.0f}\n")
        
        f.write("\n")
        f.write(f"{'accuracy':<12} {'':<10} {'':<10} {report['accuracy']:<10.4f} {report['macro avg']['support']:<10.0f}\n")
        f.write(f"{'macro avg':<12} {report['macro avg']['precision']:<10.4f} "
               f"{report['macro avg']['recall']:<10.4f} {report['macro avg']['f1-score']:<10.4f} "
               f"{report['macro avg']['support']:<10.0f}\n")
        f.write(f"{'weighted avg':<12} {report['weighted avg']['precision']:<10.4f} "
               f"{report['weighted avg']['recall']:<10.4f} {report['weighted avg']['f1-score']:<10.4f} "
               f"{report['weighted avg']['support']:<10.0f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Advanced Statistics Report\n")
    
    return output_path  # Return the full path where file was saved

def main():
    """Main function to run advanced statistics analysis"""
    print("🔬 CATTLE BREED CLASSIFICATION - ADVANCED STATISTICS")
    print("=" * 60)
    
    # Load model and data
    model, dataset, breeds, label2idx, device = load_model_and_data()
    
    if model is None:
        print("❌ Failed to load model or data. Exiting.")
        return
    
    # Get base directory for saving files
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    print(f"✅ Model and data loaded successfully")
    print(f"📊 Dataset contains {len(dataset)} samples across {len(breeds)} breeds")
    
    # Evaluate model
    y_true, y_pred, y_probs = evaluate_model(model, dataset, device)
    
    print(f"✅ Model evaluation completed")
    print(f"📊 Test set size: {len(y_true)} samples")
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(y_true, y_pred, y_probs, breeds)
    
    # Save results
    output_file = "advanced_statistics.txt"
    output_path = save_advanced_statistics(metrics, breeds, BASE_DIR, output_file)
    
    # Print summary to console
    print(f"\n📈 EVALUATION RESULTS SUMMARY:")
    print(f"   Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Balanced Acc:    {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
    print(f"   Precision (Macro): {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)")
    print(f"   Recall (Macro):    {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)")
    print(f"   F1 Score (Macro):  {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)")
    print(f"   F1 Score (Weighted): {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)")
    
    print(f"\n✅ Advanced statistics saved to: {output_path}")
    print(f"📄 Report contains comprehensive precision, recall, F1 scores and per-class analysis")
    
    # Also print the absolute path so user can easily find it
    print(f"📁 Full file path: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()