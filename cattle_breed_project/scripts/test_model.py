#!/usr/bin/env python3
"""
Test script to verify the model architecture works correctly
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import os

# Test the model architecture
class ImprovedCattleClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(ImprovedCattleClassifier, self).__init__()
        
        # Use EfficientNet-B2 for better efficiency
        self.backbone = models.efficientnet_b2(pretrained=True)
        
        # Replace classifier with custom head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Custom classifier head - simpler for cleaner data
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def test_model():
    print("Testing model architecture...")
    
    # Test with dummy data
    batch_size = 4
    num_classes = 11  # From the error log
    
    model = ImprovedCattleClassifier(num_classes=num_classes)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Model test successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected output shape: ({batch_size}, {num_classes})")
            
            if output.shape == (batch_size, num_classes):
                print("✅ Output shape is correct!")
            else:
                print("❌ Output shape is incorrect!")
                
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    return True

def test_data_loading():
    print("\nTesting data loading...")
    
    # Load breed labels
    df = pd.read_csv('../annotations/breed_labels.csv')
    print(f"✅ Loaded {len(df)} breeds from CSV")
    
    # Check cleaned dataset
    clean_data_dir = '../data/cleaned'
    if os.path.exists(clean_data_dir):
        print(f"✅ Cleaned dataset exists")
        
        # Count images
        total_images = 0
        breed_counts = {}
        
        for _, row in df.iterrows():
            folder_name = row["folder_name"]
            breed_name = row["breed_name"]
            species = row["species"]
            folder_path = os.path.join(clean_data_dir, species.upper(), folder_name)
            
            if os.path.exists(folder_path):
                image_files = [f for f in os.listdir(folder_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
                breed_counts[breed_name] = len(image_files)
                total_images += len(image_files)
        
        print(f"✅ Found {total_images} images across {len(breed_counts)} breeds")
        
        # Show distribution
        print("\nBreed distribution:")
        for breed, count in sorted(breed_counts.items()):
            print(f"  {breed}: {count} images")
        
        if len(breed_counts) > 0:
            min_count = min(breed_counts.values())
            max_count = max(breed_counts.values())
            imbalance = max_count / min_count
            print(f"\nClass balance: {min_count}-{max_count} images (ratio: {imbalance:.2f}x)")
        
        return True
    else:
        print("❌ Cleaned dataset not found")
        return False

def main():
    print("=" * 50)
    print("CATTLE BREED MODEL TEST")
    print("=" * 50)
    
    # Test model architecture
    model_ok = test_model()
    
    # Test data loading
    data_ok = test_data_loading()
    
    print("\n" + "=" * 50)
    if model_ok and data_ok:
        print("✅ ALL TESTS PASSED - Ready for training!")
        print("You can now run: python train_cleaned_model.py")
    else:
        print("❌ SOME TESTS FAILED - Please fix issues before training")
    print("=" * 50)

if __name__ == "__main__":
    main()
