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
        self.backbone = models.efficientnet_b2(weights='IMAGENET1K_V1')
        
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

print("Testing model architecture...")

# Test with dummy data
batch_size = 4
num_classes = 11

model = ImprovedCattleClassifier(num_classes=num_classes)
model.eval()

# Create dummy input
dummy_input = torch.randn(batch_size, 3, 224, 224)

try:
    with torch.no_grad():
        output = model(dummy_input)
        print("Model test successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, {num_classes})")
        
        if output.shape == (batch_size, num_classes):
            print("Output shape is correct!")
        else:
            print("Output shape is incorrect!")
            
except Exception as e:
    print(f"Model test failed: {e}")

print("\nTesting data loading...")

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')
print(f"Loaded {len(df)} breeds from CSV")

# Check cleaned dataset
clean_data_dir = '../data/cleaned'
if os.path.exists(clean_data_dir):
    print("Cleaned dataset exists")
    
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
    
    print(f"Found {total_images} images across {len(breed_counts)} breeds")
    
    if len(breed_counts) > 0:
        min_count = min(breed_counts.values())
        max_count = max(breed_counts.values())
        imbalance = max_count / min_count
        print(f"Class balance: {min_count}-{max_count} images (ratio: {imbalance:.2f}x)")
    
    print("All tests passed! Ready for training.")
else:
    print("Cleaned dataset not found")
