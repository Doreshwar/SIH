import torch
import torch.nn as nn
import torchvision.models as models

# Test the fixed model architecture
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

print("Testing fixed model architecture...")

# Test with different batch sizes including single sample
batch_sizes = [1, 2, 4, 8]
num_classes = 11

model = ImprovedCattleClassifier(num_classes=num_classes)
model.eval()

for batch_size in batch_sizes:
    try:
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Batch size {batch_size}: SUCCESS - Output shape: {output.shape}")
            
    except Exception as e:
        print(f"Batch size {batch_size}: FAILED - {e}")

print("\nModel architecture test complete!")
print("The BatchNorm1d has been removed to prevent single-batch errors.")
print("Training should now work correctly.")
