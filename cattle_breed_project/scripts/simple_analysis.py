import os
import pandas as pd

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')
print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nBreed distribution:")
print(df['breed_name'].value_counts())

# Count images per breed
data_dir = '../data/raw'
total_images = 0
breed_counts = {}

for _, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    if os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        breed_counts[breed_name] = len(image_files)
        total_images += len(image_files)
        print(f"{breed_name}: {len(image_files)} images")
    else:
        print(f"Missing folder: {folder_path}")

print(f"\nTotal images: {total_images}")

# Check for class imbalance
if breed_counts:
    min_count = min(breed_counts.values())
    max_count = max(breed_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\nClass imbalance analysis:")
    print(f"Min images per breed: {min_count}")
    print(f"Max images per breed: {max_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print("⚠️ High class imbalance detected!")
        print("This could be contributing to low accuracy.")
