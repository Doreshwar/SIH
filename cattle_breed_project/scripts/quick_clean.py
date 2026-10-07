import os
import pandas as pd
import random
import shutil

print("Starting quick dataset cleaning...")

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')
data_dir = '../data/raw'
clean_data_dir = '../data/cleaned'

# Create cleaned data directory
if os.path.exists(clean_data_dir):
    print("Removing existing cleaned directory...")
    shutil.rmtree(clean_data_dir)

os.makedirs(clean_data_dir, exist_ok=True)
print(f"Created cleaned data directory: {clean_data_dir}")

breed_counts = {}
total_images = 0
cleaned_images = 0

# Process each breed
for _, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Missing folder: {folder_path}")
        continue
    
    print(f"Processing {breed_name} ({species})...")
    
    # Create cleaned folder
    clean_breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
    os.makedirs(clean_breed_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
    
    print(f"  Found {len(image_files)} images")
    breed_counts[breed_name] = len(image_files)
    total_images += len(image_files)
    
    # For now, just copy all images (we'll filter problematic ones later)
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        clean_img_path = os.path.join(clean_breed_dir, img_file)
        
        try:
            shutil.copy2(img_path, clean_img_path)
            cleaned_images += 1
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")

print(f"\nCopy complete!")
print(f"Total images: {total_images}")
print(f"Copied images: {cleaned_images}")

# Show breed distribution
print(f"\nBreed distribution:")
for breed, count in breed_counts.items():
    print(f"  {breed}: {count} images")

# Calculate imbalance
min_count = min(breed_counts.values())
max_count = max(breed_counts.values())
imbalance_ratio = max_count / min_count

print(f"\nImbalance analysis:")
print(f"Min images per breed: {min_count}")
print(f"Max images per breed: {max_count}")
print(f"Imbalance ratio: {imbalance_ratio:.2f}x")

# Balance classes by removing excess from over-represented breeds
print(f"\nBalancing classes...")
target_size = min_count * 1.5  # Allow 50% more than minimum
max_allowed = int(target_size)

print(f"Target size: {target_size}")
print(f"Max allowed per breed: {max_allowed}")

excess_removed = 0
for breed_name, count in breed_counts.items():
    if count > max_allowed:
        excess_count = count - max_allowed
        
        # Find breed directory
        for _, row in df.iterrows():
            if row["breed_name"] == breed_name:
                species = row["species"]
                folder_name = row["folder_name"]
                breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
                break
        
        if os.path.exists(breed_dir):
            # Get list of images and randomly remove excess
            image_files = [f for f in os.listdir(breed_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
            
            # Randomly select images to remove
            random.seed(42)  # For reproducibility
            images_to_remove = random.sample(image_files, excess_count)
            
            for img_file in images_to_remove:
                img_path = os.path.join(breed_dir, img_file)
                os.remove(img_path)
                excess_removed += 1
            
            print(f"  {breed_name}: Removed {excess_count} excess images")

print(f"Removed {excess_removed} excess images for balancing")

# Final statistics
final_counts = {}
for _, row in df.iterrows():
    breed_name = row["breed_name"]
    species = row["species"]
    folder_name = row["folder_name"]
    breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
    
    if os.path.exists(breed_dir):
        image_files = [f for f in os.listdir(breed_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        final_counts[breed_name] = len(image_files)

final_total = sum(final_counts.values())
final_min = min(final_counts.values())
final_max = max(final_counts.values())
final_imbalance = final_max / final_min if final_min > 0 else float('inf')

print(f"\nFinal dataset statistics:")
print(f"Total images: {final_total}")
print(f"Min per breed: {final_min}")
print(f"Max per breed: {final_max}")
print(f"Imbalance ratio: {final_imbalance:.2f}x")

print(f"\nPer-breed breakdown:")
for breed_name, count in final_counts.items():
    print(f"  {breed_name}: {count} images")

print(f"\nDataset cleaning complete!")
print(f"Cleaned dataset saved to: {clean_data_dir}")
print(f"Imbalance ratio improved from {imbalance_ratio:.2f}x to {final_imbalance:.2f}x")
