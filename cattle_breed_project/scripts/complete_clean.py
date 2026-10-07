import os
import pandas as pd
import random
import shutil

print("Starting complete dataset cleaning...")

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')
data_dir = '../data/raw'
clean_data_dir = '../data/cleaned'

# Remove existing cleaned directory and recreate
if os.path.exists(clean_data_dir):
    shutil.rmtree(clean_data_dir)
os.makedirs(clean_data_dir, exist_ok=True)

breed_counts = {}
total_images = 0

print("Processing all breeds...")

# Process each breed
for idx, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    print(f"Processing {breed_name} ({species})...")
    
    if not os.path.exists(folder_path):
        print(f"  Missing folder: {folder_path}")
        continue
    
    # Create cleaned folder
    clean_breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
    os.makedirs(clean_breed_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
    
    breed_counts[breed_name] = len(image_files)
    total_images += len(image_files)
    
    print(f"  Found {len(image_files)} images")
    
    # Copy all images
    copied = 0
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        clean_img_path = os.path.join(clean_breed_dir, img_file)
        try:
            shutil.copy2(img_path, clean_img_path)
            copied += 1
        except Exception as e:
            print(f"  Error copying {img_file}: {e}")
    
    print(f"  Copied {copied} images")

print(f"\nCopy complete! Total images: {total_images}")

# Balance classes
print("\nBalancing classes...")
min_count = min(breed_counts.values())
target_size = int(min_count * 1.3)  # Allow 30% more than minimum
print(f"Target size: {target_size}")

excess_removed = 0
for breed_name, count in breed_counts.items():
    if count > target_size:
        excess_count = count - target_size
        
        # Find breed directory
        for _, row in df.iterrows():
            if row["breed_name"] == breed_name:
                species = row["species"]
                folder_name = row["folder_name"]
                breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
                break
        
        # Remove excess images
        image_files = [f for f in os.listdir(breed_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        
        random.seed(42)
        images_to_remove = random.sample(image_files, excess_count)
        
        for img_file in images_to_remove:
            img_path = os.path.join(breed_dir, img_file)
            os.remove(img_path)
            excess_removed += 1
        
        print(f"  {breed_name}: Removed {excess_count} excess images")

print(f"Removed {excess_removed} excess images for balancing")

# Final statistics
print("\nFinal statistics:")
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
final_imbalance = final_max / final_min

print(f"Final dataset: {final_total} images")
print(f"Min per breed: {final_min}")
print(f"Max per breed: {final_max}")
print(f"Imbalance ratio: {final_imbalance:.2f}x")

print("\nPer-breed breakdown:")
for breed_name, count in final_counts.items():
    print(f"  {breed_name}: {count} images")

print("\nCleaning complete!")
print("Ready for training with improved dataset!")
