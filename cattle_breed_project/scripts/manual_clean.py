import os
import pandas as pd
import random
import shutil

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')
data_dir = '../data/raw'
clean_data_dir = '../data/cleaned'

# Remove existing cleaned directory
if os.path.exists(clean_data_dir):
    shutil.rmtree(clean_data_dir)

os.makedirs(clean_data_dir, exist_ok=True)

breed_counts = {}
total_images = 0

# Process each breed
for _, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    if not os.path.exists(folder_path):
        continue
    
    # Create cleaned folder
    clean_breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
    os.makedirs(clean_breed_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
    
    breed_counts[breed_name] = len(image_files)
    total_images += len(image_files)
    
    # Copy all images
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        clean_img_path = os.path.join(clean_breed_dir, img_file)
        shutil.copy2(img_path, clean_img_path)

print(f"Copied {total_images} images")

# Balance classes
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
final_imbalance = final_max / final_min

print(f"Final dataset: {final_total} images")
print(f"Min per breed: {final_min}")
print(f"Max per breed: {final_max}")
print(f"Imbalance ratio: {final_imbalance:.2f}x")
print("Cleaning complete!")
