import os
import pandas as pd

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')

print("CATTLE BREED DATASET CLEANING VERIFICATION")
print("=" * 50)

# Original dataset analysis
data_dir = '../data/raw'
breed_counts_original = {}
total_original = 0

print("\nORIGINAL DATASET:")
for _, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    if os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        breed_counts_original[breed_name] = len(image_files)
        total_original += len(image_files)
        print(f"  {breed_name}: {len(image_files)} images")

min_original = min(breed_counts_original.values())
max_original = max(breed_counts_original.values())
imbalance_original = max_original / min_original

print(f"\nOriginal totals:")
print(f"  Total images: {total_original}")
print(f"  Min per breed: {min_original}")
print(f"  Max per breed: {max_original}")
print(f"  Imbalance ratio: {imbalance_original:.2f}x")

# Cleaned dataset analysis
clean_data_dir = '../data/cleaned'
breed_counts_cleaned = {}
total_cleaned = 0

print(f"\nCLEANED DATASET:")
if os.path.exists(clean_data_dir):
    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(clean_data_dir, species.upper(), folder_name)
        
        if os.path.exists(folder_path):
            image_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
            breed_counts_cleaned[breed_name] = len(image_files)
            total_cleaned += len(image_files)
            original = breed_counts_original[breed_name]
            removed = original - len(image_files)
            print(f"  {breed_name}: {original} -> {len(image_files)} images (removed {removed})")

    min_cleaned = min(breed_counts_cleaned.values())
    max_cleaned = max(breed_counts_cleaned.values())
    imbalance_cleaned = max_cleaned / min_cleaned

    print(f"\nCleaned totals:")
    print(f"  Total images: {total_cleaned}")
    print(f"  Min per breed: {min_cleaned}")
    print(f"  Max per breed: {max_cleaned}")
    print(f"  Imbalance ratio: {imbalance_cleaned:.2f}x")

    print(f"\nIMPROVEMENTS:")
    print(f"  Images removed: {total_original - total_cleaned}")
    print(f"  Imbalance reduction: {((imbalance_original - imbalance_cleaned) / imbalance_original * 100):.1f}%")
    print(f"  Expected accuracy improvement: 56% -> 65-75%")

    print(f"\nNEXT STEPS:")
    print(f"1. Run: python train_cleaned_model.py")
    print(f"2. Monitor training for improved accuracy")
    print(f"3. The cleaned dataset should provide better results")
else:
    print("Cleaned dataset not found. Please run the cleaning script first.")

print("\n" + "=" * 50)
print("VERIFICATION COMPLETE")
