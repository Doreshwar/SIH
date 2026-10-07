import os
import pandas as pd

print("CATTLE BREED DATASET ANALYSIS RESULTS")
print("=" * 50)

# Load breed labels
df = pd.read_csv('../annotations/breed_labels.csv')

# Original dataset counts
data_dir = '../data/raw'
original_counts = {}
total_original = 0

print("\nORIGINAL DATASET:")
for _, row in df.iterrows():
    folder_name = row["folder_name"]
    breed_name = row["breed_name"]
    species = row["species"]
    folder_path = os.path.join(data_dir, species.upper(), folder_name)
    
    if os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        count = len(files)
        original_counts[breed_name] = count
        total_original += count
        print(f"  {breed_name}: {count} images")

# Cleaned dataset counts
clean_data_dir = '../data/cleaned'
cleaned_counts = {}
total_cleaned = 0

print(f"\nCLEANED DATASET:")
if os.path.exists(clean_data_dir):
    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(clean_data_dir, species.upper(), folder_name)
        
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
            count = len(files)
            cleaned_counts[breed_name] = count
            total_cleaned += count
            original = original_counts[breed_name]
            removed = original - count
            print(f"  {breed_name}: {original} -> {count} images (removed {removed})")

    # Calculate improvements
    min_original = min(original_counts.values())
    max_original = max(original_counts.values())
    imbalance_original = max_original / min_original
    
    min_cleaned = min(cleaned_counts.values())
    max_cleaned = max(cleaned_counts.values())
    imbalance_cleaned = max_cleaned / min_cleaned

    print(f"\nSUMMARY:")
    print(f"  Original total: {total_original} images")
    print(f"  Cleaned total: {total_cleaned} images")
    print(f"  Images removed: {total_original - total_cleaned}")
    print(f"  Original imbalance: {imbalance_original:.2f}x")
    print(f"  Cleaned imbalance: {imbalance_cleaned:.2f}x")
    print(f"  Improvement: {((imbalance_original - imbalance_cleaned) / imbalance_original * 100):.1f}%")

    print(f"\nEXPECTED RESULTS:")
    print(f"  Original accuracy: ~56%")
    print(f"  Expected new accuracy: 65-75%")
    print(f"  Improvement: 9-19 percentage points")

    print(f"\nNEXT STEP:")
    print(f"  Run: python train_cleaned_model.py")
else:
    print("Cleaned dataset not found!")

print("\n" + "=" * 50)
