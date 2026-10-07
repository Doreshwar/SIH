import os
import pandas as pd
from collections import defaultdict

def analyze_dataset():
    """Analyze the dataset structure and identify potential issues"""
    
    # Read breed labels
    df = pd.read_csv('../annotations/breed_labels.csv')
    
    print("=== DATASET ANALYSIS ===")
    print(f"Total breeds: {len(df)}")
    print(f"Species distribution:")
    print(df['species'].value_counts())
    print(f"\nBreed distribution:")
    print(df['breed_name'].value_counts())
    
    # Count images per breed
    data_dir = '../data/raw'
    breed_counts = defaultdict(int)
    total_images = 0
    problematic_files = []
    
    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(data_dir, species.upper(), folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue
            
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]
        
        breed_counts[breed_name] = len(image_files)
        total_images += len(image_files)
        
        # Check for very small or very large files
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            try:
                file_size = os.path.getsize(img_path)
                if file_size < 1000:  # Less than 1KB
                    problematic_files.append((img_path, f"Too small: {file_size} bytes"))
                elif file_size > 50 * 1024 * 1024:  # More than 50MB
                    problematic_files.append((img_path, f"Too large: {file_size} bytes"))
            except Exception as e:
                problematic_files.append((img_path, f"Error: {str(e)}"))
    
    print(f"\n=== IMAGE COUNTS PER BREED ===")
    for breed, count in sorted(breed_counts.items()):
        print(f"{breed}: {count} images")
    
    print(f"\nTotal images: {total_images}")
    
    # Check for class imbalance
    min_count = min(breed_counts.values())
    max_count = max(breed_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n=== CLASS IMBALANCE ANALYSIS ===")
    print(f"Min images per breed: {min_count}")
    print(f"Max images per breed: {max_count}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print("⚠️ High class imbalance detected!")
    
    # Identify problematic files
    print(f"\n=== PROBLEMATIC FILES ===")
    if problematic_files:
        print(f"Found {len(problematic_files)} potentially problematic files:")
        for file_path, issue in problematic_files[:10]:  # Show first 10
            print(f"  {file_path}: {issue}")
        if len(problematic_files) > 10:
            print(f"  ... and {len(problematic_files) - 10} more")
    else:
        print("No obviously problematic files found.")
    
    return breed_counts, problematic_files

if __name__ == "__main__":
    analyze_dataset()
