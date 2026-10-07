#!/usr/bin/env python3
"""
Summary script showing the dataset cleaning results and recommendations
for improving training accuracy from 56% to a higher rate.
"""

import os
import pandas as pd

def main():
    print("=" * 60)
    print("CATTLE BREED CLASSIFICATION - DATASET ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Load breed labels
    df = pd.read_csv('../annotations/breed_labels.csv')
    
    print(f"\n📊 ORIGINAL DATASET ANALYSIS:")
    print(f"Total breeds: {len(df)}")
    print(f"Species: {df['species'].unique()}")
    
    # Count original images
    data_dir = '../data/raw'
    breed_counts_original = {}
    total_original = 0
    
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
    
    print(f"\n📈 ORIGINAL BREED DISTRIBUTION:")
    for breed, count in sorted(breed_counts_original.items()):
        print(f"  {breed}: {count} images")
    
    min_original = min(breed_counts_original.values())
    max_original = max(breed_counts_original.values())
    imbalance_original = max_original / min_original
    
    print(f"\n⚠️ ORIGINAL ISSUES IDENTIFIED:")
    print(f"  Total images: {total_original}")
    print(f"  Min per breed: {min_original}")
    print(f"  Max per breed: {max_original}")
    print(f"  Class imbalance ratio: {imbalance_original:.2f}x")
    print(f"  Training accuracy: ~56% (low due to imbalance)")
    
    # Count cleaned images
    clean_data_dir = '../data/cleaned'
    breed_counts_cleaned = {}
    total_cleaned = 0
    
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
        
        print(f"\n✅ CLEANED DATASET RESULTS:")
        print(f"📈 CLEANED BREED DISTRIBUTION:")
        for breed, count in sorted(breed_counts_cleaned.items()):
            original = breed_counts_original[breed]
            removed = original - count
            print(f"  {breed}: {original} -> {count} images (removed {removed})")
        
        min_cleaned = min(breed_counts_cleaned.values())
        max_cleaned = max(breed_counts_cleaned.values())
        imbalance_cleaned = max_cleaned / min_cleaned
        
        print(f"\n🎯 IMPROVEMENTS ACHIEVED:")
        print(f"  Total images: {total_original} -> {total_cleaned}")
        print(f"  Images removed: {total_original - total_cleaned}")
        print(f"  Min per breed: {min_original} -> {min_cleaned}")
        print(f"  Max per breed: {max_original} -> {max_cleaned}")
        print(f"  Class imbalance: {imbalance_original:.2f}x -> {imbalance_cleaned:.2f}x")
        print(f"  Imbalance reduction: {((imbalance_original - imbalance_cleaned) / imbalance_original * 100):.1f}%")
    
    print(f"\n🚀 RECOMMENDATIONS FOR IMPROVED TRAINING:")
    print(f"1. ✅ Dataset cleaned and balanced (completed)")
    print(f"2. ✅ Removed problematic images (completed)")
    print(f"3. ✅ Reduced class imbalance from {imbalance_original:.1f}x to {imbalance_cleaned:.1f}x (completed)")
    print(f"4. 🔄 Retrain model with cleaned dataset")
    print(f"5. 🔄 Use the train_cleaned_model.py script")
    
    print(f"\n📋 EXPECTED IMPROVEMENTS:")
    print(f"  • Better class balance should improve accuracy from 56% to 65-75%")
    print(f"  • Reduced overfitting due to balanced training")
    print(f"  • More stable training with consistent class representation")
    print(f"  • Better generalization to new data")
    
    print(f"\n🛠️ NEXT STEPS:")
    print(f"1. Run: python train_cleaned_model.py")
    print(f"2. Monitor training accuracy improvements")
    print(f"3. If accuracy is still low, consider:")
    print(f"   - Further data augmentation")
    print(f"   - Different model architectures")
    print(f"   - Hyperparameter tuning")
    print(f"   - Ensemble methods")
    
    print(f"\n📁 FILES CREATED:")
    print(f"  • ../data/cleaned/ - Balanced dataset")
    print(f"  • train_cleaned_model.py - Training script for cleaned data")
    print(f"  • Dataset analysis scripts")
    
    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE - READY FOR IMPROVED TRAINING")
    print("=" * 60)

if __name__ == "__main__":
    main()
