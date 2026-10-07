#!/usr/bin/env python3
"""
Script to clean the dataset by removing problematic images and balancing classes.
"""

import os
import pandas as pd
import random
import shutil
from PIL import Image
from collections import defaultdict
import json

def check_image_quality(image_path):
    """Check if an image is of good quality"""
    try:
        with Image.open(image_path) as img:
            # Check if image can be loaded
            img.verify()
            
            # Reopen for size check (verify() closes the image)
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)
                
                # Quality checks
                issues = []
                
                # File size checks
                if file_size < 2000:  # Less than 2KB
                    issues.append("too_small")
                elif file_size > 20 * 1024 * 1024:  # More than 20MB
                    issues.append("too_large")
                
                # Dimension checks
                if width < 100 or height < 100:
                    issues.append("dimensions_too_small")
                elif width > 6000 or height > 6000:
                    issues.append("dimensions_too_large")
                
                # Aspect ratio checks
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                    issues.append("extreme_aspect_ratio")
                
                # Mode checks
                if img.mode not in ['RGB', 'L']:
                    issues.append("unusual_color_mode")
                
                return len(issues) == 0, issues
                
    except Exception as e:
        return False, [f"load_error: {str(e)}"]

def analyze_and_clean_dataset():
    """Analyze dataset and create cleaned version"""
    
    print("Starting dataset analysis and cleaning...")
    
    # Load breed labels
    df = pd.read_csv('../annotations/breed_labels.csv')
    data_dir = '../data/raw'
    clean_data_dir = '../data/cleaned'
    
    # Create cleaned data directory
    if os.path.exists(clean_data_dir):
        shutil.rmtree(clean_data_dir)
    os.makedirs(clean_data_dir, exist_ok=True)
    
    breed_counts = {}
    problematic_images = []
    cleaned_images = 0
    total_processed = 0
    
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
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        
        breed_clean_count = 0
        breed_problematic = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            total_processed += 1
            
            is_good, issues = check_image_quality(img_path)
            
            if is_good:
                # Copy to cleaned directory
                clean_img_path = os.path.join(clean_breed_dir, img_file)
                shutil.copy2(img_path, clean_img_path)
                breed_clean_count += 1
                cleaned_images += 1
            else:
                breed_problematic.append({
                    'path': img_path,
                    'issues': issues
                })
                problematic_images.append({
                    'path': img_path,
                    'breed': breed_name,
                    'issues': issues
                })
            
            if total_processed % 100 == 0:
                print(f"  Processed {total_processed} images, cleaned {cleaned_images}")
        
        breed_counts[breed_name] = {
            'original': len(image_files),
            'cleaned': breed_clean_count,
            'removed': len(breed_problematic)
        }
        
        print(f"  {breed_name}: {len(image_files)} -> {breed_clean_count} images")
    
    print(f"\nCleaning complete!")
    print(f"Total processed: {total_processed}")
    print(f"Cleaned images: {cleaned_images}")
    print(f"Removed images: {len(problematic_images)}")
    
    # Balance classes by removing excess images from over-represented classes
    print(f"\nBalancing classes...")
    
    # Calculate target size (median of cleaned counts)
    cleaned_counts = [info['cleaned'] for info in breed_counts.values()]
    target_size = sorted(cleaned_counts)[len(cleaned_counts) // 2]  # Median
    max_allowed = int(target_size * 1.5)  # Allow 50% more than median
    
    print(f"Target size (median): {target_size}")
    print(f"Max allowed per breed: {max_allowed}")
    
    # Remove excess images from over-represented breeds
    excess_removed = 0
    for breed_name, info in breed_counts.items():
        if info['cleaned'] > max_allowed:
            excess_count = info['cleaned'] - max_allowed
            breed_dir = None
            
            # Find the breed directory
            for _, row in df.iterrows():
                if row["breed_name"] == breed_name:
                    species = row["species"]
                    folder_name = row["folder_name"]
                    breed_dir = os.path.join(clean_data_dir, species.upper(), folder_name)
                    break
            
            if breed_dir and os.path.exists(breed_dir):
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
                
                breed_counts[breed_name]['final'] = info['cleaned'] - excess_count
                print(f"  {breed_name}: Removed {excess_count} excess images")
            else:
                breed_counts[breed_name]['final'] = info['cleaned']
        else:
            breed_counts[breed_name]['final'] = info['cleaned']
    
    print(f"Removed {excess_removed} excess images for balancing")
    
    # Final statistics
    final_counts = [info['final'] for info in breed_counts.values()]
    final_total = sum(final_counts)
    min_final = min(final_counts)
    max_final = max(final_counts)
    final_imbalance = max_final / min_final if min_final > 0 else float('inf')
    
    print(f"\nFinal dataset statistics:")
    print(f"Total images: {final_total}")
    print(f"Min per breed: {min_final}")
    print(f"Max per breed: {max_final}")
    print(f"Imbalance ratio: {final_imbalance:.2f}x")
    
    print(f"\nPer-breed breakdown:")
    for breed_name, info in breed_counts.items():
        print(f"  {breed_name}: {info['original']} -> {info['final']} images")
    
    # Save problematic images list
    if problematic_images:
        with open('removed_problematic_images.json', 'w') as f:
            json.dump(problematic_images, f, indent=2)
        print(f"\nSaved list of removed images to 'removed_problematic_images.json'")
    
    # Save cleaning report
    cleaning_report = {
        'original_total': total_processed,
        'cleaned_total': cleaned_images,
        'final_total': final_total,
        'problematic_removed': len(problematic_images),
        'excess_removed': excess_removed,
        'breed_statistics': breed_counts,
        'final_imbalance_ratio': final_imbalance
    }
    
    with open('dataset_cleaning_report.json', 'w') as f:
        json.dump(cleaning_report, f, indent=2)
    
    print(f"\nDataset cleaning complete!")
    print(f"Cleaned dataset saved to: {clean_data_dir}")
    print(f"Cleaning report saved to: dataset_cleaning_report.json")
    
    return clean_data_dir, cleaning_report

if __name__ == "__main__":
    clean_dir, report = analyze_and_clean_dataset()
    
    print(f"\nNext steps:")
    print(f"1. Update your training script to use the cleaned dataset at: {clean_dir}")
    print(f"2. Retrain your model with the balanced dataset")
    print(f"3. Monitor accuracy improvements")
    print(f"4. The imbalance ratio improved from 3.63x to {report['final_imbalance_ratio']:.2f}x")
