#!/usr/bin/env python3
"""
Script to identify problematic images that might be causing low training accuracy.
This includes corrupted images, very small/large files, and images that are consistently misclassified.
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
from collections import defaultdict
import json

# Add parent directory to path
sys.path.append('..')

def check_image_integrity(image_path):
    """Check if an image can be loaded and is valid"""
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify it's a valid image
        return True, None
    except Exception as e:
        return False, str(e)

def get_image_properties(image_path):
    """Get image properties like size, file size, etc."""
    try:
        file_size = os.path.getsize(image_path)
        with Image.open(image_path) as img:
            width, height = img.size
            mode = img.mode
            format_name = img.format
        return {
            'file_size': file_size,
            'width': width,
            'height': height,
            'mode': mode,
            'format': format_name,
            'aspect_ratio': width / height if height > 0 else 0
        }
    except Exception as e:
        return {'error': str(e)}

def analyze_dataset_images():
    """Analyze all images in the dataset for potential issues"""
    
    print("🔍 Analyzing dataset for problematic images...")
    
    # Load breed labels
    df = pd.read_csv('../annotations/breed_labels.csv')
    data_dir = '../data/raw'
    
    problematic_images = []
    image_stats = defaultdict(list)
    breed_counts = defaultdict(int)
    
    total_images = 0
    processed = 0
    
    for _, row in df.iterrows():
        folder_name = row["folder_name"]
        breed_name = row["breed_name"]
        species = row["species"]
        folder_path = os.path.join(data_dir, species.upper(), folder_name)
        
        if not os.path.exists(folder_path):
            print(f"⚠️ Missing folder: {folder_path}")
            continue
        
        print(f"📁 Processing {breed_name} ({species})...")
        
        image_files = [f for f in os.listdir(folder_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.JPG'))]
        
        breed_counts[breed_name] = len(image_files)
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            total_images += 1
            
            # Check image integrity
            is_valid, error = check_image_integrity(img_path)
            if not is_valid:
                problematic_images.append({
                    'path': img_path,
                    'breed': breed_name,
                    'issue': 'Corrupted/Invalid',
                    'error': error
                })
                continue
            
            # Get image properties
            props = get_image_properties(img_path)
            if 'error' in props:
                problematic_images.append({
                    'path': img_path,
                    'breed': breed_name,
                    'issue': 'Properties Error',
                    'error': props['error']
                })
                continue
            
            # Check for problematic properties
            issues = []
            
            # File size checks
            if props['file_size'] < 1000:  # Less than 1KB
                issues.append(f"Too small: {props['file_size']} bytes")
            elif props['file_size'] > 10 * 1024 * 1024:  # More than 10MB
                issues.append(f"Too large: {props['file_size']} bytes")
            
            # Dimension checks
            if props['width'] < 50 or props['height'] < 50:
                issues.append(f"Too small dimensions: {props['width']}x{props['height']}")
            elif props['width'] > 8000 or props['height'] > 8000:
                issues.append(f"Too large dimensions: {props['width']}x{props['height']}")
            
            # Aspect ratio checks (very wide or very tall)
            if props['aspect_ratio'] < 0.3 or props['aspect_ratio'] > 3.0:
                issues.append(f"Extreme aspect ratio: {props['aspect_ratio']:.2f}")
            
            # Mode checks (not RGB)
            if props['mode'] not in ['RGB', 'L']:  # Not RGB or grayscale
                issues.append(f"Unusual color mode: {props['mode']}")
            
            if issues:
                problematic_images.append({
                    'path': img_path,
                    'breed': breed_name,
                    'issue': '; '.join(issues),
                    'properties': props
                })
            
            # Store stats for analysis
            image_stats[breed_name].append(props)
            processed += 1
            
            if processed % 100 == 0:
                print(f"  Processed {processed}/{total_images} images...")
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Total images processed: {total_images}")
    print(f"⚠️ Problematic images found: {len(problematic_images)}")
    
    # Print breed statistics
    print(f"\n📈 Images per breed:")
    for breed, count in sorted(breed_counts.items()):
        print(f"  {breed}: {count} images")
    
    # Print problematic images summary
    if problematic_images:
        print(f"\n🚨 Problematic images summary:")
        issue_counts = defaultdict(int)
        for img in problematic_images:
            issue_counts[img['issue']] += 1
        
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {issue}: {count} images")
    
    # Save problematic images list
    if problematic_images:
        with open('problematic_images.json', 'w') as f:
            json.dump(problematic_images, f, indent=2)
        print(f"\n💾 Saved problematic images list to 'problematic_images.json'")
    
    # Save image statistics
    with open('image_statistics.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        stats_for_json = {}
        for breed, props_list in image_stats.items():
            stats_for_json[breed] = []
            for props in props_list:
                clean_props = {}
                for k, v in props.items():
                    if isinstance(v, np.integer):
                        clean_props[k] = int(v)
                    elif isinstance(v, np.floating):
                        clean_props[k] = float(v)
                    else:
                        clean_props[k] = v
                stats_for_json[breed].append(clean_props)
        
        json.dump(stats_for_json, f, indent=2)
    
    return problematic_images, breed_counts, image_stats

def recommend_removal_strategy(problematic_images, breed_counts):
    """Recommend which images to remove based on analysis"""
    
    print(f"\n🎯 REMOVAL RECOMMENDATIONS:")
    
    # Group by issue type
    issue_groups = defaultdict(list)
    for img in problematic_images:
        issue_groups[img['issue']].append(img)
    
    recommendations = []
    
    # Always remove corrupted images
    if 'Corrupted/Invalid' in issue_groups:
        corrupted = issue_groups['Corrupted/Invalid']
        recommendations.append({
            'type': 'Corrupted Images',
            'count': len(corrupted),
            'action': 'DELETE',
            'reason': 'Cannot be processed by the model',
            'images': corrupted
        })
    
    # Remove extremely small images
    if any('Too small:' in img['issue'] for img in problematic_images):
        small_images = [img for img in problematic_images if 'Too small:' in img['issue']]
        recommendations.append({
            'type': 'Very Small Images',
            'count': len(small_images),
            'action': 'DELETE',
            'reason': 'Too small to contain useful features',
            'images': small_images
        })
    
    # Remove images with extreme aspect ratios
    if any('Extreme aspect ratio:' in img['issue'] for img in problematic_images):
        extreme_ratio = [img for img in problematic_images if 'Extreme aspect ratio:' in img['issue']]
        recommendations.append({
            'type': 'Extreme Aspect Ratio',
            'count': len(extreme_ratio),
            'action': 'DELETE',
            'reason': 'Extreme dimensions may confuse the model',
            'images': extreme_ratio
        })
    
    # For class imbalance, recommend removing excess from over-represented classes
    max_images = max(breed_counts.values())
    min_images = min(breed_counts.values())
    
    if max_images / min_images > 3:  # Significant imbalance
        print(f"\n⚠️ Class imbalance detected: {max_images/min_images:.1f}x difference")
        
        for breed, count in breed_counts.items():
            if count > min_images * 2:  # More than 2x the minimum
                excess = count - int(min_images * 1.5)  # Keep 1.5x minimum
                if excess > 0:
                    recommendations.append({
                        'type': f'Excess {breed} images',
                        'count': excess,
                        'action': 'REMOVE_RANDOM',
                        'reason': f'Reduce class imbalance (has {count}, recommend {int(min_images * 1.5)})',
                        'breed': breed
                    })
    
    # Print recommendations
    for rec in recommendations:
        print(f"\n{rec['action']}: {rec['count']} images")
        print(f"  Type: {rec['type']}")
        print(f"  Reason: {rec['reason']}")
    
    return recommendations

if __name__ == "__main__":
    problematic_images, breed_counts, image_stats = analyze_dataset_images()
    recommendations = recommend_removal_strategy(problematic_images, breed_counts)
    
    print(f"\n💡 Next steps:")
    print(f"1. Review the problematic_images.json file")
    print(f"2. Consider removing the recommended images")
    print(f"3. Retrain the model with the cleaned dataset")
    print(f"4. Monitor accuracy improvements")
