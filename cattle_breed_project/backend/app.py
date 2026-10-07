#!/usr/bin/env python3
"""
Flask backend API for cattle breed identification
Serves the trained ML model for the React frontend
"""

import os
import io
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Configuration
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "best_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "..", "annotations", "breed_labels.csv")

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model class definition (matching the training script)
class CattleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CattleClassifier, self).__init__()        
        # Use the same architecture as in training
        self.model = models.resnet50(pretrained=False)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# Breed information mapping
BREED_INFO = {
    'toda': {
        'origin': 'Tamil Nadu, India',
        'primary_use': 'Milk production',
        'average_weight': '450-550 kg',
        'distinctive_features': 'Small to medium sized, hardy, adapted to hills'
    },
    'banni': {
        'origin': 'Gujarat, India',
        'primary_use': 'Milk production',
        'average_weight': '400-500 kg',
        'distinctive_features': 'Grey to black color, compact body'
    },
    'jaffarabadi': {
        'origin': 'Gujarat, India',
        'primary_use': 'Milk and draft',
        'average_weight': '500-700 kg',
        'distinctive_features': 'Large size, curved horns, black color'
    },
    'mehsana': {
        'origin': 'Gujarat, India',
        'primary_use': 'Milk production',
        'average_weight': '450-550 kg',
        'distinctive_features': 'Medium size, black color, good milk yield'
    },
    'murrah': {
        'origin': 'Haryana, India',
        'primary_use': 'Milk production',
        'average_weight': '500-650 kg',
        'distinctive_features': 'Black color, curved horns, high milk yield'
    },
    'nagpuri': {
        'origin': 'Maharashtra, India',
        'primary_use': 'Draft and milk',
        'average_weight': '400-500 kg',
        'distinctive_features': 'Medium size, black color, good draught power'
    },
    'niliravi': {
        'origin': 'Punjab, Pakistan',
        'primary_use': 'Milk production',
        'average_weight': '450-550 kg',
        'distinctive_features': 'Black color, white markings, curved horns'
    },
    'bargur': {
        'origin': 'Tamil Nadu, India',
        'primary_use': 'Draft and milk',
        'average_weight': '350-450 kg',
        'distinctive_features': 'Small to medium, grey color, hardy'
    },
    'bhadawari': {
        'origin': 'Uttar Pradesh, India',
        'primary_use': 'Milk production',
        'average_weight': '350-400 kg',
        'distinctive_features': 'Small size, brown color, efficient converter'
    },
    'ongole': {
        'origin': 'Andhra Pradesh, India',
        'primary_use': 'Draft and beef',
        'average_weight': '500-700 kg',
        'distinctive_features': 'White/grey color, large size, prominent hump'
    },
    'gir': {
        'origin': 'Gujarat, India',
        'primary_use': 'Milk production',
        'average_weight': '350-450 kg',
        'distinctive_features': 'Red/white color, curved horns, good milk yield'
    },
    'kankrej': {
        'origin': 'Gujarat, India',
        'primary_use': 'Draft and milk',
        'average_weight': '450-550 kg',
        'distinctive_features': 'Silver grey color, lyra-shaped horns'
    },
    'amritmahal': {
        'origin': 'Karnataka, India',
        'primary_use': 'Draft work',
        'average_weight': '400-500 kg',
        'distinctive_features': 'Grey color, compact body, excellent draught'
    },
    'hallikar': {
        'origin': 'Karnataka, India',
        'primary_use': 'Draft work',
        'average_weight': '350-450 kg',
        'distinctive_features': 'Grey color, medium size, good draught power'
    },
    'krishnavalley': {
        'origin': 'Karnataka, India',
        'primary_use': 'Draft and milk',
        'average_weight': '400-500 kg',
        'distinctive_features': 'Black color, medium size, dual purpose'
    },
    'vechur': {
        'origin': 'Kerala, India',
        'primary_use': 'Milk production',
        'average_weight': '150-200 kg',
        'distinctive_features': 'Very small size, brown/red color, rare breed'
    },
    'nimari': {
        'origin': 'Madhya Pradesh, India',
        'primary_use': 'Draft and milk',
        'average_weight': '350-450 kg',
        'distinctive_features': 'White/grey color, medium size, hardy'
    }
}

# Global variables for model and data
model = None
label2idx = None
idx2label = None

def load_model():
    """Load the trained model and labels"""
    global model, label2idx, idx2label
    
    # Load breed labels
    df = pd.read_csv(LABELS_PATH)
    breeds = sorted(df["breed_name"].unique())
    label2idx = {breed: i for i, breed in enumerate(breeds)}
    idx2label = {i: breed for breed, i in label2idx.items()}
    
    print(f"Loaded {len(breeds)} breeds: {breeds}")
    
    # Try loading the comprehensive model first
    alt_model_path = os.path.join(BASE_DIR, "..", "models", "Best_model_final.pth")
    if os.path.exists(alt_model_path):
        try:
            checkpoint = torch.load(alt_model_path, map_location=DEVICE)
            
            # Get saved metadata
            saved_breeds = checkpoint.get('breeds', breeds)
            saved_num_classes = checkpoint.get('num_classes', len(breeds))
            
            print(f"Checkpoint contains {saved_num_classes} classes: {saved_breeds}")
            
            # Initialize model with saved architecture
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, saved_num_classes)
            )
            
            # Load the model state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(DEVICE)
            model.eval()
            
            # Update the label mappings with saved ones if available
            if 'label2idx' in checkpoint:
                label2idx = checkpoint['label2idx']
                idx2label = {i: breed for breed, i in label2idx.items()}
                print(f"Using saved label mappings: {list(label2idx.keys())}")
            
            print(f"Model loaded successfully from {alt_model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading comprehensive model: {e}")
    
    # Fallback to simple model loading
    if os.path.exists(MODEL_PATH):
        try:
            # Initialize model
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.4),
                nn.Linear(num_features, len(breeds))
            )
            
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            print(f"Model loaded successfully from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Error loading simple model: {e}")
    
    print(f"Warning: No model files could be loaded")
    return False

def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(DEVICE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    })

@app.route('/breeds', methods=['GET'])
def get_breeds():
    """Get list of all supported breeds with their information"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    breeds_list = []
    for breed_name, breed_id in label2idx.items():
        breed_data = {
            'id': breed_name,
            'name': breed_name.title(),
            'characteristics': BREED_INFO.get(breed_name, {
                'origin': 'India',
                'primary_use': 'Agricultural',
                'average_weight': 'Variable',
                'distinctive_features': 'Indigenous breed'
            })
        }
        breeds_list.append(breed_data)
    
    return jsonify({'breeds': breeds_list})

@app.route('/predict', methods=['POST'])
def predict_breed():
    """Predict cattle breed from uploaded image"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Load and preprocess image
        image = Image.open(io.BytesIO(file.read()))
        image_tensor = preprocess_image(image)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(label2idx)))
            
            results = []
            for i in range(len(top_indices[0])):
                breed_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                breed_name = idx2label[breed_idx]
                
                result = {
                    'breed_name': breed_name.title(),
                    'confidence': float(confidence),
                    'breed_id': breed_name,
                    'characteristics': BREED_INFO.get(breed_name, {
                        'origin': 'India',
                        'primary_use': 'Agricultural',
                        'average_weight': 'Variable',
                        'distinctive_features': 'Indigenous breed'
                    })
                }
                results.append(result)
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict-base64', methods=['POST'])
def predict_breed_base64():
    """Predict cattle breed from base64 encoded image"""
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_tensor = preprocess_image(image)
        
        # Make prediction (same as above)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, k=min(3, len(label2idx)))
            
            results = []
            for i in range(len(top_indices[0])):
                breed_idx = top_indices[0][i].item()
                confidence = top_probs[0][i].item()
                breed_name = idx2label[breed_idx]
                
                result = {
                    'breed_name': breed_name.title(),
                    'confidence': float(confidence),
                    'breed_id': breed_name,
                    'characteristics': BREED_INFO.get(breed_name, {
                        'origin': 'India',
                        'primary_use': 'Agricultural',
                        'average_weight': 'Variable',
                        'distinctive_features': 'Indigenous breed'
                    })
                }
                results.append(result)
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Cattle Breed Identification API...")
    
    # Load model
    if load_model():
        print("✅ Model loaded successfully")
        print("🚀 Starting Flask server on http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Please check model path.")