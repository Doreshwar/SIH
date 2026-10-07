# Cattle Breed Identification System - Integration

This project integrates the trained ML model with the React frontend for cattle breed identification.

## System Architecture

- **Frontend**: React.js with TypeScript (cattle-sense-main)
- **Backend**: Flask API serving the trained PyTorch model
- **Model**: Trained ResNet50 for Indian cattle and buffalo breed classification

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Option 1: Automated Setup (Windows)
```bash
# Double-click or run:
start_system.bat
```

### Option 2: Manual Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
python app.py
```

#### Frontend Setup
```bash
cd cattle-sense-main
npm install
npm run dev
```

## Usage

1. **Access the Web App**: Open http://localhost:5173
2. **Navigate to Identify**: Click \"Identify\" in the navigation
3. **Upload Image**: Drag & drop or click to upload a cattle image
4. **Get Results**: View AI-powered breed identification with confidence scores

## Supported Breeds

The model recognizes 17 Indian indigenous breeds:

**Cattle Breeds:**
- Gir, Ongole, Kankrej, Amritmahal, Hallikar
- Krishna Valley, Vechur, Nimari

**Buffalo Breeds:**
- Murrah, Jaffarabadi, Nili Ravi, Banni, Mehsana
- Nagpuri, Bargur, Bhadawari, Toda

## API Endpoints

- `GET /health` - Health check
- `GET /breeds` - List all supported breeds
- `POST /predict` - Upload image for prediction
- `POST /predict-base64` - Base64 image prediction

## Features

- **Real-time Prediction**: Upload images and get instant breed identification
- **Confidence Scores**: View prediction confidence for each result
- **Breed Information**: Detailed characteristics for each identified breed
- **Error Handling**: Graceful fallback when backend is unavailable
- **Responsive Design**: Works on desktop and mobile devices

## File Structure

```
cattle_breed_project/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── venv/              # Virtual environment
├── cattle-sense-main/      # React frontend
│   ├── src/
│   │   ├── pages/
│   │   │   └── Identify.tsx # Main identification page
│   │   └── lib/
│   │       └── mockData.ts  # Breed data definitions
│   └── package.json
├── best_model.pth         # Trained model weights
├── annotations/
│   └── breed_labels.csv   # Breed label mappings
└── start_system.bat       # Automated startup script
```

## Troubleshooting

### Backend Issues
- Ensure model file exists at `best_model.pth`
- Check Python dependencies are installed
- Verify Flask is running on port 5000

### Frontend Issues
- Run `npm install` in cattle-sense-main directory
- Check if port 5173 is available
- Ensure backend is running before frontend

### Model Loading Issues
- Verify the model file path in `backend/app.py`
- Check if the model was trained with the same architecture
- Ensure PyTorch version compatibility

## Next Steps

1. **Add More Images**: Expand the training dataset
2. **Improve Accuracy**: Fine-tune the model with more data
3. **Deploy**: Host on cloud platforms for public access
4. **Mobile App**: Create React Native version

## Tech Stack

- **ML Framework**: PyTorch
- **Model**: ResNet50 with custom classifier
- **Backend**: Flask with CORS support
- **Frontend**: React + TypeScript + Vite
- **UI Components**: Radix UI + Tailwind CSS