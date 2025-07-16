# Emergency SOS Model Combiner

## Your Models
- keyword_model_quant.tflite: 236.00 KB - Detects emergency keywords
- quantized_emotion_recognition_model.tflite: 564.05 KB - Emotion analysis
- screamdetector_model.tflite: 34,003.30 KB - Scream detection

## Setup Instructions

### Option 1: Use Python 3.9-3.11 (Recommended)
```bash
# Create virtual environment
py -3.11 -m venv sos_env
sos_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the combiner
py combine_models.py
```

### Option 2: Use Google Colab
1. Upload all .tflite files to Google Colab
2. Copy combine_models.py to a Colab notebook
3. Run the script

## How it Works
1. Keyword Detection (40%): Detects "help", "save me", etc.
2. Emotion Recognition (30%): Analyzes emotional distress
3. Scream Detection (30%): Identifies screaming sounds
4. Final Score: Weighted combination triggers SOS when > 0.7
