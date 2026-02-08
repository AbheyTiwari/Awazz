# Audio Distress Detection System

A machine learning system that detects distress in audio signals using acoustic features and a Random Forest regression model. The system analyzes voice recordings to predict arousal levels and identify potentially distressed states.

## Overview

This project uses audio feature extraction (MFCCs, pitch, energy, zero-crossing rate) to train a model that predicts arousal/distress levels from voice recordings. It includes temporal smoothing and threshold-based detection logic to minimize false positives.

## Features

- **Audio Feature Extraction**: Extracts 22 acoustic features including RMS energy, zero-crossing rate, pitch statistics, and MFCCs
- **Arousal Mapping**: Maps emotional states to arousal levels (0.0 = calm, 1.0 = high arousal)
- **Random Forest Regressor**: Non-linear model trained on emotion-labeled audio data
- **Temporal Smoothing**: Exponential smoothing to reduce frame-to-frame jitter
- **Threshold Detection**: Configurable sliding window with hit requirements
- **Hard Override**: Direct detection of scream-like signals based on energy and zero-crossing rate

## Project Structure

```
.
├── README.md                    # This file
├── features.py                  # Audio feature extraction module
├── inspect_datasets.py          # Dataset exploration script
├── load_dataset.py              # Dataset loader and CSV generator
├── build_dataset.py             # Feature extraction and dataset builder
├── train.py                     # Model training script
├── test_distress.py             # Batch testing script for WAV files
├── infer_and_detect.py          # Real-time inference module
├── data.csv                     # Audio file paths and arousal labels
├── X.npy                        # Feature matrix (samples × features)
├── y.npy                        # Target arousal values
├── distress_model.pkl           # Trained Random Forest model
└── scaler.pkl                   # Feature scaler (StandardScaler)
```

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn librosa soundfile datasets tqdm joblib
```

### Python Version

Python 3.8 or higher recommended.

## Usage

### 1. Inspect the Dataset

Explore the Hugging Face dataset structure:

```bash
python inspect_datasets.py
```

### 2. Prepare the Dataset

Load the emotion dataset and create a CSV with file paths and arousal labels:

```bash
python load_dataset.py
```

This generates `data.csv` with columns:
- `path`: Full path to audio file
- `arousal`: Mapped arousal score (0.0–1.0)

### 3. Build Feature Dataset

Extract acoustic features from all audio files:

```bash
python build_dataset.py
```

This creates:
- `X.npy`: Feature matrix (N × 22)
- `y.npy`: Arousal labels (N,)

### 4. Train the Model

Train the Random Forest regressor:

```bash
python train.py
```

This outputs:
- `distress_model.pkl`: Trained model
- `scaler.pkl`: Feature scaler
- Evaluation metrics (MSE, MAE, R²)

### 5. Test on WAV Files

Run batch distress detection on audio files:

```bash
python test_distress.py file1.wav file2.wav file3.wav
```

**Output Example:**
```
File: angry_scream.wav
  Raw score     : 0.923
  Smoothed score: 0.923
  Distressed    : YES

File: calm_speech.wav
  Raw score     : 0.142
  Smoothed score: 0.235
  Distressed    : NO

Final verdict: DISTRESSED
```

### 6. Real-time Inference

Use the inference module in your application:

```python
from infer_and_detect import process_feature_vector
from features import extract_features_from_array
import soundfile as sf

# Load audio
y, sr = sf.read("audio.wav", dtype="float32")

# Extract features
feat = extract_features_from_array(y, sr)

# Get distress prediction
result = process_feature_vector(feat)

print(f"Distressed: {result['distressed']}")
print(f"Smoothed score: {result['smoothed']:.3f}")
```

## Feature Extraction

The system extracts 22 features per audio frame:

| Feature Group | Features | Description |
|---------------|----------|-------------|
| **Energy** | RMS mean, RMS std | Root Mean Square energy statistics |
| **Zero-Crossing Rate** | ZCR mean, ZCR std | Frequency content indicator |
| **Pitch** | Pitch mean, Pitch std | Fundamental frequency statistics |
| **MFCCs** | 8 mean + 8 std | Mel-frequency cepstral coefficients |

### Audio Preprocessing

- Resampling to 8 kHz (SR)
- Silence trimming (top_db=30)
- Maximum duration capping (3 seconds)
- Frame size: 25ms, Hop: 10ms
- FFT size: 512

## Arousal Mapping

Emotions are mapped to arousal levels based on psychological arousal theory:

| Emotion | Arousal Score |
|---------|---------------|
| Angry | 1.0 |
| Fearful | 1.0 |
| Disgusted | 0.8 |
| Surprised | 0.7 |
| Sad | 0.5 |
| Neutral | 0.0 |
| Happy | 0.0 |

## Detection Logic

### Temporal Smoothing

Exponential moving average to reduce noise:

```python
smoothed = α × previous + (1 - α) × current
```

Default α = 0.6 (test_distress.py) or 0.8 (infer_and_detect.py)

### Threshold Detection

Uses a sliding window approach:

- **WINDOW**: Number of recent frames to consider (default: 3–5)
- **DISTRESS_THRESHOLD**: Minimum smoothed score to count as distressed (default: 0.5–0.6)
- **REQUIRED_HITS**: Minimum frames above threshold in window (default: 2–3)

### Hard Override (test_distress.py)

Direct distress detection for extreme signals:

```python
if energy > 0.03 and zcr > 0.15:
    # Scream/panic detected
    return distressed = True
```

## Model Configuration

### Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=300,      # Number of trees
    max_depth=12,          # Maximum tree depth
    min_samples_leaf=5,    # Minimum samples per leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)
```

### Hyperparameter Tuning

To adjust detection sensitivity, modify **detection parameters** (not the model):

**In `test_distress.py`:**
```python
ALPHA = 0.6                # Smoothing (higher = more smoothing)
DISTRESS_THRESHOLD = 0.5   # Trigger threshold
WINDOW = 3                 # Lookback frames
REQUIRED_HITS = 2          # Minimum hits in window
```

**In `infer_and_detect.py`:**
```python
ALPHA = 0.8
DISTRESS_THRESHOLD = 0.6
WINDOW = 5
REQUIRED_HITS = 3
```

## Dataset

The project uses the [Hemg/Emotion-audio-Dataset](https://huggingface.co/datasets/Hemg/Emotion-audio-Dataset) from Hugging Face, which contains labeled emotional speech recordings.

## Performance Metrics

After training, the model reports:

- **MSE** (Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average absolute prediction error
- **R² Score**: Proportion of variance explained (higher = better)

Typical performance on test set:
- R² ≈ 0.6–0.8 (depending on dataset quality)

## Limitations

- **Single Speaker**: Model trained on specific emotion dataset
- **Language Dependent**: May not generalize across languages
- **Background Noise**: Performance degrades in noisy environments
- **Short Audio**: Requires minimum audio length (~200ms)

## Future Improvements

- [ ] Multi-dataset training for better generalization
- [ ] Deep learning models (CNN/RNN) for improved accuracy
- [ ] Real-time streaming audio support
- [ ] Noise robustness improvements
- [ ] Multi-language support
- [ ] Gender/age-specific models

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- Dataset: [Hemg/Emotion-audio-Dataset](https://huggingface.co/datasets/Hemg/Emotion-audio-Dataset)
- Audio processing: `librosa` library
- Machine learning: `scikit-learn`

## Contact

For questions or issues, please refer to the source code documentation or file an issue in your repository.