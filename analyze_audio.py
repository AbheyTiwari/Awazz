import librosa
import numpy as np
import os

# Files to compare
files = ['convo.wav', 'aaa.wav', 'horror.wav', 'hey.wav']

print(f"{'FILE':<12} | {'RMS (Loudness)':<15} | {'ZCR (Noisiness)':<15} | {'CENTROID (Pitch/Bright)':<20}")
print("-" * 70)

for file in files:
    if not os.path.exists(file):
        print(f"{file:<12} | [FILE NOT FOUND]")
        continue
    
    # Load audio
    y, sr = librosa.load(file)
    
    # 1. RMS Energy (Loudness)
    rms = np.mean(librosa.feature.rms(y=y))
    
    # 2. Zero Crossing Rate (Rough proxy for noisiness/high frequency)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # 3. Spectral Centroid (Perceived "brightness" or high pitch)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    print(f"{file:<12} | {rms:.4f}          | {zcr:.4f}          | {centroid:.2f}")

print("-" * 70)
print("INTERPRETATION:")
print("- If convo.wav has higher RMS, your model is just biased toward volume.")
print("- If convo.wav has higher Centroid, it's confusing speech pitch with scream pitch.")