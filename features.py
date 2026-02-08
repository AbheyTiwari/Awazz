import numpy as np
import librosa

SR = 8000

# Analysis window: max 3 seconds
MAX_SECONDS = 3
MAX_SAMPLES = SR * MAX_SECONDS

FRAME = int(0.025 * SR)   # 25 ms
HOP = int(0.010 * SR)     # 10 ms
N_FFT = 512               # SMALL, controlled

def extract_features_from_array(y, sr):
    # Basic sanity
    if y is None:
        return None

    # Flatten stereo safely
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)

    # Trim silence (critical)
    y, _ = librosa.effects.trim(y, top_db=30)

    # Skip too-short audio
    if len(y) < FRAME:
        return None

    # Cap max length (CRITICAL FIX)
    if len(y) > MAX_SAMPLES:
        y = y[:MAX_SAMPLES]

    # RMS energy
    rms = librosa.feature.rms(
        y=y,
        frame_length=FRAME,
        hop_length=HOP
    )[0]

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(
        y,
        frame_length=FRAME,
        hop_length=HOP
    )[0]

    # Pitch (SAFE configuration)
    pitches, mags = librosa.piptrack(
        y=y,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP
    )

    pitch = pitches[mags > np.median(mags)]
    pitch_mean = np.mean(pitch) if pitch.size else 0.0
    pitch_std = np.std(pitch) if pitch.size else 0.0

    # MFCCs (safe FFT)
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=8,
        n_fft=N_FFT,
        hop_length=HOP
    )

    features = [
        np.mean(rms), np.std(rms),
        np.mean(zcr), np.std(zcr),
        pitch_mean, pitch_std
    ]

    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    return np.array(features, dtype=np.float32)