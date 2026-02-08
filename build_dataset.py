import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm
import io
import soundfile as sf

from features import extract_features_from_array

# Load dataset with audio decoding DISABLED
ds = load_dataset("Hemg/Emotion-audio-Dataset", split="train")
ds = ds.cast_column("audio", Audio(decode=False))

label_names = ds.features["label"].names

AROUSAL_MAP = {
    "angry": 1.0,
    "fearful": 1.0,
    "disgusted": 0.8,
    "suprised": 0.7,
    "sad": 0.5,
    "neutral": 0.0,
    "happy": 0.0,
}

X = []
y = []

audio_col = ds["audio"]
label_col = ds["label"]

for audio, label_id in tqdm(zip(audio_col, label_col), total=len(label_col)):
    if audio is None or "bytes" not in audio:
        continue

    try:
        # Decode audio bytes manually
        y_arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    except Exception:
        continue

    feat = extract_features_from_array(y_arr, sr)
    if feat is None:
        continue

    label_name = label_names[label_id].lower()
    arousal = AROUSAL_MAP.get(label_name, 0.0)

    X.append(feat)
    y.append(arousal)

X = np.array(X)
y = np.array(y)

np.save("X.npy", X)
np.save("y.npy", y)

print("Final dataset shape:", X.shape)