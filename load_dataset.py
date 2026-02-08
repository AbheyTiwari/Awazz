import os
import pandas as pd
from datasets import load_dataset, Audio

AROUSAL_MAP = {
    "angry": 1.0,
    "fearful": 1.0,
    "disgusted": 0.8,
    "suprised": 0.7,
    "sad": 0.5,
    "neutral": 0.0,
    "happy": 0.0,
}

def load_and_prepare():
    ds = load_dataset("Hemg/Emotion-audio-Dataset", split="train")
    ds = ds.cast_column("audio", Audio(decode=False))

    # 🔑 Find dataset cache directory
    cache_file = ds.cache_files[0]["filename"]
    base_dir = os.path.dirname(cache_file)

    label_names = ds.features["label"].names
    rows = []

    for audio, label_id in zip(ds["audio"], ds["label"]):
        if audio is None or "path" not in audio:
            continue

        # 🔑 REAL full path
        audio_path = os.path.join(base_dir, audio["path"])

        label_name = label_names[label_id].lower()
        arousal = AROUSAL_MAP.get(label_name, 0.0)

        rows.append({
            "path": audio_path,
            "arousal": arousal
        })

    df = pd.DataFrame(rows)
    df.to_csv("data.csv", index=False)

    print(df.head())
    print("Total samples:", len(df))


if __name__ == "__main__":
    load_and_prepare()