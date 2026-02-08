import sys
import numpy as np
import joblib
import soundfile as sf
from collections import deque

from features import extract_features_from_array

# --------------------
# Load trained artifacts
# --------------------
model = joblib.load("distress_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------
# Detection parameters (ZONES)
# --------------------
ALPHA = 0.6
WINDOW = 5

CALM_MAX = 0.55        # clearly calm
STRESS_MIN = 0.65     # elevated arousal
DISTRESS_MIN = 0.75   # panic-level

# --------------------
# Helpers
# --------------------
def smooth(curr, prev):
    if prev is None:
        return curr
    return ALPHA * prev + (1 - ALPHA) * curr


def is_distressed(history):
    if not history:
        return False

    recent = list(history)

    distress_hits = sum(x >= DISTRESS_MIN for x in recent)
    stress_hits = sum(x >= STRESS_MIN for x in recent)
    calm_hits = sum(x <= CALM_MAX for x in recent)

    # Immediate panic
    if distress_hits >= 1:
        return True

    # Sustained stress
    if stress_hits >= 3:
        return True

    # Calm actively cancels alarm
    if calm_hits >= 2:
        return False

    return False


def process_wav(path, state):
    try:
        y, sr = sf.read(path, dtype="float32")
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return None

    # Mono
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Loudness normalization
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    # -----------------------------
    # 🔴 HARD SCREAM / PANIC OVERRIDE
    # -----------------------------
    energy = np.mean(y ** 2)
    zcr = np.mean(np.abs(np.diff(np.sign(y))))

    if energy > 0.04 and zcr > 0.18:
        state["history"].append(1.0)
        state["prev_smoothed"] = 1.0

        return {
            "file": path,
            "raw": 1.0,
            "smoothed": 1.0,
            "distressed": True
        }

    # -----------------------------
    # Normal ML path
    # -----------------------------
    feat = extract_features_from_array(y, sr)
    if feat is None:
        print(f"[SKIP] {path}: audio too short or invalid")
        return None

    feat = scaler.transform(feat.reshape(1, -1))
    raw = float(np.clip(model.predict(feat)[0], 0.0, 1.0))

    smoothed = smooth(raw, state["prev_smoothed"])

    # -----------------------------
    # 🟢 CALM RECOVERY (important)
    # -----------------------------
    if raw <= CALM_MAX:
        smoothed *= 0.7   # decay alarm faster on calm input

    state["prev_smoothed"] = smoothed
    state["history"].append(smoothed)

    return {
        "file": path,
        "raw": raw,
        "smoothed": smoothed,
        "distressed": is_distressed(state["history"])
    }


# --------------------
# Entry point
# --------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_distress.py file1.wav [file2.wav ...]")
        sys.exit(1)

    state = {
        "history": deque(maxlen=WINDOW),
        "prev_smoothed": None
    }

    print("\n--- Distress Detection Test ---\n")

    for wav_path in sys.argv[1:]:
        result = process_wav(wav_path, state)
        if result is None:
            continue

        print(f"File: {result['file']}")
        print(f"  Raw score     : {result['raw']:.3f}")
        print(f"  Smoothed score: {result['smoothed']:.3f}")
        print(f"  Distressed    : {'YES' if result['distressed'] else 'NO'}\n")

    print(
        "Final verdict:",
        "DISTRESSED" if is_distressed(state["history"]) else "NOT DISTRESSED"
    )
