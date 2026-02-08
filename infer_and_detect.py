import numpy as np
import joblib
from collections import deque

# Load artifacts
model = joblib.load("distress_model.pkl")
scaler = joblib.load("scaler.pkl")

# Parameters (TUNE THESE, NOT THE MODEL)
ALPHA = 0.8                  # smoothing factor
DISTRESS_THRESHOLD = 0.6     # trigger threshold
WINDOW = 5                   # last N frames
REQUIRED_HITS = 3            # how many must exceed threshold

# State
history = deque(maxlen=WINDOW)
prev_smoothed = None


def smooth(curr, prev, alpha=ALPHA):
    if prev is None:
        return curr
    return alpha * prev + (1 - alpha) * curr


def is_distressed(hist):
    return sum(s > DISTRESS_THRESHOLD for s in hist) >= REQUIRED_HITS


def process_feature_vector(feat_vec):
    global prev_smoothed

    # Scale
    feat_vec = scaler.transform(feat_vec.reshape(1, -1))

    # Predict distress score
    raw_score = model.predict(feat_vec)[0]
    raw_score = np.clip(raw_score, 0.0, 1.0)

    # Smooth
    smoothed = smooth(raw_score, prev_smoothed)
    prev_smoothed = smoothed

    # Update history
    history.append(smoothed)

    # Final decision
    distressed = is_distressed(history)

    return {
        "raw": raw_score,
        "smoothed": smoothed,
        "distressed": distressed
    }