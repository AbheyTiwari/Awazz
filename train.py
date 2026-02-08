import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Load data
X = np.load("X.npy")
y = np.load("y.npy")


# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Scale features (not strictly needed for RF, but keep for consistency)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train model (NON-LINEAR, ROBUST)
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)


# Evaluate
preds = model.predict(X_test)
preds = np.clip(preds, 0.0, 1.0)  # keep it sane


mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)


print("MSE:", mse)
print("MAE:", mae)
print("R² score:", r2)


# Save artifacts
joblib.dump(model, "distress_model.pkl")
joblib.dump(scaler, "scaler.pkl")