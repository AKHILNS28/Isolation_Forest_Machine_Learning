import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import os


# ==============================
# STEP 1: LOAD DATA
# ==============================

DATA_PATH = "cicids2017_cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())


# ==============================
# STEP 2: CLEAN DATA
# ==============================

print("\nCleaning data...")

# Remove duplicates
df = df.drop_duplicates()

# Remove missing values
df = df.dropna()

print("Shape after cleaning:", df.shape)


# ==============================
# STEP 3: DROP IRRELEVANT COLUMNS
# ==============================

drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

print("Columns after dropping irrelevant features:", len(df.columns))


# ==============================
# STEP 4: SEPARATE FEATURES AND LABEL
# ==============================

possible_labels = ["Label", "label", "Attack", "attack", "Attack Type", "attack_type"]

label_col = None
for col in df.columns:
    if col in possible_labels:
        label_col = col
        break

if label_col is None:
    raise ValueError("Label column not found. Please check dataset.")

print("\nUsing label column:", label_col)

y = df[label_col]
X = df.drop(label_col, axis=1)

# Convert to binary
y = y.apply(lambda x: 0 if "normal" in str(x).lower() or "benign" in str(x).lower() else 1)

print("\nLabel distribution:")
print(y.value_counts())


# ==============================
# STEP 5: ENCODE CATEGORICAL FEATURES
# ==============================

X = pd.get_dummies(X, drop_first=True)

print("Shape after encoding:", X.shape)


# ==============================
# STEP 6: SCALE FEATURES
# ==============================

print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# STEP 7: TRAIN-TEST SPLIT
# ==============================

X_train = X_scaled[y == 0]
X_test = X_scaled
y_test = y


# ==============================
# STEP 8: TRAIN MODEL
# ==============================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X_train)

print("Model training complete.")


# ==============================
# STEP 9: PREDICT
# ==============================

print("\nPredicting anomalies...")

pred = model.predict(X_test)


pred = np.where(pred == 1, 0, 1)


# ==============================
# STEP 10: EVALUATE
# ==============================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))


# ==============================
# STEP 11: SAVE RESULTS
# ==============================

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

results_df.to_csv("results/predictions.csv", index=False)

print("\nResults saved to results/predictions.csv")


# ==============================
# STEP 12: VISUALIZATION
# ==============================

plt.figure()
plt.hist(pred)
plt.title("Anomaly Detection Results")
plt.xlabel("Prediction (0 = Normal, 1 = Attack)")
plt.ylabel("Count")
plt.savefig("results/anomaly_histogram.png")

print("Histogram saved to results/anomaly_histogram.png")

print("\nPipeline completed successfully!")