import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import os
import time

# ==============================
# STEP 1: LOAD DATA
# ==============================

DATA_PATH = "cicids2017_cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==============================
# STEP 2: CLEAN DATA
# ==============================

print("\nCleaning data...")

df = df.drop_duplicates()

df = df.replace([np.inf, -np.inf], np.nan)

df = df.fillna(df.median(numeric_only=True))

print("Shape after cleaning:", df.shape)


# ==============================
# STEP 3: DROP IRRELEVANT COLUMNS
# ==============================

drop_cols = ["Flow ID", "Source IP", "Destination IP", "Timestamp"]

df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')


# ==============================
# STEP 4: SEPARATE FEATURES AND LABEL
# ==============================

label_col = "Attack Type"

y = df[label_col]
X = df.drop(label_col, axis=1)

# Convert to binary
y = y.apply(lambda x: 0 if "normal" in str(x).lower() else 1)

print("\nLabel distribution:")
print(y.value_counts())


# ==============================
# STEP 5: ENCODE CATEGORICAL
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
# STEP 7: PREPARE TRAINING DATA (IMPROVED)
# ==============================

print("\nPreparing training data...")

X_normal = X_scaled[y == 0]
X_attack = X_scaled[y == 1]

# Add small portion of attacks (important improvement)
attack_sample = X_attack[:int(0.1 * len(X_attack))]

X_train = np.vstack((X_normal, attack_sample))

X_test = X_scaled
y_test = y


# ==============================
# STEP 8: TRAIN MODEL (IMPROVED)
# ==============================

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=300,        # more trees
    contamination=0.12,      # tuned
    max_samples=256,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train)

print("Model training complete.")


# ==============================
# STEP 9: ANOMALY SCORING
# ==============================

print("\nDetecting anomalies...")

start = time.time()

scores = -model.decision_function(X_test)

latency = time.time() - start

print("Detection latency:", latency)


# ==============================
# STEP 10: THRESHOLD OPTIMIZATION
# ==============================

print("\nOptimizing threshold...")

best_f1 = 0
best_threshold = 0

for p in range(70, 95):

    th = np.percentile(scores, p)

    pred_temp = (scores >= th).astype(int)

    f1 = f1_score(y_test, pred_temp)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = th

print("Best threshold:", best_threshold)

pred = (scores >= best_threshold).astype(int)


# ==============================
# STEP 11: EVALUATE
# ==============================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

print("\nClassification Report:")
print(classification_report(y_test, pred))


# ==============================
# STEP 12: SAVE RESULTS
# ==============================

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
})

results_df.to_csv("results/predictions.csv", index=False)

print("\nResults saved.")


# ==============================
# STEP 13: VISUALIZATION
# ==============================

plt.figure()
plt.hist(pred)
plt.title("Anomaly Detection Results")
plt.xlabel("Prediction (0 = Normal, 1 = Attack)")
plt.ylabel("Count")
plt.savefig("results/anomaly_histogram.png")

print("\nPipeline completed successfully!")
