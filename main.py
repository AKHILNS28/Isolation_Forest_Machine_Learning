import pandas as pd
import numpy as np
import os
import time

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ==============================
# STEP 1: LOAD DATA
# ==============================

DATA_PATH = "cicids2017_cleaned.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ==============================
# STEP 2: DATA CLEANING
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
# STEP 6: FEATURE SELECTION
# ==============================

selector = VarianceThreshold(threshold=0.01)
X = selector.fit_transform(X)


# ==============================
# STEP 7: FEATURE SCALING
# ==============================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ==============================
# STEP 8: PCA (DIMENSION REDUCTION)
# ==============================

print("\nApplying PCA...")

pca = PCA(n_components=0.95)   # keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print("Features after PCA:", X_pca.shape)


# ==============================
# STEP 9: TRAIN TEST SPLIT
# ==============================

X_normal = X_pca[y == 0]

X_train, _ = train_test_split(
    X_normal,
    test_size=0.2,
    random_state=42
)

X_test = X_pca
y_test = y


# ==============================
# STEP 10: TRAIN ISOLATION FOREST
# ==============================

print("\nTraining model...")

model = IsolationForest(
    n_estimators=400,
    max_samples=256,
    contamination=0.12,
    max_features=1.0,
    bootstrap=False,
    n_jobs=-1,
    random_state=42
)

model.fit(X_train)

print("Training completed.")


# ==============================
# STEP 11: ANOMALY SCORING
# ==============================

print("\nDetecting anomalies...")

start = time.time()

scores = -model.decision_function(X_test)

latency = time.time() - start

print("Detection latency:", latency)


# ==============================
# STEP 12: THRESHOLD OPTIMIZATION
# ==============================

threshold = np.percentile(scores, 88)   # optimized threshold

pred = (scores >= threshold).astype(int)


# ==============================
# STEP 13: EVALUATION
# ==============================

cm = confusion_matrix(y_test, pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, pred))


# ==============================
# STEP 14: ROC-AUC
# ==============================

auc = roc_auc_score(y_test, scores)

print("\nAUC Score:", auc)

fpr, tpr, _ = roc_curve(y_test, scores)


# ==============================
# STEP 15: SAVE RESULTS
# ==============================

os.makedirs("results", exist_ok=True)

pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred
}).to_csv("results/predictions.csv", index=False)

joblib.dump(model, "results/isolation_forest_model.pkl")

print("Results saved.")


# ==============================
# STEP 16: VISUALIZATION
# ==============================

plt.figure()
plt.hist(pred)
plt.title("Anomaly Detection Results")
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.savefig("results/anomaly_histogram.png")


plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("results/confusion_matrix.png")


plt.figure()
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.savefig("results/roc_curve.png")

print("\nPipeline completed successfully!")