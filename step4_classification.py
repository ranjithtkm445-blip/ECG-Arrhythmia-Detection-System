# step4_classification.py
# Purpose: Train ML models on combined 4-record MIT-BIH dataset
# Records used: 100, 101, 105, 200 (7008 beats total)
# Models: SVM + Random Forest
# Input:  D:\ecg_arr\feature_table.csv
# Output: best_model.pkl + scaler.pkl + step4_output.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score)
from sklearn.utils import resample
import pickle
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------------
# SECTION 1: Load Feature Table
# -------------------------------------------------------

csv_path    = r'D:\ecg_arr\feature_table.csv'
features_df = pd.read_csv(csv_path)

print("=== Step 4: ECG Beat Classification (Multi-Record) ===\n")
print(f"Dataset loaded         : {csv_path}")
print(f"Total beats            : {len(features_df)}")
print(f"\nBeats per record       :")
print(features_df.groupby('record')['label'].count())
print(f"\nBeat type distribution :")
print(features_df['label'].value_counts())

# -------------------------------------------------------
# SECTION 2: Define Features and Labels
# -------------------------------------------------------

feature_cols = [
    'rr_interval_ms',
    'heart_rate_bpm',
    'qrs_duration_ms',
    'st_deviation_mv',
    'rr_variability_ms',
    'pr_interval_ms'
]

print(f"\nFeatures used          : {feature_cols}")
print(f"Input shape            : {features_df[feature_cols].shape}")

# -------------------------------------------------------
# SECTION 3: Handle Class Imbalance
# -------------------------------------------------------
# N=6194, V=769, A=45 — still imbalanced
# Strategy: oversample minority, undersample majority
# Target: balance to 700 per class (enough data, fair training)

df_N = features_df[features_df['label'] == 'N']
df_V = features_df[features_df['label'] == 'V']
df_A = features_df[features_df['label'] == 'A']

print(f"\nBefore balancing       :")
print(f"  N (Normal)           : {len(df_N)}")
print(f"  V (Ventricular)      : {len(df_V)}")
print(f"  A (Atrial)           : {len(df_A)}")

# Balance target — use min of 700 or available V beats
target_count = min(700, len(df_V))

df_N_bal = resample(df_N,
                    n_samples=target_count,
                    random_state=42,
                    replace=False)   # Undersample N (enough data)

df_V_bal = resample(df_V,
                    n_samples=target_count,
                    random_state=42,
                    replace=len(df_V) < target_count)  # Oversample if needed

df_A_bal = resample(df_A,
                    n_samples=target_count,
                    random_state=42,
                    replace=True)    # Oversample A (only 45 samples)

# Combine and shuffle
df_balanced = pd.concat([df_N_bal, df_V_bal, df_A_bal])
df_balanced = df_balanced.sample(frac=1, random_state=42)

X_bal = df_balanced[feature_cols].values
y_bal = df_balanced['label'].values

print(f"\nAfter balancing        :")
print(f"  N (Normal)           : {len(df_N_bal)}")
print(f"  V (Ventricular)      : {len(df_V_bal)}")
print(f"  A (Atrial)           : {len(df_A_bal)}")
print(f"  Total balanced       : {len(df_balanced)}")

# -------------------------------------------------------
# SECTION 4: Train / Test Split
# -------------------------------------------------------
# 80% training, 20% testing
# stratify ensures equal class representation in both sets

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal,
    test_size=0.2,
    random_state=42,
    stratify=y_bal
)

print(f"\nTrain set size         : {len(X_train)} beats")
print(f"Test set size          : {len(X_test)} beats")

# -------------------------------------------------------
# SECTION 5: Feature Scaling
# -------------------------------------------------------

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -------------------------------------------------------
# SECTION 6: Train Model 1 — SVM
# -------------------------------------------------------

print("\n--- Training Model 1: SVM ---")

svm_model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    class_weight='balanced',
    random_state=42,
    probability=True        # Needed for confidence scores
)

svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy    = accuracy_score(y_test, svm_predictions)

print(f"SVM Accuracy           : {svm_accuracy * 100:.2f}%")
print(f"\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions,
                             target_names=['A', 'N', 'V']))

# -------------------------------------------------------
# SECTION 7: Train Model 2 — Random Forest
# -------------------------------------------------------

print("--- Training Model 2: Random Forest ---")

rf_model = RandomForestClassifier(
    n_estimators=200,       # More trees = more robust
    max_depth=15,           # Deeper trees for complex patterns
    min_samples_split=5,    # Prevents overfitting
    min_samples_leaf=2,     # Prevents overfitting
    class_weight='balanced',
    random_state=42,
    n_jobs=-1               # Use all CPU cores
)

rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy    = accuracy_score(y_test, rf_predictions)

print(f"Random Forest Accuracy : {rf_accuracy * 100:.2f}%")
print(f"\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_predictions,
                             target_names=['A', 'N', 'V']))

# -------------------------------------------------------
# SECTION 8: Cross Validation
# -------------------------------------------------------

print("--- Cross Validation (5-fold) ---")

svm_cv = cross_val_score(svm_model, X_bal,
                          y_bal, cv=5,
                          scoring='accuracy',
                          n_jobs=-1)

rf_cv  = cross_val_score(rf_model, X_bal,
                          y_bal, cv=5,
                          scoring='accuracy',
                          n_jobs=-1)

print(f"SVM CV Accuracy        : {svm_cv.mean()*100:.2f}% "
      f"(+/- {svm_cv.std()*100:.2f}%)")
print(f"RF  CV Accuracy        : {rf_cv.mean()*100:.2f}% "
      f"(+/- {rf_cv.std()*100:.2f}%)")

# -------------------------------------------------------
# SECTION 9: Pick Best Model and Save
# -------------------------------------------------------

if rf_accuracy >= svm_accuracy:
    best_model      = rf_model
    best_model_name = 'Random Forest'
    best_accuracy   = rf_accuracy
    best_predictions = rf_predictions
else:
    best_model      = svm_model
    best_model_name = 'SVM'
    best_accuracy   = svm_accuracy
    best_predictions = svm_predictions

print(f"\nBest Model             : {best_model_name}")
print(f"Best Accuracy          : {best_accuracy * 100:.2f}%")

model_path  = r'D:\ecg_arr\best_model.pkl'
scaler_path = r'D:\ecg_arr\scaler.pkl'

with open(model_path,  'wb') as f:
    pickle.dump(best_model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Model saved to         : {model_path}")
print(f"Scaler saved to        : {scaler_path}")

# -------------------------------------------------------
# SECTION 10: Plot Confusion Matrices + Feature Importance
# -------------------------------------------------------

fig = plt.figure(figsize=(18, 10))
fig.suptitle('ECG Classification Results — Multi-Record Dataset (4 Records)',
             fontsize=14, fontweight='bold')

class_names = sorted(set(y_bal))

# --- Confusion Matrix: SVM ---
ax1 = fig.add_subplot(2, 3, 1)
cm_svm = confusion_matrix(y_test, svm_predictions, labels=class_names)
im1    = ax1.imshow(cm_svm, interpolation='nearest', cmap='Blues')
ax1.set_title(f'SVM\nAccuracy: {svm_accuracy*100:.2f}%', fontsize=11)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_xticks(range(len(class_names)))
ax1.set_yticks(range(len(class_names)))
ax1.set_xticklabels(class_names)
ax1.set_yticklabels(class_names)
plt.colorbar(im1, ax=ax1)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax1.text