# step4b_cnn.py
# Purpose: Train 1D CNN on raw ECG segments for arrhythmia classification
# Input:  Raw ECG signal from 4 records (100, 101, 105, 200)
# Output: cnn_model.pkl + training history plot + comparison plot
# CNN learns directly from waveform shape — no hand-crafted features needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow as tf
from keras.models     import Sequential
from keras.layers     import (Conv1D, MaxPooling1D, Flatten,
                               Dense, Dropout, BatchNormalization)
from keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from keras.utils      import to_categorical
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=== Step 4b: CNN Deep Learning Classifier ===\n")
print(f"TensorFlow version     : {tf.__version__}")

# -------------------------------------------------------
# SECTION 1: Configuration
# -------------------------------------------------------

DATA_PATH = r'D:\ecg_arr\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'
RECORDS   = ['100', '101', '105', '200']

# Segment window around each R-peak
# 250ms before + 400ms after = 650ms total per beat
# At 360 Hz: 90 + 144 = 234 samples per segment
WINDOW_BEFORE = int(0.250 * 360)   # 90 samples before R-peak
WINDOW_AFTER  = int(0.400 * 360)   # 144 samples after R-peak
SEGMENT_LEN   = WINDOW_BEFORE + WINDOW_AFTER  # 234 samples total

# Label mapping — same as Step 3
LABEL_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'V': 'V', 'E': 'V',
    'A': 'A', 'a': 'A', 'J': 'A', 'S': 'A', 'F': 'A',
}

SKIP_LABELS = ['+', '~', '|', 'Q']

# -------------------------------------------------------
# SECTION 2: Helper Functions
# -------------------------------------------------------

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Bandpass filter — same as all previous steps"""
    nyquist = 0.5 * fs
    low     = lowcut / nyquist
    high    = highcut / nyquist
    b, a    = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs=360, threshold=0.15):
    """Adaptive R-peak detection — same as Step 3"""
    diff_signal   = np.diff(signal)
    squared       = diff_signal ** 2
    window_size   = int(0.150 * fs)
    kernel        = np.ones(window_size) / window_size
    integrated    = np.convolve(squared, kernel, mode='same')
    min_distance  = int(0.200 * fs)
    height_thresh = np.percentile(integrated, 100 * (1 - threshold))
    r_peaks, _    = find_peaks(integrated,
                                distance=min_distance,
                                height=height_thresh)
    return r_peaks

# -------------------------------------------------------
# SECTION 3: Extract Raw ECG Segments
# -------------------------------------------------------
# Instead of extracting features (like Step 3),
# we cut out a fixed-length raw ECG window around each R-peak
# This is what the CNN will learn from directly

def extract_segments(record_name, data_path):
    """
    Extracts fixed-length raw ECG segments around each R-peak.

    Parameters:
        record_name : MIT-BIH record number
        data_path   : path to dataset folder

    Returns:
        segments : numpy array of shape (n_beats, SEGMENT_LEN)
        labels   : list of beat type labels (N, V, A)
    """

    # Load and filter record
    record       = wfdb.rdrecord(f'{data_path}\\{record_name}')
    ecg_signal   = record.p_signal[:, 0]
    fs           = record.fs
    ecg_filtered = bandpass_filter(ecg_signal, fs=fs)

    # Detect R-peaks with adaptive threshold
    threshold_map = {
        '100': 0.20,
        '101': 0.20,
        '105': 0.08,
        '200': 0.12,
    }
    threshold = threshold_map.get(record_name, 0.15)
    r_peaks   = detect_r_peaks(ecg_filtered, fs=fs, threshold=threshold)

    # Load annotations
    annotation  = wfdb.rdann(f'{data_path}\\{record_name}', 'atr')
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    segments = []
    labels   = []

    for r in r_peaks:

        # Skip if segment goes outside signal boundaries
        if r - WINDOW_BEFORE < 0 or r + WINDOW_AFTER >= len(ecg_filtered):
            continue

        # Match annotation
        distances   = np.abs(ann_samples - r)
        closest_idx = int(np.argmin(distances))

        if distances[closest_idx] >= int(0.050 * fs):
            continue

        raw_label = ann_symbols[closest_idx]

        # Skip non-beat annotations
        if raw_label in SKIP_LABELS:
            continue

        # Map to N, V, A
        mapped_label = LABEL_MAP.get(raw_label, None)
        if mapped_label is None:
            continue

        # Extract raw segment around R-peak
        segment = ecg_filtered[r - WINDOW_BEFORE : r + WINDOW_AFTER]

        # Normalize segment to [-1, 1] range
        # This makes CNN training more stable
        seg_min = segment.min()
        seg_max = segment.max()
        if seg_max - seg_min > 0:
            segment = 2 * (segment - seg_min) / (seg_max - seg_min) - 1
        else:
            continue

        segments.append(segment)
        labels.append(mapped_label)

    print(f"  Record {record_name} → {len(segments)} segments extracted")
    return np.array(segments), np.array(labels)

# -------------------------------------------------------
# SECTION 4: Load All Records
# -------------------------------------------------------

print("Extracting raw ECG segments from all records...\n")

all_segments = []
all_labels   = []

for record_name in RECORDS:
    print(f"Processing Record {record_name}...")
    segs, labs = extract_segments(record_name, DATA_PATH)
    all_segments.append(segs)
    all_labels.append(labs)
    print(f"  → {len(segs)} segments | "
          f"Types: {dict(zip(*np.unique(labs, return_counts=True)))}\n")

# Combine all records
X_raw = np.concatenate(all_segments, axis=0)
y_raw = np.concatenate(all_labels,   axis=0)

print(f"Total segments         : {X_raw.shape[0]}")
print(f"Segment length         : {X_raw.shape[1]} samples "
      f"({SEGMENT_LEN/360*1000:.0f}ms)")
print(f"Label distribution     :")
unique, counts = np.unique(y_raw, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u} : {c} ({c/len(y_raw)*100:.1f}%)")

# -------------------------------------------------------
# SECTION 5: Balance Classes
# -------------------------------------------------------

from sklearn.utils import resample

# Separate classes
idx_N = np.where(y_raw == 'N')[0]
idx_V = np.where(y_raw == 'V')[0]
idx_A = np.where(y_raw == 'A')[0]

# Balance to 700 per class
target = min(700, len(idx_V))

idx_N_bal = resample(idx_N, n_samples=target,
                     random_state=42, replace=False)
idx_V_bal = resample(idx_V, n_samples=target,
                     random_state=42,
                     replace=len(idx_V) < target)
idx_A_bal = resample(idx_A, n_samples=target,
                     random_state=42, replace=True)

idx_bal   = np.concatenate([idx_N_bal, idx_V_bal, idx_A_bal])
np.random.shuffle(idx_bal)

X_bal = X_raw[idx_bal]
y_bal = y_raw[idx_bal]

print(f"\nAfter balancing        : {len(X_bal)} segments "
      f"({target} per class)")

# -------------------------------------------------------
# SECTION 6: Encode Labels + Reshape for CNN
# -------------------------------------------------------

# Encode string labels to integers
# N=0, V=1, A=2 (alphabetical by LabelEncoder)
encoder  = LabelEncoder()
y_encoded = encoder.fit_transform(y_bal)
y_cat     = to_categorical(y_encoded)   # One-hot encoding for CNN

print(f"Label encoding         : {dict(zip(encoder.classes_, range(len(encoder.classes_))))}")

# Reshape X for CNN input
# CNN expects shape: (samples, timesteps, channels)
# Our shape: (2100, 234, 1)
X_cnn = X_bal.reshape(X_bal.shape[0], X_bal.shape[1], 1)
print(f"CNN input shape        : {X_cnn.shape}")

# -------------------------------------------------------
# SECTION 7: Train / Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_cnn, y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# Keep original labels for evaluation
_, _, y_train_raw, y_test_raw = train_test_split(
    X_cnn, y_bal,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print(f"\nTrain set              : {len(X_train)} segments")
print(f"Test set               : {len(X_test)} segments")

# -------------------------------------------------------
# SECTION 8: Build CNN Architecture
# -------------------------------------------------------
# Architecture:
#   Input (234 samples, 1 channel)
#     → Conv1D (learns local waveform patterns like QRS shape)
#     → BatchNorm + MaxPool (normalize + downsample)
#     → Conv1D (learns higher level patterns)
#     → BatchNorm + MaxPool
#     → Conv1D (learns global patterns)
#     → Flatten
#     → Dense (fully connected)
#     → Dropout (prevents overfitting)
#     → Dense output (3 classes: N, V, A)

model = Sequential([

    # Block 1: Detect low-level waveform features (QRS spikes, P waves)
    Conv1D(filters=32,
           kernel_size=5,
           activation='relu',
           padding='same',
           input_shape=(SEGMENT_LEN, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Block 2: Detect mid-level patterns (beat shape, morphology)
    Conv1D(filters=64,
           kernel_size=5,
           activation='relu',
           padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    # Block 3: Detect high-level patterns (overall beat character)
    Conv1D(filters=128,
           kernel_size=3,
           activation='relu',
           padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Flatten + Fully Connected
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(64, activation='relu'),
    Dropout(0.3),

    # Output layer: 3 classes (N, V, A)
    Dense(3, activation='softmax')
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- CNN Architecture ---")
model.summary()

# -------------------------------------------------------
# SECTION 9: Training Callbacks
# -------------------------------------------------------

# EarlyStopping: stop if validation accuracy stops improving
# Prevents overfitting and saves training time
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=10,           # Stop after 10 epochs of no improvement
    restore_best_weights=True,
    verbose=1
)

# ReduceLROnPlateau: reduce learning rate when stuck
# Helps model find better solution
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,            # Halve the learning rate
    patience=5,            # After 5 epochs of no improvement
    min_lr=0.00001,
    verbose=1
)

# -------------------------------------------------------
# SECTION 10: Train CNN
# -------------------------------------------------------

print("\n--- Training CNN ---")
print(f"Epochs    : up to 50 (early stopping enabled)")
print(f"Batch size: 32 segments per update")
print(f"Callbacks : EarlyStopping + ReduceLROnPlateau\n")

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------------------------------------------
# SECTION 11: Evaluate CNN
# -------------------------------------------------------

print("\n--- CNN Evaluation ---")

# Predict on test set
y_pred_prob = model.predict(X_test, verbose=0)
y_pred_idx  = np.argmax(y_pred_prob, axis=1)
y_test_idx  = np.argmax(y_test,      axis=1)

y_pred_labels = encoder.inverse_transform(y_pred_idx)
y_test_labels = encoder.inverse_transform(y_test_idx)

cnn_accuracy = (y_pred_labels == y_test_labels).mean()

print(f"CNN Test Accuracy      : {cnn_accuracy * 100:.2f}%")
print(f"\nCNN Classification Report:")
print(classification_report(y_test_labels, y_pred_labels,
                             target_names=encoder.classes_))

# -------------------------------------------------------
# SECTION 12: Save CNN Model
# -------------------------------------------------------

cnn_model_path   = r'D:\ecg_arr\cnn_model.keras'
encoder_path     = r'D:\ecg_arr\cnn_encoder.pkl'

model.save(cnn_model_path)

with open(encoder_path, 'wb') as f:
    pickle.dump(encoder, f)

print(f"CNN model saved to     : {cnn_model_path}")
print(f"Encoder saved to       : {encoder_path}")

# -------------------------------------------------------
# SECTION 13: Plot Training History + Confusion Matrix
# -------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('CNN Training Results — ECG Arrhythmia Classification',
             fontsize=14, fontweight='bold')

# --- Plot 1: Training Accuracy per Epoch ---
ax1 = axes[0, 0]
ax1.plot(history.history['accuracy'],
         color='#3498db', linewidth=2, label='Train Accuracy')
ax1.plot(history.history['val_accuracy'],
         color='#e74c3c', linewidth=2,
         linestyle='--', label='Val Accuracy')
ax1.axhline(y=cnn_accuracy,
            color='#2ecc71', linewidth=1.5,
            linestyle=':', label=f'Test: {cnn_accuracy*100:.1f}%')
ax1.set_title('Accuracy per Epoch')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.05])

# --- Plot 2: Training Loss per Epoch ---
ax2 = axes[0, 1]
ax2.plot(history.history['loss'],
         color='#3498db', linewidth=2, label='Train Loss')
ax2.plot(history.history['val_loss'],
         color='#e74c3c', linewidth=2,
         linestyle='--', label='Val Loss')
ax2.set_title('Loss per Epoch')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- Plot 3: CNN Confusion Matrix ---
ax3 = axes[1, 0]
cm  = confusion_matrix(y_test_labels, y_pred_labels,
                        labels=encoder.classes_)
im  = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title(f'CNN Confusion Matrix\nAccuracy: {cnn_accuracy*100:.2f}%')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_xticks(range(len(encoder.classes_)))
ax3.set_yticks(range(len(encoder.classes_)))
ax3.set_xticklabels(encoder.classes_)
ax3.set_yticklabels(encoder.classes_)
plt.colorbar(im, ax=ax3)
for i in range(len(encoder.classes_)):
    for j in range(len(encoder.classes_)):
        ax3.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 fontsize=12, fontweight='bold',
                 color='white' if cm[i,j] > cm.max()/2 else 'black')

# --- Plot 4: CNN vs Random Forest Comparison ---
ax4 = axes[1, 1]

# Load RF accuracy from step4 results
rf_cv_accuracy  = 97.52   # From Step 4 output
rf_test_accuracy = 96.19  # From Step 4 output

models    = ['SVM\n(Step 4)', 'Random Forest\n(Step 4)', 'CNN\n(Step 4b)']
test_accs = [92.14, rf_test_accuracy, cnn_accuracy * 100]
cv_accs   = [75.95, rf_cv_accuracy,   cnn_accuracy * 100]

x      = np.arange(len(models))
width  = 0.35
bars1  = ax4.bar(x - width/2, test_accs,
                  width, label='Test Accuracy',
                  color=['#3498db', '#2ecc71', '#e74c3c'],
                  edgecolor='white', linewidth=0.5)
bars2  = ax4.bar(x + width/2, cv_accs,
                  width, label='CV Accuracy',
                  color=['#85c1e9', '#82e0aa', '#f1948a'],
                  edgecolor='white', linewidth=0.5)

ax4.set_title('Model Comparison: SVM vs RF vs CNN')
ax4.set_xticks(x)
ax4.set_xticklabels(models, fontsize=9)
ax4.set_ylabel('Accuracy (%)')
ax4.set_ylim([60, 105])
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

for bar in bars1:
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%',
             ha='center', fontsize=8, fontweight='bold')

for bar in bars2:
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{bar.get_height():.1f}%',
             ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
output_path = r'D:\ecg_arr\step4b_output.png'
plt.savefig(output_path, dpi=150)
plt.close()
print(f"Plot saved to          : {output_path}")

# -------------------------------------------------------
# SECTION 14: Final Summary
# -------------------------------------------------------

epochs_run = len(history.history['accuracy'])

print(f"\n{'='*50}")
print(f"  CNN TRAINING COMPLETE")
print(f"{'='*50}")
print(f"  Segment length  : {SEGMENT_LEN} samples ({SEGMENT_LEN/360*1000:.0f}ms)")
print(f"  Total segments  : {len(X_bal)}")
print(f"  Epochs run      : {epochs_run} (early stopping)")
print(f"  CNN Accuracy    : {cnn_accuracy*100:.2f}%")
print(f"  RF  Accuracy    : {rf_test_accuracy}%")
print(f"  SVM Accuracy    : 92.14%")
print(f"  Best Model      : {'CNN' if cnn_accuracy*100 > rf_test_accuracy else 'Random Forest'}")
print(f"{'='*50}")
print(f"\nNow run: python step5_visualization.py")