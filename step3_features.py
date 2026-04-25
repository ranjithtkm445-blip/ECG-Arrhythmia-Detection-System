# step3_features.py
# Purpose: Extract features from multiple MIT-BIH records
# Records: 100, 101, 105, 200, 209, 222, 208, 213
# Added 208 (375 Atrial) and 213 (390 Atrial) to fix Atrial imbalance
# Expected Atrial beats: ~1251 (vs 486 before)
# Output: feature_table.csv with all beats from all 8 records combined

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# -------------------------------------------------------
# SECTION 1: Configuration
# -------------------------------------------------------

DATA_PATH = r'D:\ecg_arr\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'

# Updated to 8 records
# 208 → 375 Atrial beats
# 213 → 390 Atrial beats
RECORDS = ['100', '101', '105', '200', '209', '222', '208', '213']

# -------------------------------------------------------
# SECTION 2: Helper Functions
# -------------------------------------------------------

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Bandpass filter to clean ECG signal"""
    nyquist = 0.5 * fs
    low     = lowcut / nyquist
    high    = highcut / nyquist
    b, a    = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs=360, threshold=0.15):
    """
    Adaptive R-peak detection using Pan-Tompkins inspired method.

    Parameters:
        signal    : filtered ECG signal
        fs        : sampling frequency
        threshold : detection sensitivity
                    0.20 = moderate (clean signals)
                    0.08 = very relaxed (noisy signals)

    Returns:
        r_peaks : array of R-peak sample indices
    """
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

def extract_features_single_record(record_name, data_path):
    """
    Extracts all clinical features from a single ECG record.

    Parameters:
        record_name : MIT-BIH record number as string
        data_path   : path to folder containing .dat and .hea files

    Returns:
        DataFrame with features for all valid beats in this record
    """

    # --- Load record ---
    record     = wfdb.rdrecord(f'{data_path}\\{record_name}')
    ecg_signal = record.p_signal[:, 0]
    fs         = record.fs

    # --- Filter ---
    ecg_filtered = bandpass_filter(ecg_signal, fs=fs)

    # --- Adaptive threshold per record ---
    threshold_map = {
        '100': 0.20,   # Clean normal signal
        '101': 0.20,   # Clean normal signal
        '105': 0.08,   # Very noisy + heavy arrhythmia
        '200': 0.12,   # Heavy PVC arrhythmia
        '209': 0.15,   # Atrial rich record
        '222': 0.15,   # Atrial rich record
        '208': 0.12,   # Heavy arrhythmia — Atrial + Ventricular mix
        '213': 0.15,   # Atrial rich record
    }
    threshold = threshold_map.get(record_name, 0.15)

    # --- Detect R-peaks ---
    r_peaks = detect_r_peaks(ecg_filtered, fs=fs, threshold=threshold)

    # --- Load annotations ---
    annotation  = wfdb.rdann(f'{data_path}\\{record_name}', 'atr')
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol

    print(f"  Record {record_name} → {len(r_peaks)} R-peaks detected | "
          f"Annotated: {len(ann_samples)} beats | "
          f"Types: {set(ann_symbols)}")

    # --- Feature extraction windows ---
    qrs_before = int(0.050 * fs)
    qrs_after  = int(0.050 * fs)
    p_start    = int(0.200 * fs)
    st_start   = int(0.080 * fs)
    st_end     = int(0.120 * fs)

    # --- Label mapping ---
    # N = Normal family (includes RBBB, LBBB, escape beats)
    # V = Ventricular family
    # A = Atrial / Supraventricular family
    label_map = {
        'N': 'N', 'L': 'N', 'R': 'N',   # Normal + bundle branch blocks
        'e': 'N', 'j': 'N',              # Escape beats
        'V': 'V', 'E': 'V',              # Ventricular beats
        'A': 'A', 'a': 'A', 'J': 'A',   # Atrial beats
        'S': 'A', 'F': 'A',              # Supraventricular beats
    }

    # Skip these — not actual heartbeats
    skip_labels = ['+', '~', '|', 'Q', 'U', 'f', 'x']

    features_list = []

    for i, r in enumerate(r_peaks):

        if i == 0:
            continue

        if (r - p_start) < 0 or (r + st_end) >= len(ecg_filtered):
            continue

        try:
            # --- RR Interval (ms) ---
            rr_samples   = int(r_peaks[i]) - int(r_peaks[i - 1])
            rr_interval  = (rr_samples / fs) * 1000

            if rr_interval < 200 or rr_interval > 3000:
                continue

            # --- Heart Rate (BPM) ---
            heart_rate   = round(60000 / rr_interval, 2)

            # --- QRS Duration (ms) ---
            qrs_duration = round(((qrs_before + qrs_after) / fs) * 1000, 2)

            # --- PR Interval (ms) ---
            pr_interval  = round((p_start / fs) * 1000, 2)

            # --- ST Deviation (mV) ---
            st_segment   = ecg_filtered[r + st_start : r + st_end]
            st_deviation = round(float(np.mean(st_segment)), 4)

            # --- RR Variability (ms) ---
            if i >= 2:
                prev_rr        = (int(r_peaks[i-1]) - int(r_peaks[i-2])) / fs * 1000
                rr_variability = round(abs(rr_interval - prev_rr), 2)
            else:
                rr_variability = 0.0

            # --- Match annotation label ---
            distances   = np.abs(ann_samples - r)
            closest_idx = int(np.argmin(distances))

            if distances[closest_idx] >= int(0.050 * fs):
                continue

            raw_label = ann_symbols[closest_idx]

            # Skip non-beat annotations
            if raw_label in skip_labels:
                continue

            # Map to N, V, A
            label = label_map.get(raw_label, None)
            if label is None:
                continue

            features_list.append({
                'record'           : record_name,
                'beat_index'       : i,
                'r_peak_sample'    : int(r),
                'rr_interval_ms'   : round(rr_interval, 2),
                'heart_rate_bpm'   : heart_rate,
                'qrs_duration_ms'  : qrs_duration,
                'pr_interval_ms'   : pr_interval,
                'st_deviation_mv'  : st_deviation,
                'rr_variability_ms': rr_variability,
                'label'            : label
            })

        except Exception:
            continue

    return pd.DataFrame(features_list)

# -------------------------------------------------------
# SECTION 3: Process All 8 Records
# -------------------------------------------------------

print("=== Step 3: Multi-Record Feature Extraction ===\n")
print(f"Processing {len(RECORDS)} records: {RECORDS}\n")

all_features = []

for record_name in RECORDS:
    print(f"Processing Record {record_name}...")
    df_record = extract_features_single_record(record_name, DATA_PATH)
    all_features.append(df_record)
    print(f"  → {len(df_record)} beats extracted\n")

# Combine all records
features_df = pd.concat(all_features, ignore_index=True)

# -------------------------------------------------------
# SECTION 4: Summary
# -------------------------------------------------------

print("=" * 55)
print(f"COMBINED DATASET SUMMARY")
print("=" * 55)
print(f"Total beats            : {len(features_df)}")
print(f"\nBeats per record       :")
print(features_df.groupby('record')['label'].count())
print(f"\nBeat type distribution :")
print(features_df['label'].value_counts())
print(f"\nBreakdown per record   :")
print(features_df.groupby(['record', 'label'])['beat_index'].count())

# -------------------------------------------------------
# SECTION 5: Save Combined Feature Table
# -------------------------------------------------------

csv_path = r'D:\ecg_arr\feature_table.csv'
features_df.to_csv(csv_path, index=False)
print(f"\nFeature table saved to : {csv_path}")

# -------------------------------------------------------
# SECTION 6: Plot Summary
# -------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Multi-Record Feature Extraction Summary '
             '(8 Records)', fontsize=14)

# Plot 1: Beats per record
record_counts = features_df.groupby('record')['label'].count()
bar_colors    = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c',
                 '#9b59b6', '#1abc9c', '#f39c12', '#2980b9']
bars = axes[0].bar(record_counts.index,
                   record_counts.values,
                   color=bar_colors[:len(record_counts)],
                   edgecolor='white',
                   linewidth=0.5)
axes[0].set_title('Total Beats per Record')
axes[0].set_xlabel('Record')
axes[0].set_ylabel('Beat Count')
axes[0].grid(True, alpha=0.3)
for i, (rec, val) in enumerate(record_counts.items()):
    axes[0].text(i, val + 10, str(val),
                 ha='center', fontsize=8,
                 fontweight='bold')

# Plot 2: Beat type distribution
label_counts = features_df['label'].value_counts()
colors_pie   = ['#2ecc71', '#e74c3c', '#e67e22']
axes[1].pie(label_counts.values,
            labels=label_counts.index,
            autopct='%1.1f%%',
            colors=colors_pie[:len(label_counts)],
            startangle=90)
axes[1].set_title('Beat Type Distribution (All 8 Records)')

plt.tight_layout()
output_path = r'D:\ecg_arr\step3_output.png'
plt.savefig(output_path, dpi=150)
plt.close()

print(f"Plot saved to          : {output_path}")
print(f"\nStep 3 Complete! Now run step4_classification.py")