# step5_visualization.py
# Purpose: End user ECG clinical dashboard with file picker
# NEW: Big verdict banner at top showing arrhythmia present/absent
# Models: Random Forest + CNN (trained on 8 records)
# Output: step5_dashboard.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import wfdb
import pickle
import os
import sys
import tensorflow as tf
from scipy.signal import butter, filtfilt, find_peaks
from tkinter import Tk, filedialog
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=== ECG Arrhythmia Detection System ===\n")

# -------------------------------------------------------
# SECTION 1: Configuration
# -------------------------------------------------------

MODEL_DIR     = r'D:\ecg_arr'
WINDOW_BEFORE = int(0.250 * 360)
WINDOW_AFTER  = int(0.400 * 360)
SEGMENT_LEN   = WINDOW_BEFORE + WINDOW_AFTER

LABEL_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N',
    'e': 'N', 'j': 'N', 'B': 'N',
    'V': 'V', 'E': 'V',
    'A': 'A', 'a': 'A', 'J': 'A',
    'S': 'A', 'F': 'A',
}

SKIP_LABELS = ['+', '~', '|', 'Q', 'U', 'f', 'x']

BEAT_COLORS = {
    'N': '#2ecc71',
    'A': '#e67e22',
    'V': '#e74c3c',
}

# -------------------------------------------------------
# SECTION 2: Load Models
# -------------------------------------------------------

print("Loading trained models...")

with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'cnn_model.keras'),
    compile=False)

with open(os.path.join(MODEL_DIR, 'cnn_encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)

print("Models loaded          : Random Forest + CNN")

# -------------------------------------------------------
# SECTION 3: File Picker Dialog
# -------------------------------------------------------

def pick_ecg_file():
    """Opens file picker for user to select .dat ECG file"""
    print("\nOpening file picker — please select your ECG .dat file...\n")

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    file_path = filedialog.askopenfilename(
        title="Select ECG .dat file",
        filetypes=[
            ("ECG Data Files", "*.dat"),
            ("All Files", "*.*")
        ]
    )
    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        sys.exit()

    data_folder = os.path.dirname(file_path)
    record_name = os.path.splitext(os.path.basename(file_path))[0]
    record_path = os.path.join(data_folder, record_name)

    hea_path = record_path + '.hea'
    if not os.path.exists(hea_path):
        print(f"ERROR: Could not find {hea_path}")
        sys.exit()

    print(f"File selected          : {file_path}")
    print(f"Record name            : {record_name}")
    print(f"Data folder            : {data_folder}")

    return record_path, record_name, data_folder

# -------------------------------------------------------
# SECTION 4: Helper Functions
# -------------------------------------------------------

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Bandpass filter — removes noise from ECG signal"""
    nyquist = 0.5 * fs
    low     = lowcut / nyquist
    high    = highcut / nyquist
    b, a    = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_r_peaks(signal, fs=360, threshold=0.15):
    """Adaptive R-peak detection"""
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

def assess_risk(predictions):
    """Assess cardiac risk based on beat classifications"""
    counts  = pd.Series(predictions).value_counts()
    v_count = counts.get('V', 0)
    a_count = counts.get('A', 0)
    total   = len(predictions)
    v_pct   = v_count / total * 100
    a_pct   = a_count / total * 100

    if v_pct > 10 or v_count > 100:
        return 'HIGH RISK',  '#e74c3c'
    elif v_pct > 2 or a_pct > 5 or a_count > 30:
        return 'MODERATE',   '#e67e22'
    else:
        return 'LOW RISK',   '#2ecc71'

def get_verdict(rf_predictions, cnn_predictions):
    """
    Determines arrhythmia verdict using average of RF and CNN predictions.

    Returns:
        verdict       : verdict text
        verdict_color : color for text
        verdict_bg    : background color
        verdict_icon  : emoji indicator
        verdict_detail: detailed breakdown text
    """
    total           = len(rf_predictions)
    rf_counts_dict  = pd.Series(rf_predictions).value_counts().to_dict()
    cnn_counts_dict = pd.Series(cnn_predictions).value_counts().to_dict()

    # Average both models for final verdict
    rf_v_pct  = rf_counts_dict.get('V', 0)  / total * 100
    rf_a_pct  = rf_counts_dict.get('A', 0)  / total * 100
    cnn_v_pct = cnn_counts_dict.get('V', 0) / total * 100
    cnn_a_pct = cnn_counts_dict.get('A', 0) / total * 100

    avg_v_pct = (rf_v_pct  + cnn_v_pct)  / 2
    avg_a_pct = (rf_a_pct  + cnn_a_pct)  / 2

    has_v = avg_v_pct > 5
    has_a = avg_a_pct > 5

    if has_v and has_a:
        return (
            'MULTIPLE ARRHYTHMIAS DETECTED',
            '#e74c3c', '#2d0a0a', '🔴',
            f'Ventricular: {avg_v_pct:.1f}%  |  Atrial: {avg_a_pct:.1f}%'
        )
    elif has_v:
        return (
            'VENTRICULAR ARRHYTHMIA DETECTED',
            '#e74c3c', '#2d0a0a', '🔴',
            f'Ventricular beats: {avg_v_pct:.1f}% of total beats'
        )
    elif has_a:
        return (
            'ATRIAL ARRHYTHMIA DETECTED',
            '#e67e22', '#2d1a00', '⚠️',
            f'Atrial beats: {avg_a_pct:.1f}% of total beats'
        )
    else:
        return (
            'NO ARRHYTHMIA DETECTED',
            '#2ecc71', '#0a2d1a', '✅',
            'Rhythm appears normal'
        )

# -------------------------------------------------------
# SECTION 5: Pick ECG File
# -------------------------------------------------------

record_path, record_name, data_folder = pick_ecg_file()

# -------------------------------------------------------
# SECTION 6: Load and Process Selected ECG
# -------------------------------------------------------

print(f"\nProcessing Record {record_name}...")

record     = wfdb.rdrecord(record_path)
ecg_signal = record.p_signal[:, 0]
fs         = record.fs

ecg_filtered = bandpass_filter(ecg_signal, fs=fs)
r_peaks      = detect_r_peaks(ecg_filtered, fs=fs, threshold=0.15)

print(f"Sampling rate          : {fs} Hz")
print(f"Signal duration        : {len(ecg_signal)/fs/60:.1f} minutes")
print(f"R-peaks detected       : {len(r_peaks)}")

atr_path       = record_path + '.atr'
has_annotation = os.path.exists(atr_path)

if has_annotation:
    annotation  = wfdb.rdann(record_path, 'atr')
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol
    print(f"Annotations found      : {len(ann_samples)} beats")
    print(f"Beat types (raw)       : {set(ann_symbols)}")

    mapped_all    = [LABEL_MAP.get(s, None) for s in ann_symbols]
    mapped_all    = [m for m in mapped_all if m is not None]
    mapped_counts = pd.Series(mapped_all).value_counts()
    print(f"Beat types (mapped)    :")
    for label, count in mapped_counts.items():
        print(f"  {label} : {count} ({count/len(mapped_all)*100:.1f}%)")
else:
    ann_samples = np.array([])
    ann_symbols = np.array([])
    print("Annotations            : Not found")

# -------------------------------------------------------
# SECTION 7: Extract Features + Segments
# -------------------------------------------------------

qrs_before   = int(0.050 * fs)
qrs_after    = int(0.050 * fs)
p_start      = int(0.200 * fs)
st_start     = int(0.080 * fs)
st_end       = int(0.120 * fs)

feature_cols = [
    'rr_interval_ms', 'heart_rate_bpm', 'qrs_duration_ms',
    'st_deviation_mv', 'rr_variability_ms', 'pr_interval_ms'
]

features_list = []
segments_list = []
true_labels   = []
valid_r_peaks = []

print("\nExtracting features and segments...")

for i, r in enumerate(r_peaks):

    if i == 0:
        continue

    if (r - max(p_start, WINDOW_BEFORE)) < 0:
        continue

    if (r + max(st_end, WINDOW_AFTER)) >= len(ecg_filtered):
        continue

    try:
        # RR Interval
        rr_samples     = int(r_peaks[i]) - int(r_peaks[i-1])
        rr_interval    = (rr_samples / fs) * 1000

        if rr_interval < 200 or rr_interval > 3000:
            continue

        heart_rate     = round(60000 / rr_interval, 2)
        qrs_duration   = round(((qrs_before + qrs_after) / fs) * 1000, 2)
        pr_interval    = round((p_start / fs) * 1000, 2)
        st_segment     = ecg_filtered[r + st_start : r + st_end]
        st_deviation   = round(float(np.mean(st_segment)), 4)

        if i >= 2:
            prev_rr        = (int(r_peaks[i-1]) - int(r_peaks[i-2])) / fs * 1000
            rr_variability = round(abs(rr_interval - prev_rr), 2)
        else:
            rr_variability = 0.0

        # CNN Segment
        segment  = ecg_filtered[r - WINDOW_BEFORE : r + WINDOW_AFTER]
        seg_min  = segment.min()
        seg_max  = segment.max()

        if seg_max - seg_min <= 0:
            continue

        segment_norm = 2 * (segment - seg_min) / (seg_max - seg_min) - 1

        # Ground Truth Label
        if has_annotation and len(ann_samples) > 0:
            distances   = np.abs(ann_samples - r)
            closest_idx = int(np.argmin(distances))

            if distances[closest_idx] < int(0.050 * fs):
                raw_label = ann_symbols[closest_idx]
                if raw_label in SKIP_LABELS:
                    continue
                mapped = LABEL_MAP.get(raw_label, None)
                if mapped is None:
                    continue
                true_labels.append(mapped)
            else:
                true_labels.append('Unknown')
        else:
            true_labels.append('Unknown')

        features_list.append({
            'r_peak_sample'    : int(r),
            'rr_interval_ms'   : round(rr_interval, 2),
            'heart_rate_bpm'   : heart_rate,
            'qrs_duration_ms'  : qrs_duration,
            'pr_interval_ms'   : pr_interval,
            'st_deviation_mv'  : st_deviation,
            'rr_variability_ms': rr_variability,
        })

        segments_list.append(segment_norm)
        valid_r_peaks.append(r)

    except Exception:
        continue

features_df   = pd.DataFrame(features_list)
segments_arr  = np.array(segments_list)
valid_r_peaks = np.array(valid_r_peaks)
true_arr      = np.array(true_labels)

print(f"Valid beats extracted  : {len(features_df)}")

# -------------------------------------------------------
# SECTION 8: Random Forest Predictions
# -------------------------------------------------------

X_rf           = scaler.transform(features_df[feature_cols].values)
rf_predictions = rf_model.predict(X_rf)
features_df['rf_predicted'] = rf_predictions

rf_counts = pd.Series(rf_predictions).value_counts()
print(f"\nRandom Forest Results  :")
for label, count in rf_counts.items():
    pct = count / len(rf_predictions) * 100
    print(f"  {label} : {count} beats ({pct:.1f}%)")

# -------------------------------------------------------
# SECTION 9: CNN Predictions
# -------------------------------------------------------

X_cnn           = segments_arr.reshape(segments_arr.shape[0],
                                        segments_arr.shape[1], 1)
cnn_pred_prob   = cnn_model.predict(X_cnn, verbose=0)
cnn_pred_idx    = np.argmax(cnn_pred_prob, axis=1)
cnn_predictions = encoder.inverse_transform(cnn_pred_idx)
features_df['cnn_predicted'] = cnn_predictions

cnn_counts = pd.Series(cnn_predictions).value_counts()
print(f"\nCNN Results            :")
for label, count in cnn_counts.items():
    pct = count / len(cnn_predictions) * 100
    print(f"  {label} : {count} beats ({pct:.1f}%)")

# -------------------------------------------------------
# SECTION 10: Accuracy vs Ground Truth
# -------------------------------------------------------

known_mask  = true_arr != 'Unknown'
known_count = known_mask.sum()

if known_count > 0:
    rf_correct  = (rf_predictions[known_mask] == true_arr[known_mask]).sum()
    cnn_correct = (cnn_predictions[known_mask] == true_arr[known_mask]).sum()
    rf_acc      = rf_correct  / known_count
    cnn_acc     = cnn_correct / known_count

    print(f"\nGround Truth Comparison:")
    print(f"  Known beats          : {known_count}")
    print(f"  RF  Accuracy         : {rf_acc*100:.2f}%")
    print(f"  CNN Accuracy         : {cnn_acc*100:.2f}%")

    print(f"\nPer Class Accuracy     :")
    for label in ['N', 'A', 'V']:
        mask = true_arr[known_mask] == label
        if mask.sum() > 0:
            rf_ca  = (rf_predictions[known_mask][mask] == label).mean()
            cnn_ca = (cnn_predictions[known_mask][mask] == label).mean()
            print(f"  {label} — RF: {rf_ca*100:.1f}% | "
                  f"CNN: {cnn_ca*100:.1f}% | "
                  f"Count: {mask.sum()}")
else:
    rf_acc  = None
    cnn_acc = None

# -------------------------------------------------------
# SECTION 11: Verdict + Risk
# -------------------------------------------------------

verdict, verdict_color, verdict_bg, \
verdict_icon, verdict_detail = get_verdict(rf_predictions, cnn_predictions)

rf_risk,  rf_risk_color  = assess_risk(rf_predictions)
cnn_risk, cnn_risk_color = assess_risk(cnn_predictions)

rf_acc_str  = f"{rf_acc*100:.1f}%"  if rf_acc  else "N/A"
cnn_acc_str = f"{cnn_acc*100:.1f}%" if cnn_acc else "N/A"

print(f"\nVerdict                : {verdict_icon} {verdict}")
print(f"RF  Risk               : {rf_risk}")
print(f"CNN Risk               : {cnn_risk}")

# -------------------------------------------------------
# SECTION 12: Build Dashboard
# -------------------------------------------------------

print("\nBuilding clinical dashboard...")

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor('#0f1117')

# -------------------------------------------------------
# VERDICT BANNER — Big and clear at very top
# -------------------------------------------------------

# Colored background box for verdict
verdict_ax = fig.add_axes([0.02, 0.955, 0.96, 0.038])
verdict_ax.set_facecolor(verdict_bg)
verdict_ax.axis('off')
for spine in verdict_ax.spines.values():
    spine.set_edgecolor(verdict_color)
    spine.set_linewidth(2.5)
    spine.set_visible(True)

# Main verdict text — very large and bold
fig.text(0.5, 0.976,
         f'{verdict_icon}   {verdict}   {verdict_icon}',
         ha='center', va='center',
         fontsize=20, fontweight='bold',
         color=verdict_color)

# Verdict detail line
fig.text(0.5, 0.961,
         verdict_detail,
         ha='center', va='center',
         fontsize=11, color='#dddddd')

# -------------------------------------------------------
# INFO BAR — below verdict
# -------------------------------------------------------

info_ax = fig.add_axes([0.02, 0.915, 0.96, 0.033])
info_ax.set_facecolor('#1a1a2e')
info_ax.axis('off')
for spine in info_ax.spines.values():
    spine.set_edgecolor('#444444')
    spine.set_linewidth(0.5)
    spine.set_visible(True)

fig.text(0.5, 0.932,
         f'Record: {record_name}   |   '
         f'Total Beats: {len(features_df)}   |   '
         f'Duration: {len(ecg_signal)/fs/60:.1f} min   |   '
         f'RF Accuracy: {rf_acc_str}   |   '
         f'CNN Accuracy: {cnn_acc_str}   |   '
         f'RF Risk: {rf_risk}   |   '
         f'CNN Risk: {cnn_risk}',
         ha='center', va='center',
         fontsize=10, color='#aaaaaa')

# -------------------------------------------------------
# GRID LAYOUT for panels
# -------------------------------------------------------

gs = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.5,
    wspace=0.35,
    top=0.90,
    bottom=0.05,
    left=0.06,
    right=0.97
)

# -------------------------------------------------------
# PANEL 1: ECG Waveform (first 10 seconds)
# -------------------------------------------------------

ax1 = fig.add_subplot(gs[0, :])
ax1.set_facecolor('#1a1a2e')

n_samples = min(10 * fs, len(ecg_filtered))
time_axis = np.arange(n_samples) / fs

ax1.plot(time_axis, ecg_filtered[:n_samples],
         color='#4a9eff', linewidth=0.8,
         alpha=0.9, zorder=2)

r_in_window = valid_r_peaks[valid_r_peaks < n_samples]
for r in r_in_window:
    idx = np.where(valid_r_peaks == r)[0]
    if len(idx) == 0:
        continue
    pred  = rf_predictions[idx[0]]
    color = BEAT_COLORS.get(pred, 'white')
    ax1.scatter(r/fs, ecg_filtered[r],
                color=color, s=80, zorder=3)
    ax1.axvspan((r - qrs_before)/fs,
                (r + qrs_after)/fs,
                alpha=0.15, color=color, zorder=1)

legend_patches = [
    mpatches.Patch(color='#2ecc71', label='Normal (N)'),
    mpatches.Patch(color='#e67e22', label='Atrial (A)'),
    mpatches.Patch(color='#e74c3c', label='Ventricular (V)'),
]
ax1.legend(handles=legend_patches, loc='upper right',
           facecolor='#1a1a2e', edgecolor='#444444',
           labelcolor='white', fontsize=9)
ax1.set_title(f'ECG Waveform — Color Coded by Beat Type (First 10 seconds)',
              color='white', fontsize=11, pad=8)
ax1.set_xlabel('Time (seconds)', color='#aaaaaa')
ax1.set_ylabel('Amplitude (mV)', color='#aaaaaa')
ax1.tick_params(colors='#aaaaaa')
ax1.spines['bottom'].set_color('#444444')
ax1.spines['left'].set_color('#444444')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(True, alpha=0.15, color='#555555')

# -------------------------------------------------------
# PANEL 2: RF Beat Distribution
# -------------------------------------------------------

ax2 = fig.add_subplot(gs[1, 0])
ax2.set_facecolor('#1a1a2e')

rf_labels  = list(rf_counts.index)
rf_sizes   = list(rf_counts.values)
rf_colors  = [BEAT_COLORS.get(l, '#888888') for l in rf_labels]

ax2.pie(rf_sizes, labels=rf_labels,
        colors=rf_colors,
        explode=[0.05]*len(rf_labels),
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 10})
ax2.set_title('RF Beat Distribution',
              color='white', fontsize=11, pad=8)

# -------------------------------------------------------
# PANEL 3: CNN Beat Distribution
# -------------------------------------------------------

ax3 = fig.add_subplot(gs[1, 1])
ax3.set_facecolor('#1a1a2e')

cnn_labels = list(cnn_counts.index)
cnn_sizes  = list(cnn_counts.values)
cnn_colors = [BEAT_COLORS.get(l, '#888888') for l in cnn_labels]

ax3.pie(cnn_sizes, labels=cnn_labels,
        colors=cnn_colors,
        explode=[0.05]*len(cnn_labels),
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 10})
ax3.set_title('CNN Beat Distribution',
              color='white', fontsize=11, pad=8)

# -------------------------------------------------------
# PANEL 4: Clinical Summary
# -------------------------------------------------------

ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor('#1a1a2e')
ax4.axis('off')

avg_hr   = features_df['heart_rate_bpm'].mean()
min_hr   = features_df['heart_rate_bpm'].min()
max_hr   = features_df['heart_rate_bpm'].max()
avg_st   = features_df['st_deviation_mv'].mean()
duration = len(ecg_signal) / fs / 60

# Normal heart rate check
if avg_hr < 60:
    hr_status = 'BRADYCARDIA'
    hr_color  = '#e67e22'
elif avg_hr > 100:
    hr_status = 'TACHYCARDIA'
    hr_color  = '#e74c3c'
else:
    hr_status = 'NORMAL'
    hr_color  = '#2ecc71'

# ST deviation check
if abs(avg_st) > 0.1:
    st_status = 'ABNORMAL'
    st_color  = '#e74c3c'
else:
    st_status = 'NORMAL'
    st_color  = '#2ecc71'

# Per class accuracy
per_class_lines = []
if known_count > 0:
    for label in ['N', 'A', 'V']:
        mask = true_arr[known_mask] == label
        if mask.sum() > 0:
            rf_ca  = (rf_predictions[known_mask][mask] == label).mean()
            cnn_ca = (cnn_predictions[known_mask][mask] == label).mean()
            per_class_lines.append(
                (f'{label}: RF {rf_ca*100:.0f}%  CNN {cnn_ca*100:.0f}%',
                 BEAT_COLORS.get(label, 'white'), 9, False)
            )

summary_lines = [
    ('CLINICAL SUMMARY',                    'white',       12, True),
    ('',                                    'white',        8, False),
    (f'Record       : {record_name}',       '#aaaaaa',     9, False),
    (f'Duration     : {duration:.1f} min',  '#aaaaaa',     9, False),
    (f'Total Beats  : {len(features_df)}',  '#aaaaaa',     9, False),
    ('',                                    'white',        8, False),
    ('HEART RATE',                          '#4a9eff',      9, True),
    (f'Average : {avg_hr:.1f} BPM',         'white',        9, False),
    (f'Min     : {min_hr:.1f} BPM',         'white',        9, False),
    (f'Max     : {max_hr:.1f} BPM',         'white',        9, False),
    (f'Status  : {hr_status}',              hr_color,       9, True),
    ('',                                    'white',        8, False),
    ('ST DEVIATION',                        '#4a9eff',      9, True),
    (f'Average : {avg_st:.4f} mV',          'white',        9, False),
    (f'Status  : {st_status}',              st_color,       9, True),
    ('',                                    'white',        8, False),
    ('MODEL ACCURACY',                      '#4a9eff',      9, True),
    (f'RF  : {rf_acc_str}',                 '#2ecc71',      9, False),
    (f'CNN : {cnn_acc_str}',                '#2ecc71',      9, False),
    ('',                                    'white',        8, False),
    ('PER CLASS ACCURACY',                  '#4a9eff',      9, True),
] + per_class_lines

y_pos = 0.97
for text, color, size, bold in summary_lines:
    weight = 'bold' if bold else 'normal'
    ax4.text(0.05, y_pos, text,
             transform=ax4.transAxes,
             color=color, fontsize=size,
             fontweight=weight,
             verticalalignment='top',
             fontfamily='monospace')
    y_pos -= 0.048

for spine in ax4.spines.values():
    spine.set_edgecolor('#444444')
    spine.set_linewidth(0.5)
    spine.set_visible(True)

# -------------------------------------------------------
# PANEL 5: Heart Rate Over Time
# -------------------------------------------------------

ax5 = fig.add_subplot(gs[2, :2])
ax5.set_facecolor('#1a1a2e')

time_minutes = features_df['r_peak_sample'] / fs / 60

for label, color in BEAT_COLORS.items():
    mask = features_df['rf_predicted'] == label
    if mask.sum() > 0:
        ax5.scatter(time_minutes[mask],
                    features_df[mask]['heart_rate_bpm'],
                    color=color, s=6, alpha=0.7, label=label)

ax5.axhline(y=60,  color='#ffff00', linewidth=0.8,
            linestyle='--', alpha=0.5, label='60 BPM')
ax5.axhline(y=100, color='#ff6666', linewidth=0.8,
            linestyle='--', alpha=0.5, label='100 BPM')
ax5.set_title('Heart Rate Over Time (RF Classification)',
              color='white', fontsize=11, pad=8)
ax5.set_xlabel('Time (minutes)', color='#aaaaaa')
ax5.set_ylabel('Heart Rate (BPM)', color='#aaaaaa')
ax5.tick_params(colors='#aaaaaa')
ax5.spines['bottom'].set_color('#444444')
ax5.spines['left'].set_color('#444444')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.grid(True, alpha=0.15, color='#555555')
ax5.legend(facecolor='#1a1a2e', edgecolor='#444444',
           labelcolor='white', fontsize=8, markerscale=3)

# -------------------------------------------------------
# PANEL 6: RF vs CNN Agreement
# -------------------------------------------------------

ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor('#1a1a2e')

agree    = (rf_predictions == cnn_predictions).sum()
disagree = (rf_predictions != cnn_predictions).sum()
total    = len(rf_predictions)

ax6.pie(
    [agree, disagree],
    labels=[f'Agree\n{agree}', f'Disagree\n{disagree}'],
    colors=['#2ecc71', '#e74c3c'],
    explode=[0.05, 0.05],
    autopct='%1.1f%%', startangle=90,
    textprops={'color': 'white', 'fontsize': 10}
)
ax6.set_title(f'RF vs CNN Agreement\n({agree/total*100:.1f}% match)',
              color='white', fontsize=11, pad=8)

# -------------------------------------------------------
# PANEL 7: ST Deviation Over Time
# -------------------------------------------------------

ax7 = fig.add_subplot(gs[3, :2])
ax7.set_facecolor('#1a1a2e')

for label, color in BEAT_COLORS.items():
    mask = features_df['rf_predicted'] == label
    if mask.sum() > 0:
        ax7.scatter(time_minutes[mask],
                    features_df[mask]['st_deviation_mv'],
                    color=color, s=5, alpha=0.7, label=label)

ax7.axhline(y=0.1,  color='#ffff00', linewidth=0.8,
            linestyle='--', alpha=0.6, label='+0.1 mV')
ax7.axhline(y=-0.1, color='#ff6666', linewidth=0.8,
            linestyle='--', alpha=0.6, label='-0.1 mV')
ax7.axhline(y=0,    color='#ffffff', linewidth=0.5,
            linestyle='-', alpha=0.3)
ax7.set_title('ST Deviation Over Time (Ischemia Indicator)',
              color='white', fontsize=11, pad=8)
ax7.set_xlabel('Time (minutes)', color='#aaaaaa')
ax7.set_ylabel('ST Deviation (mV)', color='#aaaaaa')
ax7.tick_params(colors='#aaaaaa')
ax7.spines['bottom'].set_color('#444444')
ax7.spines['left'].set_color('#444444')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.grid(True, alpha=0.15, color='#555555')
ax7.legend(facecolor='#1a1a2e', edgecolor='#444444',
           labelcolor='white', fontsize=8, markerscale=3)

# -------------------------------------------------------
# PANEL 8: RR Interval Distribution
# -------------------------------------------------------

ax8 = fig.add_subplot(gs[3, 2])
ax8.set_facecolor('#1a1a2e')

for label, color in BEAT_COLORS.items():
    mask = features_df['rf_predicted'] == label
    if mask.sum() > 0:
        ax8.hist(features_df[mask]['rr_interval_ms'],
                 bins=30, alpha=0.7,
                 color=color, label=label,
                 edgecolor='none')
ax8.set_title('RR Interval Distribution',
              color='white', fontsize=11, pad=8)
ax8.set_xlabel('RR Interval (ms)', color='#aaaaaa')
ax8.set_ylabel('Count', color='#aaaaaa')
ax8.tick_params(colors='#aaaaaa')
ax8.spines['bottom'].set_color('#444444')
ax8.spines['left'].set_color('#444444')
ax8.spines['top'].set_visible(False)
ax8.spines['right'].set_visible(False)
ax8.grid(True, alpha=0.15, color='#555555')
ax8.legend(facecolor='#1a1a2e', edgecolor='#444444',
           labelcolor='white', fontsize=9)

# -------------------------------------------------------
# SECTION 13: Save and Open Dashboard
# -------------------------------------------------------

output_path = r'D:\ecg_arr\step5_dashboard.png'
plt.savefig(output_path, dpi=150,
            facecolor=fig.get_facecolor())
plt.close()

import subprocess
subprocess.Popen(['start', output_path], shell=True)

print(f"\nDashboard saved to     : {output_path}")
print(f"\n{'='*55}")
print(f"  ANALYSIS COMPLETE — Record {record_name}")
print(f"{'='*55}")
print(f"  {verdict_icon}  {verdict}")
print(f"  {verdict_detail}")
print(f"{'='*55}")
print(f"  Total beats     : {len(features_df)}")
print(f"  RF  Accuracy    : {rf_acc_str}")
print(f"  CNN Accuracy    : {cnn_acc_str}")
print(f"  RF  Risk        : {rf_risk}")
print(f"  CNN Risk        : {cnn_risk}")
print(f"  Model Agreement : {agree/total*100:.1f}%")
print(f"{'='*55}")
print(f"\nDashboard opening automatically...")