# app.py
# Purpose: Flask backend for ECG Arrhythmia Detection Web App
# Updated: Added full clinical explanation generator
# Run with: python app.py

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
import os
import base64
import io
import tensorflow as tf
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# -------------------------------------------------------
# SECTION 1: Configuration
# -------------------------------------------------------

MODEL_DIR     = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(MODEL_DIR, 'uploads')
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

# -------------------------------------------------------
# SECTION 2: Load Models at Startup
# -------------------------------------------------------

print("Loading models...")

with open(os.path.join(MODEL_DIR, 'best_model.pkl'), 'rb') as f:
    rf_model = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

cnn_model = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, 'cnn_model.keras'),
    compile=False)

with open(os.path.join(MODEL_DIR, 'cnn_encoder.pkl'), 'rb') as f:
    encoder = pickle.load(f)

print("Models loaded successfully!")

# -------------------------------------------------------
# SECTION 3: Helper Functions
# -------------------------------------------------------

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Bandpass filter to remove ECG noise"""
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
        return 'HIGH RISK'
    elif v_pct > 2 or a_pct > 5 or a_count > 30:
        return 'MODERATE'
    else:
        return 'LOW RISK'

def get_verdict(rf_predictions, cnn_predictions):
    """
    Determines arrhythmia verdict using average of RF and CNN.

    Returns:
        dict with verdict text, type, icon and detail
    """
    total           = len(rf_predictions)
    rf_counts_dict  = pd.Series(rf_predictions).value_counts().to_dict()
    cnn_counts_dict = pd.Series(cnn_predictions).value_counts().to_dict()
    rf_v_pct        = rf_counts_dict.get('V', 0)  / total * 100
    rf_a_pct        = rf_counts_dict.get('A', 0)  / total * 100
    cnn_v_pct       = cnn_counts_dict.get('V', 0) / total * 100
    cnn_a_pct       = cnn_counts_dict.get('A', 0) / total * 100
    avg_v_pct       = (rf_v_pct  + cnn_v_pct)  / 2
    avg_a_pct       = (rf_a_pct  + cnn_a_pct)  / 2
    has_v           = avg_v_pct > 5
    has_a           = avg_a_pct > 5

    if has_v and has_a:
        return {
            'verdict': 'MULTIPLE ARRHYTHMIAS DETECTED',
            'type'   : 'danger',
            'icon'   : '🔴',
            'detail' : f'Ventricular: {avg_v_pct:.1f}%  |  '
                       f'Atrial: {avg_a_pct:.1f}%',
            'v_pct'  : round(avg_v_pct, 1),
            'a_pct'  : round(avg_a_pct, 1),
        }
    elif has_v:
        return {
            'verdict': 'VENTRICULAR ARRHYTHMIA DETECTED',
            'type'   : 'danger',
            'icon'   : '🔴',
            'detail' : f'Ventricular beats: {avg_v_pct:.1f}% of total',
            'v_pct'  : round(avg_v_pct, 1),
            'a_pct'  : 0,
        }
    elif has_a:
        return {
            'verdict': 'ATRIAL ARRHYTHMIA DETECTED',
            'type'   : 'warning',
            'icon'   : '⚠️',
            'detail' : f'Atrial beats: {avg_a_pct:.1f}% of total',
            'v_pct'  : 0,
            'a_pct'  : round(avg_a_pct, 1),
        }
    else:
        return {
            'verdict': 'NO ARRHYTHMIA DETECTED',
            'type'   : 'success',
            'icon'   : '✅',
            'detail' : 'Rhythm appears normal',
            'v_pct'  : 0,
            'a_pct'  : 0,
        }

def generate_explanation(rf_predictions, cnn_predictions,
                          features_df, verdict_data,
                          rf_risk, cnn_risk):
    """
    Generates detailed clinical explanation for the verdict.

    Parameters:
        rf_predictions  : RF model predictions array
        cnn_predictions : CNN model predictions array
        features_df     : DataFrame with extracted features
        verdict_data    : verdict dict from get_verdict()
        rf_risk         : RF risk level string
        cnn_risk        : CNN risk level string

    Returns:
        dict with 5 explanation sections
    """
    total           = len(rf_predictions)
    rf_counts_dict  = pd.Series(rf_predictions).value_counts().to_dict()
    cnn_counts_dict = pd.Series(cnn_predictions).value_counts().to_dict()

    rf_n  = rf_counts_dict.get('N', 0)
    rf_a  = rf_counts_dict.get('A', 0)
    rf_v  = rf_counts_dict.get('V', 0)
    cnn_n = cnn_counts_dict.get('N', 0)
    cnn_a = cnn_counts_dict.get('A', 0)
    cnn_v = cnn_counts_dict.get('V', 0)

    avg_v_pct = ((rf_v / total) + (cnn_v / total)) / 2 * 100
    avg_a_pct = ((rf_a / total) + (cnn_a / total)) / 2 * 100

    avg_hr  = round(float(features_df['heart_rate_bpm'].mean()), 1)
    max_hr  = round(float(features_df['heart_rate_bpm'].max()),  1)
    min_hr  = round(float(features_df['heart_rate_bpm'].min()),  1)
    avg_rr  = round(float(features_df['rr_interval_ms'].mean()), 1)
    avg_rrv = round(float(features_df['rr_variability_ms'].mean()), 1)
    avg_st  = round(float(features_df['st_deviation_mv'].mean()), 4)

    verdict_type = verdict_data['type']

    # -----------------------------------------------
    # WHY IT WAS DETECTED
    # -----------------------------------------------
    why = []

    if rf_v > 0:
        why.append(
            f'RF model detected {rf_v} Ventricular beats '
            f'({rf_v/total*100:.1f}% of total)'
        )
    if cnn_v > 0:
        why.append(
            f'CNN model detected {cnn_v} Ventricular beats '
            f'({cnn_v/total*100:.1f}% of total)'
        )
    if rf_a > 0:
        why.append(
            f'RF model detected {rf_a} Atrial beats '
            f'({rf_a/total*100:.1f}% of total)'
        )
    if cnn_a > 0:
        why.append(
            f'CNN model detected {cnn_a} Atrial beats '
            f'({cnn_a/total*100:.1f}% of total)'
        )
    if rf_risk == cnn_risk:
        why.append(
            f'Both RF and CNN models agree: {rf_risk}'
        )
    else:
        why.append(
            f'RF assessed {rf_risk} — CNN assessed {cnn_risk}'
        )
    if verdict_type == 'success':
        why.append(
            'Less than 5% abnormal beats detected by both models'
        )

    # -----------------------------------------------
    # FEATURES THAT TRIGGERED DETECTION
    # -----------------------------------------------
    features_triggered = []

    if avg_rrv > 100:
        features_triggered.append(
            f'RR Variability is HIGH ({avg_rrv}ms avg) — '
            f'indicates irregular rhythm between beats'
        )
    elif avg_rrv > 50:
        features_triggered.append(
            f'RR Variability is MODERATE ({avg_rrv}ms avg) — '
            f'some beat-to-beat irregularity detected'
        )
    else:
        features_triggered.append(
            f'RR Variability is LOW ({avg_rrv}ms avg) — '
            f'regular rhythm pattern'
        )

    if abs(avg_st) > 0.1:
        features_triggered.append(
            f'ST Deviation is ABNORMAL ({avg_st}mV) — '
            f'possible cardiac stress or ischemia'
        )
    else:
        features_triggered.append(
            f'ST Deviation is normal ({avg_st}mV) — '
            f'no ischemia signs detected'
        )

    if max_hr > 100:
        features_triggered.append(
            f'Heart rate reached {max_hr} BPM — '
            f'tachycardia episodes detected'
        )
    if min_hr < 60:
        features_triggered.append(
            f'Heart rate dropped to {min_hr} BPM — '
            f'bradycardia episodes detected'
        )

    features_triggered.append(
        f'Average RR Interval: {avg_rr}ms '
        f'(normal range: 600-1000ms)'
    )

    # -----------------------------------------------
    # BEAT BY BEAT BREAKDOWN
    # -----------------------------------------------
    breakdown = [
        {
            'type'   : 'Normal (N)',
            'rf'     : rf_n,
            'cnn'    : cnn_n,
            'pct'    : round(rf_n / total * 100, 1),
            'meaning': 'Regular heartbeat with normal '
                       'electrical conduction pathway',
            'color'  : 'green'
        },
        {
            'type'   : 'Atrial (A)',
            'rf'     : rf_a,
            'cnn'    : cnn_a,
            'pct'    : round(rf_a / total * 100, 1),
            'meaning': 'Premature beat from atria — '
                       'PAC or Atrial Fibrillation pattern',
            'color'  : 'orange'
        },
        {
            'type'   : 'Ventricular (V)',
            'rf'     : rf_v,
            'cnn'    : cnn_v,
            'pct'    : round(rf_v / total * 100, 1),
            'meaning': 'Premature beat from ventricles — '
                       'PVC or Ventricular Tachycardia pattern',
            'color'  : 'red'
        },
    ]

    # -----------------------------------------------
    # CLINICAL INTERPRETATION
    # -----------------------------------------------
    clinical = []

    if avg_a_pct > 10:
        clinical.append(
            f'HIGH Atrial activity ({avg_a_pct:.1f}%) suggests '
            f'possible Atrial Fibrillation (AFib) or frequent '
            f'Premature Atrial Contractions (PAC). AFib is the '
            f'most common sustained arrhythmia and significantly '
            f'increases stroke risk. Immediate evaluation recommended.'
        )
    elif avg_a_pct > 2:
        clinical.append(
            f'MODERATE Atrial activity ({avg_a_pct:.1f}%) detected. '
            f'Occasional Premature Atrial Contractions (PAC) are '
            f'common and often benign but should be monitored '
            f'if symptomatic.'
        )

    if avg_v_pct > 10:
        clinical.append(
            f'HIGH Ventricular activity ({avg_v_pct:.1f}%) suggests '
            f'possible Premature Ventricular Contractions (PVC) or '
            f'Ventricular Tachycardia (VT). Frequent PVCs can '
            f'indicate underlying heart disease and require '
            f'immediate evaluation.'
        )
    elif avg_v_pct > 2:
        clinical.append(
            f'MODERATE Ventricular activity ({avg_v_pct:.1f}%) '
            f'detected. Occasional PVCs can occur in healthy '
            f'individuals but frequent episodes require monitoring.'
        )

    if abs(avg_st) > 0.1:
        clinical.append(
            f'ST segment deviation of {avg_st}mV detected. '
            f'Abnormal ST changes can indicate myocardial '
            f'ischemia, injury, or infarction. '
            f'Immediate cardiac evaluation is strongly recommended.'
        )
    else:
        clinical.append(
            f'ST segment deviation of {avg_st}mV is within '
            f'normal limits. No signs of ischemia detected '
            f'in this ECG recording.'
        )

    if avg_hr > 100:
        clinical.append(
            f'Average heart rate of {avg_hr} BPM indicates '
            f'Tachycardia (normal: 60-100 BPM). This can be '
            f'caused by arrhythmia, stress, dehydration, '
            f'fever, or underlying cardiac conditions.'
        )
    elif avg_hr < 60:
        clinical.append(
            f'Average heart rate of {avg_hr} BPM indicates '
            f'Bradycardia (normal: 60-100 BPM). This may be '
            f'normal in trained athletes but can also indicate '
            f'conduction system disease or medication effects.'
        )
    else:
        clinical.append(
            f'Average heart rate of {avg_hr} BPM is within '
            f'normal range (60-100 BPM). No rate abnormality '
            f'detected in this recording.'
        )

    if verdict_type == 'success':
        clinical.append(
            'No significant arrhythmia detected in this ECG. '
            'Heart rhythm appears normal based on AI analysis. '
            'Both Random Forest and CNN models agree on '
            'normal classification.'
        )

    # -----------------------------------------------
    # RECOMMENDED NEXT STEPS
    # -----------------------------------------------
    if verdict_type == 'danger':
        next_steps = [
            '🚨 Consult a cardiologist immediately',
            '📋 Perform a standard 12-lead ECG for confirmation',
            '🏥 Consider hospital evaluation if symptomatic',
            '📱 Request 24-hour Holter monitor study',
            '💊 Discuss antiarrhythmic medication with your doctor',
            '⚠️  Avoid strenuous physical activity until evaluated',
        ]
    elif verdict_type == 'warning':
        next_steps = [
            '👨‍⚕️ Schedule appointment with cardiologist',
            '📋 Perform a standard 12-lead ECG for confirmation',
            '📱 Consider 24-hour Holter monitor study',
            '📝 Keep a symptom diary (palpitations, dizziness)',
            '🧘 Reduce caffeine, alcohol, and stress',
            '🔄 Follow up ECG in 3-6 months',
        ]
    else:
        next_steps = [
            '✅ No immediate action required',
            '📅 Routine annual cardiac checkup recommended',
            '🏃 Maintain regular physical activity',
            '🥗 Continue heart-healthy diet and lifestyle',
            '📝 Monitor for new symptoms (palpitations, chest pain)',
            '🔄 Repeat ECG if symptoms develop',
        ]

    return {
        'why'               : why,
        'features_triggered': features_triggered,
        'breakdown'         : breakdown,
        'clinical'          : clinical,
        'next_steps'        : next_steps,
    }

def generate_ecg_plot(ecg_filtered, valid_r_peaks,
                       rf_predictions, fs, n_seconds=10):
    """Generates ECG waveform plot and returns as base64 string"""

    BEAT_COLORS = {'N': '#2ecc71', 'A': '#e67e22', 'V': '#e74c3c'}
    qrs_before  = int(0.050 * fs)
    qrs_after   = int(0.050 * fs)
    n_samples   = min(n_seconds * fs, len(ecg_filtered))
    time_axis   = np.arange(n_samples) / fs

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    ax.plot(time_axis, ecg_filtered[:n_samples],
            color='#4a9eff', linewidth=0.8, alpha=0.9)

    r_in_window = valid_r_peaks[valid_r_peaks < n_samples]
    for r in r_in_window:
        idx = np.where(valid_r_peaks == r)[0]
        if len(idx) == 0:
            continue
        pred  = rf_predictions[idx[0]]
        color = BEAT_COLORS.get(pred, 'white')
        ax.scatter(r/fs, ecg_filtered[r],
                   color=color, s=60, zorder=3)
        ax.axvspan((r - qrs_before)/fs,
                   (r + qrs_after)/fs,
                   alpha=0.15, color=color)

    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color='#2ecc71', label='Normal'),
        mpatches.Patch(color='#e67e22', label='Atrial'),
        mpatches.Patch(color='#e74c3c', label='Ventricular'),
    ]
    ax.legend(handles=patches, loc='upper right',
              facecolor='#1a1a2e', edgecolor='#444444',
              labelcolor='white', fontsize=9)

    ax.set_xlabel('Time (seconds)', color='#aaaaaa')
    ax.set_ylabel('Amplitude (mV)', color='#aaaaaa')
    ax.tick_params(colors='#aaaaaa')
    ax.spines['bottom'].set_color('#444444')
    ax.spines['left'].set_color('#444444')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#555555')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                facecolor=fig.get_facecolor())
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

# -------------------------------------------------------
# SECTION 4: Routes
# -------------------------------------------------------

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')
@app.route('/test')
def test():
    """Test route to confirm server is running"""
    return jsonify({
        'status' : 'Server is running',
        'models' : 'RF + CNN loaded',
        'message': 'Ready to analyze ECG'
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Main analysis endpoint.
    Receives .dat + .hea + .atr files
    Returns full analysis + explanation as JSON
    """

    try:
        if 'dat_file' not in request.files:
            return jsonify({'error': 'No .dat file uploaded'}), 400

        dat_file = request.files['dat_file']
        hea_file = request.files.get('hea_file')
        atr_file = request.files.get('atr_file')

        if dat_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        record_name = os.path.splitext(dat_file.filename)[0]

        # Save uploaded files
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        dat_path = os.path.join(UPLOAD_FOLDER, dat_file.filename)
        dat_file.save(dat_path)

        if hea_file:
            hea_path = os.path.join(UPLOAD_FOLDER, hea_file.filename)
            hea_file.save(hea_path)

        if atr_file:
            atr_path = os.path.join(UPLOAD_FOLDER, atr_file.filename)
            atr_file.save(atr_path)

        record_path = os.path.join(UPLOAD_FOLDER, record_name)

        if not os.path.exists(record_path + '.hea'):
            return jsonify({
                'error': 'Missing .hea file. '
                         'Please upload .dat AND .hea files together.'
            }), 400

        # -----------------------------------------------
        # Load and process ECG
        # -----------------------------------------------

        record     = wfdb.rdrecord(record_path)
        ecg_signal = record.p_signal[:, 0]
        fs         = record.fs

        ecg_filtered = bandpass_filter(ecg_signal, fs=fs)
        r_peaks      = detect_r_peaks(ecg_filtered, fs=fs)

        # Load annotations if available
        atr_exists  = os.path.exists(record_path + '.atr')
        ann_samples = np.array([])
        ann_symbols = np.array([])

        if atr_exists:
            annotation  = wfdb.rdann(record_path, 'atr')
            ann_samples = annotation.sample
            ann_symbols = annotation.symbol

        # -----------------------------------------------
        # Extract features + segments
        # -----------------------------------------------

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

        for i, r in enumerate(r_peaks):
            if i == 0:
                continue
            if (r - max(p_start, WINDOW_BEFORE)) < 0:
                continue
            if (r + max(st_end, WINDOW_AFTER)) >= len(ecg_filtered):
                continue
            try:
                rr_samples     = int(r_peaks[i]) - int(r_peaks[i-1])
                rr_interval    = (rr_samples / fs) * 1000
                if rr_interval < 200 or rr_interval > 3000:
                    continue
                heart_rate     = round(60000 / rr_interval, 2)
                qrs_duration   = round(
                    ((qrs_before + qrs_after) / fs) * 1000, 2)
                pr_interval    = round((p_start / fs) * 1000, 2)
                st_segment     = ecg_filtered[r + st_start : r + st_end]
                st_deviation   = round(float(np.mean(st_segment)), 4)
                if i >= 2:
                    prev_rr        = (int(r_peaks[i-1]) -
                                      int(r_peaks[i-2])) / fs * 1000
                    rr_variability = round(abs(rr_interval - prev_rr), 2)
                else:
                    rr_variability = 0.0

                segment  = ecg_filtered[
                    r - WINDOW_BEFORE : r + WINDOW_AFTER]
                seg_min  = segment.min()
                seg_max  = segment.max()
                if seg_max - seg_min <= 0:
                    continue
                segment_norm = (2 * (segment - seg_min) /
                                (seg_max - seg_min) - 1)

                if atr_exists and len(ann_samples) > 0:
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

        # -----------------------------------------------
        # RF Predictions
        # -----------------------------------------------

        X_rf           = scaler.transform(
            features_df[feature_cols].values)
        rf_predictions = rf_model.predict(X_rf)
        features_df['rf_predicted'] = rf_predictions

        # -----------------------------------------------
        # CNN Predictions
        # -----------------------------------------------

        X_cnn           = segments_arr.reshape(
            segments_arr.shape[0], segments_arr.shape[1], 1)
        cnn_pred_prob   = cnn_model.predict(X_cnn, verbose=0)
        cnn_pred_idx    = np.argmax(cnn_pred_prob, axis=1)
        cnn_predictions = encoder.inverse_transform(cnn_pred_idx)
        features_df['cnn_predicted'] = cnn_predictions

        # -----------------------------------------------
        # Accuracy vs ground truth
        # -----------------------------------------------

        known_mask  = true_arr != 'Unknown'
        known_count = known_mask.sum()
        rf_acc      = None
        cnn_acc     = None

        if known_count > 0:
            rf_acc  = (rf_predictions[known_mask] ==
                       true_arr[known_mask]).mean()
            cnn_acc = (cnn_predictions[known_mask] ==
                       true_arr[known_mask]).mean()

        per_class = {}
        if known_count > 0:
            for label in ['N', 'A', 'V']:
                mask = true_arr[known_mask] == label
                if mask.sum() > 0:
                    per_class[label] = {
                        'count'  : int(mask.sum()),
                        'rf_acc' : round(float(
                            (rf_predictions[known_mask][mask] ==
                             label).mean()) * 100, 1),
                        'cnn_acc': round(float(
                            (cnn_predictions[known_mask][mask] ==
                             label).mean()) * 100, 1),
                    }

        # -----------------------------------------------
        # Verdict + Risk + Explanation
        # -----------------------------------------------

        verdict_data = get_verdict(rf_predictions, cnn_predictions)
        rf_risk      = assess_risk(rf_predictions)
        cnn_risk     = assess_risk(cnn_predictions)
        explanation  = generate_explanation(
            rf_predictions, cnn_predictions,
            features_df, verdict_data,
            rf_risk, cnn_risk
        )

        # -----------------------------------------------
        # Beat counts + clinical stats
        # -----------------------------------------------

        rf_counts  = pd.Series(rf_predictions).value_counts().to_dict()
        cnn_counts = pd.Series(cnn_predictions).value_counts().to_dict()

        avg_hr = round(float(features_df['heart_rate_bpm'].mean()), 1)
        min_hr = round(float(features_df['heart_rate_bpm'].min()),  1)
        max_hr = round(float(features_df['heart_rate_bpm'].max()),  1)
        avg_st = round(float(features_df['st_deviation_mv'].mean()), 4)
        avg_rr = round(float(features_df['rr_interval_ms'].mean()),  1)

        if avg_hr < 60:
            hr_status = 'BRADYCARDIA'
        elif avg_hr > 100:
            hr_status = 'TACHYCARDIA'
        else:
            hr_status = 'NORMAL'

        st_status = 'ABNORMAL' if abs(avg_st) > 0.1 else 'NORMAL'

        agree     = int((rf_predictions == cnn_predictions).sum())
        agree_pct = round(agree / len(rf_predictions) * 100, 1)

        # -----------------------------------------------
        # Generate ECG plot
        # -----------------------------------------------

        ecg_plot = generate_ecg_plot(
            ecg_filtered, valid_r_peaks,
            rf_predictions, fs
        )

        # Heart rate time series
        time_minutes = (features_df['r_peak_sample'] /
                        fs / 60).tolist()
        hr_values    = features_df['heart_rate_bpm'].tolist()
        hr_labels    = features_df['rf_predicted'].tolist()

        # -----------------------------------------------
        # Return all results
        # -----------------------------------------------

        return jsonify({
            'success'     : True,
            'record_name' : record_name,
            'duration_min': round(len(ecg_signal) / fs / 60, 1),
            'total_beats' : len(features_df),
            'r_peaks'     : len(r_peaks),

            # Verdict
            'verdict'     : verdict_data,

            # Risk
            'rf_risk'     : rf_risk,
            'cnn_risk'    : cnn_risk,

            # Beat counts
            'rf_counts'   : rf_counts,
            'cnn_counts'  : cnn_counts,

            # Accuracy
            'rf_acc'      : round(rf_acc  * 100, 1) if rf_acc  else None,
            'cnn_acc'     : round(cnn_acc * 100, 1) if cnn_acc else None,
            'per_class'   : per_class,

            # Clinical stats
            'avg_hr'      : avg_hr,
            'min_hr'      : min_hr,
            'max_hr'      : max_hr,
            'hr_status'   : hr_status,
            'avg_st'      : avg_st,
            'st_status'   : st_status,
            'avg_rr'      : avg_rr,

            # Model agreement
            'agree_pct'   : agree_pct,

            # ECG plot (base64)
            'ecg_plot'    : ecg_plot,

            # Heart rate time series
            'time_minutes': time_minutes,
            'hr_values'   : hr_values,
            'hr_labels'   : hr_labels,

            # Full explanation
            'explanation' : explanation,
        })

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"ERROR: {error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500

# -------------------------------------------------------
# SECTION 5: Run App
# -------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("\n" + "="*50)
    print("  ECG Arrhythmia Detection Web App")
    print("="*50)
    print("  Open browser at: http://localhost:5000")
