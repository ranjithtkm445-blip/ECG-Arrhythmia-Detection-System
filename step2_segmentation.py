# step2_segmentation.py
# Purpose: Detect R-peaks and segment P, QRS, T waves from cleaned ECG signal
# Uses Pan-Tompkins inspired algorithm for R-peak detection
# Output: Color-coded ECG plot showing all wave components

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# -------------------------------------------------------
# SECTION 1: Load and Filter ECG (Same as Step 1)
# -------------------------------------------------------

DATA_PATH   = r'D:\ecg_arr\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'
RECORD_NAME = '100'

record     = wfdb.rdrecord(f'{DATA_PATH}\\{RECORD_NAME}')
ecg_signal = record.p_signal[:, 0]
fs         = record.fs  # 360 Hz

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """Bandpass filter to clean ECG signal (same as Step 1)"""
    nyquist = 0.5 * fs
    low     = lowcut / nyquist
    high    = highcut / nyquist
    b, a    = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Apply bandpass filter to clean the signal first
ecg_filtered = bandpass_filter(ecg_signal, fs=fs)
print(f"Record {RECORD_NAME} loaded and filtered. fs={fs} Hz")

# -------------------------------------------------------
# SECTION 2: R-Peak Detection (Pan-Tompkins Inspired)
# -------------------------------------------------------
# Pan-Tompkins is the gold standard algorithm for R-peak detection
# Steps:
#   1. Differentiate  → highlights steep slopes (QRS has steepest slope)
#   2. Square         → makes all values positive, amplifies large slopes
#   3. Moving average → smooths the signal into a clear envelope
#   4. Find peaks     → locate the highest points = R-peaks

def detect_r_peaks(signal, fs=360):
    """
    Detects R-peaks in ECG signal using Pan-Tompkins inspired method.

    Parameters:
        signal : filtered ECG signal (numpy array)
        fs     : sampling frequency in Hz

    Returns:
        r_peaks : array of sample indices where R-peaks occur
    """

    # Step 1: Differentiate the signal
    # Highlights rapid changes (QRS complex has the steepest rise)
    diff_signal = np.diff(signal)

    # Step 2: Square the signal
    # Makes all values positive and amplifies large slopes
    squared = diff_signal ** 2

    # Step 3: Moving window integration (moving average)
    # Window size = 150ms worth of samples (standard Pan-Tompkins window)
    window_size = int(0.150 * fs)   # 0.150 sec × 360 Hz = 54 samples
    kernel      = np.ones(window_size) / window_size
    integrated  = np.convolve(squared, kernel, mode='same')

    # Step 4: Find peaks in the integrated signal
    # min_distance = 200ms (no two heartbeats closer than 200ms = 300 bpm max)
    # height threshold = 40% of max (filters out small noise peaks)
    min_distance = int(0.200 * fs)       # 0.200 sec × 360 Hz = 72 samples
    height_thresh = 0.4 * np.max(integrated)

    r_peaks, _ = find_peaks(integrated,
                             distance=min_distance,
                             height=height_thresh)

    return r_peaks

# Detect R-peaks
r_peaks = detect_r_peaks(ecg_filtered, fs=fs)
print(f"Total R-peaks detected : {len(r_peaks)}")
print(f"Average Heart Rate     : {len(r_peaks) / (len(ecg_filtered)/fs) * 60:.1f} BPM")

# -------------------------------------------------------
# SECTION 3: P Wave, QRS, T Wave Segmentation
# -------------------------------------------------------
# Clinical timing windows around each R-peak:
#   QRS complex : 50ms before to 50ms after R-peak
#   P wave      : 200ms to 50ms BEFORE R-peak
#   T wave      : 50ms to 350ms AFTER R-peak

def segment_waves(signal, r_peaks, fs=360):
    """
    Estimates P wave, QRS complex, and T wave boundaries around each R-peak.

    Parameters:
        signal  : filtered ECG signal
        r_peaks : array of R-peak sample indices
        fs      : sampling frequency

    Returns:
        Dictionary with lists of (start, end) sample pairs for each wave
    """

    # Convert time windows to samples
    qrs_before = int(0.050 * fs)   # 50ms before R-peak  = 18 samples
    qrs_after  = int(0.050 * fs)   # 50ms after R-peak   = 18 samples
    p_start    = int(0.200 * fs)   # 200ms before R-peak = 72 samples
    p_end      = int(0.050 * fs)   # 50ms before R-peak  = 18 samples
    t_start    = int(0.050 * fs)   # 50ms after R-peak   = 18 samples
    t_end      = int(0.350 * fs)   # 350ms after R-peak  = 126 samples

    segments = {'P': [], 'QRS': [], 'T': []}

    for r in r_peaks:
        # QRS complex boundaries
        qrs_s = r - qrs_before
        qrs_e = r + qrs_after

        # P wave boundaries (before QRS)
        p_s = r - p_start
        p_e = r - p_end

        # T wave boundaries (after QRS)
        t_s = r + t_start
        t_e = r + t_end

        # Only add if all boundaries are within signal range
        if p_s >= 0 and t_e < len(signal):
            segments['P'].append((p_s, p_e))
            segments['QRS'].append((qrs_s, qrs_e))
            segments['T'].append((t_s, t_e))

    return segments

# Get wave segments for all beats
segments = segment_waves(ecg_filtered, r_peaks, fs=fs)
print(f"Segments extracted     : {len(segments['QRS'])} beats")

# -------------------------------------------------------
# SECTION 4: Plot Segmented ECG (First 5 Seconds)
# -------------------------------------------------------

duration_to_show = 5
samples_to_show  = duration_to_show * fs
time_axis        = np.arange(samples_to_show) / fs

fig, ax = plt.subplots(figsize=(16, 6))

# Plot the filtered ECG signal
ax.plot(time_axis,
        ecg_filtered[:samples_to_show],
        color='black', linewidth=0.8, label='Filtered ECG', zorder=2)

# Highlight wave regions with colored shading
for (p_s, p_e) in segments['P']:
    if p_e < samples_to_show:
        ax.axvspan(p_s/fs, p_e/fs, alpha=0.25, color='blue', zorder=1)

for (q_s, q_e) in segments['QRS']:
    if q_e < samples_to_show:
        ax.axvspan(q_s/fs, q_e/fs, alpha=0.35, color='red', zorder=1)

for (t_s, t_e) in segments['T']:
    if t_e < samples_to_show:
        ax.axvspan(t_s/fs, t_e/fs, alpha=0.25, color='green', zorder=1)

# Mark R-peaks with orange dots
r_in_window = r_peaks[r_peaks < samples_to_show]
ax.scatter(r_in_window / fs,
           ecg_filtered[r_in_window],
           color='orange', s=60, zorder=3, label='R-peaks')

# Legend
p_patch   = mpatches.Patch(color='blue',   alpha=0.4, label='P Wave')
qrs_patch = mpatches.Patch(color='red',    alpha=0.5, label='QRS Complex')
t_patch   = mpatches.Patch(color='green',  alpha=0.4, label='T Wave')
ax.legend(handles=[p_patch, qrs_patch, t_patch],
          loc='upper right', fontsize=10)

ax.set_title(f'ECG Segmentation — Record {RECORD_NAME} | '
             f'{len(r_peaks)} beats detected | '
             f'{len(r_peaks)/(len(ecg_filtered)/fs)*60:.1f} BPM avg',
             fontsize=13)
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Amplitude (mV)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = r'D:\ecg_arr\step2_output.png'
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nPlot saved to: {output_path}")
print("Step 2 Complete! Ready for Step 3: Feature Extraction")