# step1_preprocessing.py
# Purpose: Load and clean raw ECG signals from your local MIT-BIH dataset
# Dataset path: D:\ecg_arr\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0
# Output: Cleaned ECG signal + comparison plot saved as step1_output.png

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

# -------------------------------------------------------
# SECTION 1: Define Dataset Path
# -------------------------------------------------------

# Full path to where your .dat and .hea files are located
DATA_PATH = r'D:\ecg_arr\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0'

# Record to load (record 100 = good starting point, has normal + arrhythmia beats)
RECORD_NAME = '100'

# -------------------------------------------------------
# SECTION 2: Load ECG Record
# -------------------------------------------------------

# Load the ECG record from your local folder
# rdrecord reads the .hea and .dat files together
record = wfdb.rdrecord(f'{DATA_PATH}\\{RECORD_NAME}')

# Extract channel 0 (MLII lead — the standard clinical lead in MIT-BIH)
ecg_signal = record.p_signal[:, 0]

# Get sampling frequency (MIT-BIH records are sampled at 360 Hz)
fs = record.fs

print(f"Record        : {RECORD_NAME}")
print(f"Sampling Rate : {fs} Hz")
print(f"Total Samples : {len(ecg_signal)}")
print(f"Duration      : {len(ecg_signal)/fs:.1f} seconds ({len(ecg_signal)/fs/60:.1f} minutes)")
print(f"Lead Name     : {record.sig_name[0]}")

# -------------------------------------------------------
# SECTION 3: Bandpass Filter (0.5 Hz – 40 Hz)
# -------------------------------------------------------
# Why these limits?
#   Below 0.5 Hz → baseline wander (body/breathing movement) → REMOVE
#   0.5 to 40 Hz → real ECG waveforms (P, QRS, T waves)    → KEEP
#   Above 40 Hz  → muscle noise + powerline hum (50/60 Hz) → REMOVE

def bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=360, order=4):
    """
    Butterworth bandpass filter to remove ECG noise.

    Parameters:
        signal  : raw ECG signal (numpy array)
        lowcut  : lower frequency cutoff in Hz (removes baseline wander)
        highcut : upper frequency cutoff in Hz (removes high freq noise)
        fs      : sampling frequency in Hz
        order   : filter sharpness (4 is standard for ECG)

    Returns:
        filtered_signal : cleaned ECG signal (numpy array)
    """
    nyquist = 0.5 * fs                               # Nyquist = half of sampling rate
    low     = lowcut / nyquist                       # Normalize lower cutoff
    high    = highcut / nyquist                      # Normalize upper cutoff
    b, a    = butter(order, [low, high], btype='band')  # Design Butterworth filter
    filtered_signal = filtfilt(b, a, signal)         # Zero-phase filtering (no signal shift)
    return filtered_signal

# Apply filter to the loaded ECG signal
ecg_filtered = bandpass_filter(ecg_signal, fs=fs)
print("\nFiltering complete!")

# -------------------------------------------------------
# SECTION 4: Plot Raw vs Filtered (First 5 Seconds)
# -------------------------------------------------------

duration_to_show = 5                              # Show first 5 seconds
samples_to_show  = duration_to_show * fs          # 5 × 360 = 1800 samples
time_axis        = np.arange(samples_to_show) / fs  # X-axis in seconds

plt.figure(figsize=(14, 7))

# --- Top plot: Raw Signal ---
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_signal[:samples_to_show],
         color='steelblue', linewidth=0.8, label='Raw ECG')
plt.title(f'RAW ECG Signal — Record {RECORD_NAME} (Before Filtering)', fontsize=13)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

# --- Bottom plot: Filtered Signal ---
plt.subplot(2, 1, 2)
plt.plot(time_axis, ecg_filtered[:samples_to_show],
         color='crimson', linewidth=0.8, label='Filtered ECG')
plt.title(f'FILTERED ECG Signal — After Bandpass Filter (0.5–40 Hz)', fontsize=13)
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()

# Save plot to same folder as the script
output_path = r'D:\ecg_arr\step1_output.png'
plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nPlot saved to: {output_path}")
print("\nStep 1 Complete! Ready for Step 2: Segmentation")