ECG Arrhythmia Detection System
AI-Powered Cardiac Analysis using Machine Learning and Deep Learning
Live App: https://ranjith445-ecg-arrhythmia-detection.hf.space
GitHub: https://github.com/ranjithtkm445-blip/ecg-arrhythmia-detection

1. Problem Statement
Every year thousands of people die from undetected heart arrhythmias. The main reason is simple — there are not enough cardiologists to manually read every ECG recording. A single 30-minute ECG has over 2000 heartbeats. A cardiologist needs 15 to 30 minutes to review one recording. In rural areas patients wait days for a result.

This project was built to solve that problem. The idea is straightforward — train a machine learning model on expert-labeled ECG data, then let it automatically scan an ECG and flag any abnormal beats in seconds.

Current Challenges:
•	Manual ECG reading takes 15 to 30 minutes per patient
•	Highly dependent on cardiologist experience and fatigue
•	Not enough cardiologists in rural and remote areas
•	Patients wait days or weeks for ECG interpretation

2. Project Objective
To design and implement an automated end-to-end ECG signal analysis system that analyzes 30 minutes of ECG data in under 60 seconds and provides a clinical verdict to assist early diagnosis of arrhythmia.

#	Objective	What We Built
1	Clean raw ECG signals	Bandpass filter 0.5 to 40 Hz
2	Segment heartbeats	Pan-Tompkins R-peak detection
3	Extract clinical features	6 feature extraction per beat
4	Classify each beat	Random Forest + CNN models
5	Generate clinical report	Dashboard with full explanation

3. What is Arrhythmia?
3.1 Normal Heart Function
The heart beats 60 to 100 times per minute at rest. Each beat follows a precise electrical pathway starting from the SA node, traveling through the atria, pausing at the AV node, then firing the ventricles. This produces the classic PQRST wave pattern on an ECG.

3.2 Beat Types Detected
Beat	Symbol	Origin	Risk Level
Normal	N	SA Node — correct electrical pathway	Safe
Atrial	A	Fires early from atria	Moderate
Ventricular	V	Fires directly from ventricles	High

3.3 Common Conditions
Condition	Description	Risk
Atrial Fibrillation (AFib)	Atria fire chaotically — most common arrhythmia	High — 5x stroke risk
PAC	Single early atrial beat	Low to moderate
PVC	Single early ventricular beat	Moderate
Ventricular Tachycardia	Rapid series of ventricular beats	Very high
Bradycardia	Heart rate below 60 BPM	Moderate
Tachycardia	Heart rate above 100 BPM	Moderate

4. How to Read an ECG for Arrhythmia
4.1 The PQRST Complex
Every heartbeat produces a characteristic wave called the PQRST complex:

Wave	Duration	Meaning
P wave	80 to 120 ms	Atria contracting — upper chambers fire
QRS Complex	80 to 120 ms	Ventricles contracting — main pump fires
T wave	160 ms	Ventricles recovering — resetting for next beat
RR Interval	600 to 1000 ms	Time between two beats — determines heart rate
ST Segment	80 to 120 ms	Ventricles fully depolarized — ischemia indicator

4.2 Arrhythmia Signs to Look For
•	Heart rate outside 60 to 100 BPM range
•	Irregular RR spacing — unequal time between beats
•	Missing P wave before QRS — suggests ventricular origin
•	Wide QRS above 120ms — confirms ventricular beat
•	ST segment elevation or depression — suggests ischemia

4.3 ECG Appearance of Each Beat Type
Beat Type	Visual Characteristics
Normal (N)	Narrow QRS, clear P wave before it, regular timing from previous beat
Atrial (A)	Abnormal or absent P wave, similar QRS shape, arrives earlier than expected
Ventricular (V)	Wide bizarre QRS above 120ms, no P wave, inverted T wave, compensatory pause after

4.4 Key Clinical Measurements
Measurement	Normal Range	Abnormal Indicates
Heart Rate	60 to 100 BPM	Bradycardia or Tachycardia
RR Interval	600 to 1000 ms	Rate abnormality
RR Variability	Low	High = irregular rhythm
QRS Duration	80 to 120 ms	Wide = ventricular origin
PR Interval	120 to 200 ms	Long = heart block
ST Deviation	-0.1 to 0.1 mV	Outside range = ischemia or injury

5. How the AI Detects Arrhythmia
5.1 Model 1 — Random Forest (Feature Based)
Extracts 6 numerical measurements from each beat and feeds them into 200 decision trees. Each tree votes on the beat type and the majority vote wins.

Feature	Importance	What It Reveals
RR Interval	31.6%	Timing between beats — rhythm regularity
ST Deviation	29.8%	Cardiac stress — ischemia indicator
Heart Rate	24.4%	Speed of heart — rate abnormality
RR Variability	14.2%	Beat to beat irregularity
PR Interval	0%	Conduction delay — fixed in most records
QRS Duration	0%	Beat width — fixed window in this implementation

5.2 Model 2 — CNN Deep Learning (Waveform Based)
A 650ms raw ECG window (234 samples) is cut around each R peak and fed directly into a convolutional neural network. The CNN learns to recognize the shapes of normal, atrial and ventricular beats without any manual feature engineering.

CNN Architecture:
Input (234 samples, 1 channel)
  -> Conv1D(32 filters) + BatchNorm + MaxPool + Dropout(0.2)
  -> Conv1D(64 filters) + BatchNorm + MaxPool + Dropout(0.2)
  -> Conv1D(128 filters) + BatchNorm + MaxPool + Dropout(0.3)
  -> Flatten -> Dense(128) + BatchNorm + Dropout(0.4)
  -> Dense(64) + Dropout(0.3)
  -> Dense(3) Softmax -> N / A / V
Epochs: 36 | Batch: 32 | Optimizer: Adam | Loss: Categorical Crossentropy

5.3 Combined Decision Making
Both models predict every beat independently. Results are averaged. If either abnormal beat type exceeds 5% of total beats the system raises an arrhythmia verdict. High agreement between models indicates a reliable result.

6. System Pipeline
Step	File	Purpose	Output
1	step1_preprocessing.py	Load .dat file and apply bandpass filter 0.5 to 40 Hz	Filtered ECG signal
2	step2_segmentation.py	Pan-Tompkins R-peak detection and wave segmentation	P QRS T wave boundaries
3	step3_features.py	Extract 6 features per beat from 8 training records	feature_table.csv 14902 beats
4	step4_classification.py	Train SVM and Random Forest on balanced dataset	best_model.pkl 85% accuracy
4b	step4b_cnn.py	Train CNN on raw ECG segments 234 samples each	cnn_model.keras 97.38% accuracy
5	step5_visualization.py	Classify beats and generate full clinical dashboard	Dashboard image and report

7. Dataset
MIT-BIH Arrhythmia Database from PhysioNet — 48 half-hour two-lead ECG recordings at 360 Hz with expert cardiologist beat annotations.
Download: https://physionet.org/content/mitdb/1.0.0/

Record	Beats	Normal	Atrial	Ventricular	Notes
100	2270	2236	33	1	Normal baseline
101	1858	1855	3	0	Normal rhythm
105	1285	1251	0	34	Arrhythmia variety
200	1595	852	9	734	Heavy PVC
208	1581	1268	311	2	Atrial rich
209	2124	1864	260	0	Atrial rich
213	2439	2162	229	48	Atrial rich
222	1750	1569	181	0	Atrial rich
Total	14902	13057	1026	819	8 records combined

8. Models and Results
Model	Test Accuracy	CV Accuracy	Input Type
SVM	83.33%	68.10%	6 clinical features
Random Forest	85.00%	87.14%	6 clinical features
CNN (Best)	97.38%	97.38%	234 raw ECG samples

8.1 CNN Per Class Performance
Beat Type	Precision	Recall	F1 Score	Support
Normal (N)	97%	95%	96%	140
Atrial (A)	97%	98%	98%	140
Ventricular (V)	98%	99%	99%	140

8.2 Why CNN Outperforms Random Forest
The CNN learns directly from raw waveform shapes — it can detect the wide bizarre QRS of ventricular beats and the abnormal P wave morphology of atrial beats without any hand-crafted features. The Random Forest is stronger on atrial beats because timing features like RR interval and RR variability directly capture the premature nature of atrial contractions.

9. Tech Stack
Category	Technology	Purpose
Signal Processing	SciPy, NumPy	Filtering, feature math, R-peak detection
ECG Data	WFDB	Loading MIT-BIH .dat .hea .atr files
Machine Learning	Scikit-learn	Random Forest and SVM training
Deep Learning	TensorFlow 2.17, Keras	CNN model definition and training
Web Backend	Flask, Gunicorn	REST API server and request handling
Frontend	HTML, CSS, JavaScript	Upload form and results display
Charts	Chart.js	Interactive beat distribution charts
Container	Docker	Consistent deployment environment
Hosting	Hugging Face Spaces	Free cloud deployment with 16GB RAM
Version Control	Git, GitHub	Code management and collaboration

10. How to Run Locally
Prerequisites: Python 3.11, 4GB RAM minimum, MIT-BIH dataset downloaded.

git clone https://github.com/ranjithtkm445-blip/ecg-arrhythmia-detection
cd ecg-arrhythmia-detection
python -m venv ecg_env
ecg_env\Scripts\activate
pip install -r requirements.txt
python app.py

Open http://localhost:7860 in your browser and upload ECG files.

11. Understanding the Output
11.1 Verdict Banner
Shown at the top of the page in large bold colored text immediately after analysis:
Color	Verdict	Meaning	Triggered When
Green	NO ARRHYTHMIA DETECTED	Normal rhythm	Less than 5% abnormal beats
Orange	ATRIAL ARRHYTHMIA DETECTED	Irregular atrial beats	Atrial beats exceed 5%
Red	VENTRICULAR ARRHYTHMIA DETECTED	Dangerous ventricular beats	Ventricular beats exceed 5%
Red	MULTIPLE ARRHYTHMIAS DETECTED	Both types found	Both types exceed 5%

11.2 Info Bar
Field	Meaning
Record	ECG file name that was analyzed
Duration	Recording length in minutes
Total Beats	Number of heartbeats analyzed
RF Accuracy	Random Forest accuracy vs expert ground truth
CNN Accuracy	CNN accuracy vs expert ground truth
Agreement	Percentage of beats where RF and CNN agreed

11.3 ECG Waveform Plot
Shows the first 10 seconds of the filtered ECG signal color coded by beat type. Green dots are normal beats, orange dots are atrial beats, red dots are ventricular beats. The shaded region around each dot shows the QRS complex boundary. Abnormal beats are immediately visible at irregular positions.

11.4 Beat Distribution Charts
Two pie charts side by side — one from Random Forest and one from CNN. A third chart shows model agreement percentage. High agreement above 80% means the result is reliable. Low agreement suggests the case is borderline and manual review is recommended.

11.5 Heart Rate Over Time
Scatter plot of heart rate at every beat across the full recording. Yellow dashed line marks 60 BPM lower limit and red marks 100 BPM upper limit. Clusters of red dots above 100 BPM indicate tachycardia episodes. Dots below 60 BPM indicate bradycardia.

11.6 Clinical Summary
Metric	Normal Range	Status Labels
Heart Rate	60 to 100 BPM	NORMAL / BRADYCARDIA / TACHYCARDIA
ST Deviation	-0.1 to 0.1 mV	NORMAL / ABNORMAL
RF Risk	Based on beat %	LOW RISK / MODERATE / HIGH RISK
CNN Risk	Based on beat %	LOW RISK / MODERATE / HIGH RISK

11.7 Clinical Explanation Panel
Section	Content
Why Detected	Exact beat counts from each model that triggered the verdict
Features Triggered	Which clinical measurements were outside normal range
Beat Breakdown	Cards showing N A V counts from both models with progress bars
Clinical Interpretation	Plain English explanation of what the findings mean medically
Next Steps	Specific recommended actions based on the risk level

11.8 Recommended Next Steps by Risk Level
Risk Level	Recommended Action
HIGH RISK	Consult cardiologist immediately, perform 12-lead ECG, consider hospital evaluation, discuss antiarrhythmic medication
MODERATE	Schedule cardiologist appointment, consider Holter monitor, keep symptom diary, reduce caffeine and stress
LOW RISK	Routine annual cardiac checkup, maintain healthy lifestyle, monitor for new symptoms

12. Disclaimer
This project was built for research and educational purposes only. The predictions are not medically certified and should not be used for clinical diagnosis. The AI models may not be 100% accurate for all ECG types. Always consult a qualified cardiologist for any cardiac concerns.

Final year project — Biomedical Signal Processing and Machine Learning applied to cardiac arrhythmia detection.
