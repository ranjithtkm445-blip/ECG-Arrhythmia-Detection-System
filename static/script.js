// script.js
// Purpose: Frontend logic for ECG Arrhythmia Detection Web App
// Updated: Added full explanation panel rendering
// Handles: file upload, API call, all charts, clinical summary, explanation

// ================================================
// SECTION 1: Global Variables
// ================================================

let selectedFiles = {};
let rfChart       = null;
let cnnChart      = null;
let agreeChart    = null;
let hrChart       = null;
let accChart      = null;

// ================================================
// SECTION 2: File Upload Handling
// ================================================

document.getElementById('file-input').addEventListener('change', function(e) {
  handleFiles(e.target.files);
});

// Drag and drop
const dropZone = document.getElementById('drop-zone');

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
  selectedFiles = {};

  Array.from(files).forEach(file => {
    const ext = file.name.split('.').pop().toLowerCase();
    if (['dat', 'hea', 'atr'].includes(ext)) {
      selectedFiles[ext] = file;
    }
  });

  renderFileList();
  updateAnalyzeButton();
}

function renderFileList() {
  const list = document.getElementById('file-list');
  list.innerHTML = '';

  const required = ['dat', 'hea', 'atr'];
  const labels   = {
    dat: '📊 Signal Data (.dat)',
    hea: '📋 Header Info (.hea)',
    atr: '🏷️ Annotations (.atr)'
  };

  required.forEach(ext => {
    const file = selectedFiles[ext];
    const item = document.createElement('div');
    item.className = 'file-item';

    if (file) {
      const size = (file.size / 1024).toFixed(1);
      item.innerHTML = `
        <span class="file-icon">
          ${labels[ext].split(' ')[0]}
        </span>
        <span class="file-name">${file.name}</span>
        <span class="file-size">${size} KB</span>
        <span class="file-check">✅</span>
      `;
    } else {
      item.style.opacity = '0.4';
      item.innerHTML = `
        <span class="file-icon">📄</span>
        <span class="file-name">
          ${labels[ext]} — not selected
          ${ext === 'atr' ? '(optional)' : '(required)'}
        </span>
        <span class="file-size"></span>
        <span class="file-check">⬜</span>
      `;
    }

    list.appendChild(item);
  });
}

function updateAnalyzeButton() {
  const btn    = document.getElementById('btn-analyze');
  btn.disabled = !(selectedFiles['dat'] && selectedFiles['hea']);
}

// ================================================
// SECTION 3: Analyze ECG
// ================================================

async function analyzeECG() {

  if (!selectedFiles['dat'] || !selectedFiles['hea']) {
    alert('Please select at least .dat and .hea files');
    return;
  }

  showSection('loading');
  animateLoadingSteps();

  try {
    const formData = new FormData();
    formData.append('dat_file', selectedFiles['dat']);
    formData.append('hea_file', selectedFiles['hea']);

    if (selectedFiles['atr']) {
      formData.append('atr_file', selectedFiles['atr']);
    }

    const response = await fetch('/analyze', {
      method: 'POST',
      body  : formData
    });

    const data = await response.json();

    if (data.error) {
      alert('Error: ' + data.error);
      showSection('upload');
      return;
    }

    renderResults(data);
    showSection('results');

  } catch (err) {
    alert('Connection error: ' + err.message);
    showSection('upload');
  }
}

// ================================================
// SECTION 4: Loading Animation
// ================================================

function animateLoadingSteps() {
  const steps  = ['step-1','step-2','step-3','step-4','step-5','step-6'];
  const labels = [
    '✅ Filtering signal...',
    '✅ Detecting R-peaks...',
    '✅ Extracting features...',
    '✅ Running Random Forest...',
    '✅ Running CNN...',
    '✅ Generating report...'
  ];

  // Reset all steps first
  steps.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'loading-step';
  });

  steps.forEach((id, i) => {
    setTimeout(() => {
      if (i > 0) {
        const prev = document.getElementById(steps[i-1]);
        prev.className   = 'loading-step done';
        prev.textContent = labels[i-1];
      }
      const curr = document.getElementById(id);
      curr.className = 'loading-step active';
    }, i * 800);
  });
}

// ================================================
// SECTION 5: Render All Results
// ================================================

function renderResults(data) {

  // Verdict
  renderVerdict(data.verdict);

  // Info bar
  document.getElementById('info-record').textContent =
    `Record ${data.record_name}`;
  document.getElementById('info-duration').textContent =
    `${data.duration_min} min`;
  document.getElementById('info-beats').textContent =
    data.total_beats.toLocaleString();
  document.getElementById('info-rf-acc').textContent =
    data.rf_acc !== null ? `${data.rf_acc}%` : 'N/A';
  document.getElementById('info-cnn-acc').textContent =
    data.cnn_acc !== null ? `${data.cnn_acc}%` : 'N/A';
  document.getElementById('info-agreement').textContent =
    `${data.agree_pct}%`;

  // ECG waveform
  document.getElementById('ecg-plot').src =
    'data:image/png;base64,' + data.ecg_plot;

  // Charts
  renderBeatChart(
    'rf-chart', data.rf_counts, rfChart,
    c => rfChart = c
  );
  renderBeatChart(
    'cnn-chart', data.cnn_counts, cnnChart,
    c => cnnChart = c
  );
  renderAgreeChart(data.agree_pct);
  renderHRChart(
    data.time_minutes, data.hr_values, data.hr_labels
  );
  renderAccChart(data.rf_acc, data.cnn_acc);

  // Clinical summary
  renderClinicalSummary(data);

  // Per class table
  renderPerClassTable(data.per_class);

  // Explanation panel
  renderExplanation(data.explanation);
}

// ================================================
// SECTION 6: Verdict Banner
// ================================================

function renderVerdict(verdict) {
  const banner = document.getElementById('verdict-banner');
  const icon   = document.getElementById('verdict-icon');
  const title  = document.getElementById('verdict-title');
  const detail = document.getElementById('verdict-detail');

  banner.classList.remove('success', 'warning', 'danger');
  banner.classList.add(verdict.type);

  icon.textContent   = verdict.icon;
  title.textContent  = verdict.verdict;
  detail.textContent = verdict.detail;
}

// ================================================
// SECTION 7: Beat Distribution Charts
// ================================================

function renderBeatChart(canvasId, counts, existingChart, setChart) {

  const labels = Object.keys(counts);
  const values = Object.values(counts);
  const total  = values.reduce((a, b) => a + b, 0);

  const colorMap = {
    N: '#2ecc71',
    A: '#e67e22',
    V: '#e74c3c'
  };
  const colors = labels.map(l => colorMap[l] || '#888888');

  if (existingChart) existingChart.destroy();

  const ctx   = document.getElementById(canvasId).getContext('2d');
  const chart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels  : labels.map(l => {
        const names = {
          N: 'Normal',
          A: 'Atrial',
          V: 'Ventricular'
        };
        return `${names[l] || l}: ${counts[l]}`;
      }),
      datasets: [{
        data           : values,
        backgroundColor: colors,
        borderColor    : '#1a1a2e',
        borderWidth    : 3,
        hoverOffset    : 8
      }]
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels  : {
            color   : '#aaaaaa',
            font    : { size: 11 },
            padding : 12,
            boxWidth: 12,
          }
        },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const pct = ((ctx.parsed / total) * 100).toFixed(1);
              return ` ${ctx.parsed} beats (${pct}%)`;
            }
          }
        }
      }
    }
  });

  setChart(chart);
}

// ================================================
// SECTION 8: Agreement Chart
// ================================================

function renderAgreeChart(agreePct) {
  if (agreeChart) agreeChart.destroy();

  const ctx = document.getElementById('agree-chart').getContext('2d');
  agreeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: [
        `Agree: ${agreePct}%`,
        `Disagree: ${(100 - agreePct).toFixed(1)}%`
      ],
      datasets: [{
        data           : [agreePct, 100 - agreePct],
        backgroundColor: ['#2ecc71', '#e74c3c'],
        borderColor    : '#1a1a2e',
        borderWidth    : 3,
        hoverOffset    : 8
      }]
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels  : {
            color   : '#aaaaaa',
            font    : { size: 11 },
            padding : 12,
            boxWidth: 12,
          }
        }
      }
    }
  });
}

// ================================================
// SECTION 9: Heart Rate Chart
// ================================================

function renderHRChart(timeMinutes, hrValues, hrLabels) {
  if (hrChart) hrChart.destroy();

  const colorMap = { N: '#2ecc71', A: '#e67e22', V: '#e74c3c' };
  const names    = { N: 'Normal',  A: 'Atrial',  V: 'Ventricular' };
  const classes  = ['N', 'A', 'V'];

  const datasets = classes.map(cls => {
    const points = timeMinutes
      .map((t, i) => hrLabels[i] === cls
        ? { x: t, y: hrValues[i] }
        : null)
      .filter(Boolean);

    return {
      label           : names[cls],
      data            : points,
      backgroundColor : colorMap[cls],
      pointRadius     : 2,
      pointHoverRadius: 5,
    };
  });

  // Threshold lines
  const minT = Math.min(...timeMinutes);
  const maxT = Math.max(...timeMinutes);

  datasets.push({
    label      : '60 BPM',
    data       : [{ x: minT, y: 60 }, { x: maxT, y: 60 }],
    type       : 'line',
    borderColor: '#f1c40f',
    borderWidth: 1,
    borderDash : [5, 5],
    pointRadius: 0,
    fill       : false,
  });

  datasets.push({
    label      : '100 BPM',
    data       : [{ x: minT, y: 100 }, { x: maxT, y: 100 }],
    type       : 'line',
    borderColor: '#e74c3c',
    borderWidth: 1,
    borderDash : [5, 5],
    pointRadius: 0,
    fill       : false,
  });

  const ctx = document.getElementById('hr-chart').getContext('2d');
  hrChart   = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text   : 'Time (minutes)',
            color  : '#aaaaaa'
          },
          ticks: { color: '#aaaaaa' },
          grid : { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          title: {
            display: true,
            text   : 'Heart Rate (BPM)',
            color  : '#aaaaaa'
          },
          ticks: { color: '#aaaaaa' },
          grid : { color: 'rgba(255,255,255,0.05)' }
        }
      },
      plugins: {
        legend: {
          labels: {
            color   : '#aaaaaa',
            font    : { size: 11 },
            boxWidth: 12,
          }
        }
      }
    }
  });
}

// ================================================
// SECTION 10: Model Accuracy Comparison Chart
// ================================================

function renderAccChart(rfAcc, cnnAcc) {
  if (accChart) accChart.destroy();

  const ctx = document.getElementById('acc-chart').getContext('2d');
  accChart  = new Chart(ctx, {
    type: 'bar',
    data: {
      labels  : ['SVM', 'Random Forest', 'CNN'],
      datasets: [{
        label          : 'Test Accuracy (%)',
        data           : [83.33, rfAcc || 85.0, cnnAcc || 97.38],
        backgroundColor: ['#3498db', '#2ecc71', '#9b59b6'],
        borderColor    : ['#2980b9', '#27ae60', '#8e44ad'],
        borderWidth    : 1,
        borderRadius   : 6,
      }]
    },
    options: {
      responsive         : true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min  : 60,
          max  : 105,
          ticks: { color: '#aaaaaa' },
          grid : { color: 'rgba(255,255,255,0.05)' }
        },
        x: {
          ticks: { color: '#aaaaaa' },
          grid : { color: 'transparent' }
        }
      },
      plugins: {
        legend : { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ` ${ctx.parsed.y.toFixed(1)}%`
          }
        }
      }
    }
  });
}

// ================================================
// SECTION 11: Clinical Summary
// ================================================

function renderClinicalSummary(data) {

  document.getElementById('sum-avg-hr').textContent =
    `${data.avg_hr} BPM`;
  document.getElementById('sum-min-hr').textContent =
    `${data.min_hr} BPM`;
  document.getElementById('sum-max-hr').textContent =
    `${data.max_hr} BPM`;
  document.getElementById('sum-avg-rr').textContent =
    `${data.avg_rr} ms`;
  document.getElementById('sum-avg-st').textContent =
    `${data.avg_st} mV`;

  const hrStatus     = document.getElementById('sum-hr-status');
  hrStatus.textContent = data.hr_status;
  hrStatus.className   = 'summary-status ' + (
    data.hr_status === 'NORMAL'
      ? 'status-normal'
      : 'status-abnormal'
  );

  const stStatus     = document.getElementById('sum-st-status');
  stStatus.textContent = data.st_status;
  stStatus.className   = 'summary-status ' + (
    data.st_status === 'NORMAL'
      ? 'status-normal'
      : 'status-abnormal'
  );

  const rfRisk     = document.getElementById('sum-rf-risk');
  rfRisk.textContent = data.rf_risk;
  rfRisk.className   = 'summary-value ' + getRiskClass(data.rf_risk);

  const cnnRisk    = document.getElementById('sum-cnn-risk');
  cnnRisk.textContent = data.cnn_risk;
  cnnRisk.className   = 'summary-value ' + getRiskClass(data.cnn_risk);
}

function getRiskClass(risk) {
  if (risk === 'LOW RISK')  return 'status-risk-low';
  if (risk === 'MODERATE')  return 'status-risk-moderate';
  if (risk === 'HIGH RISK') return 'status-risk-high';
  return '';
}

// ================================================
// SECTION 12: Per Class Accuracy Table
// ================================================

function renderPerClassTable(perClass) {
  const container = document.getElementById('per-class-table');
  container.innerHTML = '';

  if (!perClass || Object.keys(perClass).length === 0) {
    container.innerHTML =
      '<p style="color:#666;font-size:0.85rem;margin-top:8px">' +
      'No ground truth available for accuracy calculation</p>';
    return;
  }

  const colorMap = { N: '#2ecc71', A: '#e67e22', V: '#e74c3c' };
  const nameMap  = { N: 'Normal',  A: 'Atrial',  V: 'Ventricular' };

  Object.entries(perClass).forEach(([label, info]) => {
    const row = document.createElement('div');
    row.className = 'per-class-row';
    row.innerHTML = `
      <div class="per-class-label">
        <div class="per-class-dot"
             style="background:${colorMap[label]}">
        </div>
        ${nameMap[label]}
      </div>
      <div class="per-class-bars">
        <div class="per-class-bar-row">
          <span class="per-class-bar-label">RF</span>
          <div class="per-class-bar-bg">
            <div class="per-class-bar-fill"
                 style="width:${info.rf_acc}%;
                        background:${colorMap[label]}">
            </div>
          </div>
          <span class="per-class-bar-val">${info.rf_acc}%</span>
        </div>
        <div class="per-class-bar-row">
          <span class="per-class-bar-label">CNN</span>
          <div class="per-class-bar-bg">
            <div class="per-class-bar-fill"
                 style="width:${info.cnn_acc}%;
                        background:${colorMap[label]}80">
            </div>
          </div>
          <span class="per-class-bar-val">${info.cnn_acc}%</span>
        </div>
      </div>
      <span class="per-class-count">${info.count} beats</span>
    `;
    container.appendChild(row);
  });
}

// ================================================
// SECTION 13: Explanation Panel
// ================================================

function renderExplanation(exp) {

  // --- Why detected ---
  const whyList = document.getElementById('exp-why');
  whyList.innerHTML = '';
  exp.why.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    whyList.appendChild(li);
  });

  // --- Features triggered ---
  const featList = document.getElementById('exp-features');
  featList.innerHTML = '';
  exp.features_triggered.forEach(item => {
    const li = document.createElement('li');
    li.textContent = item;
    featList.appendChild(li);
  });

  // --- Beat breakdown cards ---
  const breakdown = document.getElementById('exp-breakdown');
  breakdown.innerHTML = '<div class="breakdown-grid"></div>';
  const grid = breakdown.querySelector('.breakdown-grid');

  const colorMap = {
    green : '#2ecc71',
    orange: '#e67e22',
    red   : '#e74c3c'
  };

  exp.breakdown.forEach(beat => {
    const card      = document.createElement('div');
    card.className  = `breakdown-card ${beat.color}`;
    const fillColor = colorMap[beat.color] || '#888';

    card.innerHTML = `
      <div class="breakdown-type">${beat.type}</div>
      <div class="breakdown-pct">${beat.pct}%</div>
      <div class="breakdown-counts">
        <span>RF: ${beat.rf} beats</span>
        <span>CNN: ${beat.cnn} beats</span>
      </div>
      <div class="breakdown-bar-bg">
        <div class="breakdown-bar-fill"
             style="width:${Math.min(beat.pct, 100)}%;
                    background:${fillColor}">
        </div>
      </div>
      <div class="breakdown-meaning">${beat.meaning}</div>
    `;
    grid.appendChild(card);
  });

  // --- Clinical interpretation ---
  const clinical = document.getElementById('exp-clinical');
  clinical.innerHTML = '';
  exp.clinical.forEach(item => {
    const div       = document.createElement('div');
    div.className   = 'clinical-item';
    div.textContent = item;
    clinical.appendChild(div);
  });

  // --- Next steps ---
  const steps = document.getElementById('exp-steps');
  steps.innerHTML = '';
  exp.next_steps.forEach(item => {
    const li       = document.createElement('li');
    li.textContent = item;
    steps.appendChild(li);
  });
}

// ================================================
// SECTION 14: Section Switching
// ================================================

function showSection(section) {
  document.getElementById('upload-section').classList.add('hidden');
  document.getElementById('loading-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');

  if (section === 'upload') {
    document.getElementById('upload-section')
      .classList.remove('hidden');
  } else if (section === 'loading') {
    document.getElementById('loading-section')
      .classList.remove('hidden');
  } else if (section === 'results') {
    document.getElementById('results-section')
      .classList.remove('hidden');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

// ================================================
// SECTION 15: Reset App
// ================================================

function resetApp() {
  selectedFiles = {};
  document.getElementById('file-input').value    = '';
  document.getElementById('file-list').innerHTML = '';
  document.getElementById('btn-analyze').disabled = true;

  // Destroy all charts
  [rfChart, cnnChart, agreeChart, hrChart, accChart].forEach(c => {
    if (c) c.destroy();
  });
  rfChart = cnnChart = agreeChart = hrChart = accChart = null;

  showSection('upload');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}