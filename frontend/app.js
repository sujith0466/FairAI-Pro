/* ═══════════════════════════════════════════════════════════
   FairAI Pro — Frontend Application Logic
   ═══════════════════════════════════════════════════════════ */

const BACKEND_BASE_URL =
(window.location.hostname === "localhost" ||
window.location.hostname === "127.0.0.1" ||
window.location.hostname === "")
? "http://localhost:5000"
: "https://fairai-pro.onrender.com";

const API_BASE = BACKEND_BASE_URL + "/api";
const API_TIMEOUT_MS = 60000;
const DEFAULT_LOADING_MESSAGE = 'Processing fairness analysis... This may take a few seconds';
const ANALYZE_LOADING_MESSAGE = 'Analyzing dataset...';
const MITIGATION_LOADING_MESSAGE = 'Running mitigation...';
const EXPLAIN_LOADING_MESSAGE = 'Generating explanation...';
const EXPORT_LOADING_MESSAGE = 'Exporting report...';

// ── State ─────────────────────────────────────────────────
let datasetInfo = null;

// ── DOM References ────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const uploadPanel = $('#upload-panel');
const configPanel = $('#config-panel');
const loadingOverlay = $('#loading-overlay');
const resultsDashboard = $('#results-dashboard');
const uploadZone = $('#upload-zone');
const fileInput = $('#file-input');
const btnSample = $('#btn-sample');
const btnAnalyze = $('#btn-analyze');
const btnChangeFile = $('#btn-change-file');
const btnNewAnalysis = $('#btn-new-analysis');
const btnExportReport = $('#btn-export-report');
const btnExplain = $('#btn-explain');
const btnMitigation = $('#btn-mitigation');
const targetColSel = $('#target-col');
const sensitiveColSel = $('#sensitive-col');
const privilegedValSel = $('#privileged-val');
let apiInFlightCount = 0;

// ── Init ──────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initUpload();
    initButtons();
});

// ── Navbar ────────────────────────────────────────────────
function initNavbar() {
    const navbar = $('#navbar');
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 40);
    });

    // Active nav link
    const sections = $$('section[id]');
    const navLinks = $$('.nav-link');
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(s => {
            if (window.scrollY >= s.offsetTop - 200) {
                current = s.id;
            }
        });
        navLinks.forEach(l => {
            l.classList.toggle('active', l.getAttribute('href') === `#${current}`);
        });
    });
}

// ── Upload Logic ──────────────────────────────────────────
function initUpload() {
    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    });

    // Drag & drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) uploadFile(e.dataTransfer.files[0]);
    });
}

function initButtons() {
    btnSample.addEventListener('click', loadSampleDataset);
    btnAnalyze.addEventListener('click', runAnalysis);
    btnChangeFile.addEventListener('click', resetToUpload);
    btnNewAnalysis.addEventListener('click', resetToUpload);

    // Export report
    if (btnExportReport) btnExportReport.addEventListener('click', exportReport);
    if (btnExplain) btnExplain.addEventListener('click', runExplain);
    if (btnMitigation) btnMitigation.addEventListener('click', runMitigation);

    // Update privileged values when sensitive column changes
    sensitiveColSel.addEventListener('change', () => {
        updatePrivilegedValues();
    });
}

// ── API Calls ─────────────────────────────────────────────
function setApiLoading(isLoading) {
    if (isLoading) apiInFlightCount += 1;
    else apiInFlightCount = Math.max(0, apiInFlightCount - 1);

    const isBusy = apiInFlightCount > 0;
    if (btnAnalyze) btnAnalyze.disabled = isBusy;
    if (btnSample) btnSample.disabled = isBusy;
    if (fileInput) fileInput.disabled = isBusy;
    if (btnExportReport) btnExportReport.disabled = isBusy;
    if (btnExplain) btnExplain.disabled = isBusy;
    if (btnMitigation) btnMitigation.disabled = isBusy;
    if (btnChangeFile) btnChangeFile.disabled = isBusy;
    if (btnNewAnalysis) btnNewAnalysis.disabled = isBusy;
}

async function parseJsonSafe(response) {
    try {
        return await response.json();
    } catch (e) {
        return null;
    }
}

async function apiRequest(path, options = {}, showInlineLoading = false, loadingMessage = DEFAULT_LOADING_MESSAGE, manageBusyState = true) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
    const mergedOptions = { ...options, signal: controller.signal };

    if (manageBusyState) setApiLoading(true);
    if (showInlineLoading) showLoading(loadingMessage);

    try {
        const response = await fetch(`${API_BASE}${path}`, mergedOptions);
        const data = await parseJsonSafe(response);

        if (!response.ok) {
            const message = data?.error || `Request failed (${response.status})`;
            throw new Error(message);
        }
        return data || {};
    } catch (err) {
        if (err.name === 'AbortError') {
            throw new Error('Request timed out. Please try again.');
        }
        if (err instanceof TypeError) {
            throw new Error('Cannot reach backend service. Please verify deployment URL and server status.');
        }
        throw err;
    } finally {
        clearTimeout(timeoutId);
        if (manageBusyState) setApiLoading(false);
        if (showInlineLoading) hideLoading();
    }
}

async function uploadFile(file) {
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('dataset', file);

    try {
        showToast('Uploading dataset...', 'info');
        const data = await apiRequest('/upload', { method: 'POST', body: formData });

        datasetInfo = data;
        showConfigPanel(data);
        showToast(`Loaded: ${data.filename} (${data.rows} rows)`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function loadSampleDataset() {
    try {
        showToast('Loading sample dataset...', 'info');
        const data = await apiRequest('/sample', { method: 'POST' });

        datasetInfo = data;
        showConfigPanel(data);
        showToast(`Loaded sample: ${data.filename} (${data.rows} rows)`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function runAnalysis() {
    const targetCol = targetColSel.value;
    const sensitiveCol = sensitiveColSel.value;
    const privilegedVal = privilegedValSel.value;

    if (!targetCol || !sensitiveCol || !privilegedVal) {
        showToast('Please fill in all configuration fields', 'error');
        return;
    }

    showLoading(ANALYZE_LOADING_MESSAGE);
    setApiLoading(true);
    try {
        const data = await apiRequest('/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_column: targetCol,
                sensitive_column: sensitiveCol,
                privileged_value: privilegedVal
            })
        }, false, DEFAULT_LOADING_MESSAGE, false);
        data.mitigation = null;
        renderResults(data);
        showToast('Analysis complete! Run mitigation to compare fairness before and after.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoading();
        setApiLoading(false);
    }
}

async function runMitigation() {
    const targetCol = targetColSel.value;
    const sensitiveCol = sensitiveColSel.value;

    if (!targetCol || !sensitiveCol) {
        showToast('Please select target and sensitive columns before mitigation.', 'error');
        return;
    }

    showLoading(MITIGATION_LOADING_MESSAGE);
    setApiLoading(true);
    try {
        const mitigation = await apiRequest('/mitigation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                target_column: targetCol,
                sensitive_column: sensitiveCol
            })
        }, false, DEFAULT_LOADING_MESSAGE, false);

        if (!window._lastAnalysisData) window._lastAnalysisData = {};
        window._lastAnalysisData.mitigation = mitigation;
        renderMitigationScores(mitigation);
        showToast('Mitigation completed successfully.', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoading();
        setApiLoading(false);
        if (window._lastAnalysisData) {
            resultsDashboard.classList.remove('hidden');
        }
    }
}

// ── Config Panel ──────────────────────────────────────────
function showConfigPanel(info) {
    uploadPanel.classList.add('hidden');
    configPanel.classList.remove('hidden');

    // File info
    $('#config-filename').textContent = info.filename;
    $('#config-rows').textContent = `${info.rows} rows · ${info.columns.length} columns`;

    // Data Preview table
    renderDataPreview(info.preview, info.columns);

    // Populate dropdowns
    populateSelect(targetColSel, info.columns.map(c => c.name), 'Select target column...');
    populateSelect(sensitiveColSel, info.columns.map(c => c.name), 'Select sensitive column...');
    privilegedValSel.innerHTML = '<option value="">Select sensitive column first...</option>';

    // Auto-detect common column names
    const colNames = info.columns.map(c => c.name.toLowerCase());

    // Auto-select target
    const targetGuess = info.columns.find(c =>
        ['hired', 'target', 'label', 'outcome', 'class', 'income', 'default', 'approved'].includes(c.name.toLowerCase())
    );
    if (targetGuess) targetColSel.value = targetGuess.name;

    // Auto-select sensitive
    const sensitiveGuess = info.columns.find(c =>
        ['gender', 'sex', 'race', 'ethnicity', 'age_group', 'religion'].includes(c.name.toLowerCase())
    );
    if (sensitiveGuess) {
        sensitiveColSel.value = sensitiveGuess.name;
        updatePrivilegedValues();
    }
}

function renderDataPreview(preview, columns) {
    if (!preview || preview.length === 0) return;

    const cols = columns.map(c => c.name);
    let html = '<table><thead><tr>';
    cols.forEach(c => html += `<th>${c}</th>`);
    html += '</tr></thead><tbody>';

    preview.forEach(row => {
        html += '<tr>';
        cols.forEach(c => html += `<td>${row[c] ?? ''}</td>`);
        html += '</tr>';
    });

    html += '</tbody></table>';
    $('#data-preview').innerHTML = html;
}

function populateSelect(select, options, placeholder) {
    select.innerHTML = `<option value="">${placeholder}</option>`;
    options.forEach(opt => {
        select.innerHTML += `<option value="${opt}">${opt}</option>`;
    });
}

function updatePrivilegedValues() {
    const col = sensitiveColSel.value;
    if (!col || !datasetInfo) return;

    const colInfo = datasetInfo.columns.find(c => c.name === col);
    if (!colInfo) return;

    populateSelect(privilegedValSel, colInfo.sample_values.map(String), 'Select privileged value...');

    // Auto-select common privileged values
    const vals = colInfo.sample_values.map(v => String(v).toLowerCase());
    const privGuess = colInfo.sample_values.find(v =>
        ['male', '1', 'white', 'yes', 'true'].includes(String(v).toLowerCase())
    );
    if (privGuess) privilegedValSel.value = String(privGuess);
}

// ── Loading ───────────────────────────────────────────────
function showLoading(message = DEFAULT_LOADING_MESSAGE) {
    configPanel.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    loadingOverlay.classList.remove('hidden');

    const loadingTitle = $('#loading-title');
    const loadingText = $('#loading-text');
    if (loadingTitle) loadingTitle.textContent = message;
    if (loadingText) loadingText.textContent = message;

    // Animate steps
    const steps = ['ls-1', 'ls-2', 'ls-3', 'ls-4'];
    steps.forEach(s => {
        $(`#${s}`).className = 'loading-step';
    });
    $(`#${steps[0]}`).classList.add('active');

    let i = 1;
    const interval = setInterval(() => {
        if (i >= steps.length) {
            clearInterval(interval);
            return;
        }
        $(`#${steps[i - 1]}`).classList.remove('active');
        $(`#${steps[i - 1]}`).classList.add('done');
        $(`#${steps[i]}`).classList.add('active');
        i++;
    }, 800);

    loadingOverlay._interval = interval;
}

function hideLoading() {
    if (loadingOverlay._interval) clearInterval(loadingOverlay._interval);
    loadingOverlay.classList.add('hidden');
}

// ── Render Results ────────────────────────────────────────
function renderResults(data) {
    if (!data || !data.fairness || !data.model || !data.groups || !data.dataset) {
        showToast('Incomplete analysis response received from backend.', 'error');
        return;
    }

    resultsDashboard.classList.remove('hidden');

    const { fairness, model, groups, dataset } = data;

    // Determine color theme
    const scoreMeta = getBiasLevelMeta(fairness.fairness_score);
    const c = scoreMeta.color;

    // ── Score Ring ──
    const scoreHero = $('#score-hero');
    scoreHero.className = `score-hero score-${c}`;

    // Add SVG gradient dynamically
    const gradColors = {
        green: ['#00E676', '#00D2FF'],
        yellow: ['#FFD93D', '#FF9F43'],
        red: ['#FF6B6B', '#FD79A8']
    };

    const svgNS = 'http://www.w3.org/2000/svg';
    let scoreSvg = scoreHero.querySelector('.score-ring');

    // Remove old defs
    let oldDefs = scoreSvg.querySelector('defs');
    if (oldDefs) oldDefs.remove();

    let defs = document.createElementNS(svgNS, 'defs');
    let grad = document.createElementNS(svgNS, 'linearGradient');
    grad.setAttribute('id', 'score-gradient');
    grad.setAttribute('x1', '0%');
    grad.setAttribute('y1', '0%');
    grad.setAttribute('x2', '100%');
    grad.setAttribute('y2', '100%');
    let stop1 = document.createElementNS(svgNS, 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', gradColors[c][0]);
    let stop2 = document.createElementNS(svgNS, 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', gradColors[c][1]);
    grad.appendChild(stop1);
    grad.appendChild(stop2);
    defs.appendChild(grad);
    scoreSvg.insertBefore(defs, scoreSvg.firstChild);

    // Animate ring
    const circumference = 2 * Math.PI * 85; // ~534
    const offset = circumference - (fairness.fairness_score / 100) * circumference;
    const ringFill = $('#score-ring-fill');
    ringFill.style.strokeDasharray = circumference;
    ringFill.style.strokeDashoffset = circumference;
    setTimeout(() => {
        ringFill.style.strokeDashoffset = offset;
    }, 100);

    // Animate counter
    animateCounter($('#score-value'), 0, fairness.fairness_score, 1500);

    // Badge
    const badge = $('#score-badge');
    badge.textContent = scoreMeta.label;
    badge.className = `score-badge badge-${c}`;

    // Headline
    const headlines = {
        green: 'Your model demonstrates fair treatment across groups',
        yellow: 'Moderate bias detected — review recommended',
        red: 'Significant bias detected — mitigation required'
    };
    $('#score-headline').textContent = headlines[c];
    $('#score-direction').textContent = fairness.bias_direction;

    // ── Metrics ──
    const spdVal = fairness.statistical_parity_difference;
    const dirVal = fairness.disparate_impact_ratio;

    const metricSpd = $('#metric-spd');
    metricSpd.textContent = spdVal.toFixed(4);
    metricSpd.className = 'metric-value ' + (Math.abs(spdVal) <= 0.10 ? 'val-fair' : Math.abs(spdVal) <= 0.25 ? 'val-warn' : 'val-danger');

    const metricDir = $('#metric-dir');
    metricDir.textContent = dirVal.toFixed(4);
    metricDir.className = 'metric-value ' + (dirVal >= 0.80 && dirVal <= 1.25 ? 'val-fair' : dirVal >= 0.60 ? 'val-warn' : 'val-danger');

    // SPD marker position (map -0.5...+0.5 to 0%...100%)
    const spdPos = Math.max(0, Math.min(100, ((spdVal + 0.5) / 1.0) * 100));
    $('#spd-marker').style.left = `${spdPos}%`;

    // DIR marker position (map 0...2.0 to 0%...100%)
    const dirPos = Math.max(0, Math.min(100, (dirVal / 2.0) * 100));
    $('#dir-marker').style.left = `${dirPos}%`;

    // SPD/DIR tag styling
    $('#spd-tag').className = 'metric-tag' + (Math.abs(spdVal) <= 0.10 ? '' : ' tag-out-range');
    $('#dir-tag').className = 'metric-tag' + (dirVal >= 0.80 && dirVal <= 1.25 ? '' : ' tag-out-range');

    // ── Selection Rate Comparison ──
    const compBars = $('#comparison-bars');
    let compHtml = '';

    groups.details.forEach(g => {
        const barClass = g.is_privileged ? 'bar-privileged' : 'bar-unprivileged';
        const tagClass = g.is_privileged ? 'tag-priv' : 'tag-unpriv';
        const tagLabel = g.is_privileged ? 'Privileged' : 'Unprivileged';
        const pct = (g.selection_rate * 100).toFixed(1);

        compHtml += `
            <div class="comparison-row">
                <div class="comparison-label">
                    ${g.group}
                    <span class="sub-label"><span class="group-tag ${tagClass}">${tagLabel}</span> · n=${g.total}</span>
                </div>
                <div class="comparison-bar-track">
                    <div class="comparison-bar-fill ${barClass}" style="width: 0%;">${pct}%</div>
                </div>
                <div class="comparison-percent">${pct}%</div>
            </div>
        `;
    });
    compBars.innerHTML = compHtml;

    // Animate bars
    setTimeout(() => {
        compBars.querySelectorAll('.comparison-bar-fill').forEach(bar => {
            const fillText = bar.textContent;
            bar.style.width = fillText;
        });
    }, 100);

    // ── Model Performance ──
    const perfGrid = $('#perf-grid');
    const perfMetrics = [
        { label: 'Accuracy', value: `${model.accuracy}%`, color: '' },
        { label: 'Precision', value: `${model.precision}%`, color: '' },
        { label: 'Recall', value: `${model.recall}%`, color: '' },
        { label: 'F1 Score', value: `${model.f1_score}%`, color: '' }
    ];

    perfGrid.innerHTML = perfMetrics.map(m => `
        <div class="perf-item">
            <div class="perf-item-label">${m.label}</div>
            <div class="perf-item-value">${m.value}</div>
        </div>
    `).join('');

    // ── Group Table ──
    const groupTable = $('#group-table-wrap');
    let tHtml = `<table>
        <thead><tr>
            <th>Group</th><th>Role</th><th>Total</th><th>Positive</th><th>Sel. Rate</th><th>Accuracy</th>
        </tr></thead><tbody>`;

    groups.details.forEach(g => {
        const tagClass = g.is_privileged ? 'tag-priv' : 'tag-unpriv';
        const tagText = g.is_privileged ? 'Privileged' : 'Unprivileged';
        tHtml += `<tr>
            <td style="font-family:var(--font-primary);font-weight:600">${g.group}</td>
            <td><span class="group-tag ${tagClass}">${tagText}</span></td>
            <td>${g.total}</td>
            <td>${g.positive_predictions}</td>
            <td>${(g.selection_rate * 100).toFixed(1)}%</td>
            <td>${(g.accuracy * 100).toFixed(1)}%</td>
        </tr>`;
    });

    tHtml += '</tbody></table>';
    groupTable.innerHTML = tHtml;

    // ── Feature Importance ──
    const featureBars = $('#feature-bars');
    if (model.feature_importance && model.feature_importance.length > 0) {
        const maxImp = model.feature_importance[0].importance;
        featureBars.innerHTML = model.feature_importance.map(f => {
            const width = maxImp > 0 ? (f.importance / maxImp * 100) : 0;
            const barClass = f.direction === 'negative' ? 'negative' : '';
            return `
                <div class="feature-row">
                    <span class="feature-name">${f.feature}</span>
                    <div class="feature-bar-track">
                        <div class="feature-bar-fill ${barClass}" style="width: 0%;" data-width="${width}%"></div>
                    </div>
                    <span class="feature-val">${f.importance.toFixed(3)}</span>
                </div>
            `;
        }).join('');

        // Animate
        setTimeout(() => {
            featureBars.querySelectorAll('.feature-bar-fill').forEach(bar => {
                bar.style.width = bar.dataset.width;
            });
        }, 200);
    }

    // ── Recommendations ──
    renderRecommendations(data);
    renderMitigationScores(data.mitigation);

    // Store data for export
    window._lastAnalysisData = data;

    // Scroll to results
    resultsDashboard.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function getBiasLevelMeta(score) {
    if (score < 50) {
        return { label: 'High Bias', color: 'red' };
    }
    if (score <= 75) {
        return { label: 'Moderate Bias', color: 'yellow' };
    }
    return { label: 'Fair', color: 'green' };
}

function getMitigationScoreColorClass(score) {
    if (score < 50) return 'mitigation-red';
    if (score <= 75) return 'mitigation-yellow';
    return 'mitigation-green';
}

function renderMitigationScores(mitigation) {
    const card = $('#mitigation-scores-card');
    if (!card || !mitigation) {
        if (card) card.classList.add('hidden');
        return;
    }

    card.classList.remove('hidden');

    const before = Number(mitigation.before_score ?? 0);
    const after = Number(mitigation.after_score ?? 0);
    const improvement = Number(mitigation.improvement ?? 0);

    const beforeEl = $('#mitigation-before');
    const afterEl = $('#mitigation-after');
    const improvementEl = $('#mitigation-improvement');

    beforeEl.textContent = `${before.toFixed(2)}%`;
    afterEl.textContent = `${after.toFixed(2)}%`;
    improvementEl.textContent = `${improvement >= 0 ? '+' : ''}${improvement.toFixed(2)}%`;

    beforeEl.className = `mitigation-score-value ${getMitigationScoreColorClass(before)}`;
    afterEl.className = `mitigation-score-value ${getMitigationScoreColorClass(after)}`;

    // Improvement is directional, so use sign-based coloring.
    const improvementClass = improvement > 0 ? 'mitigation-green' : (improvement < 0 ? 'mitigation-red' : 'mitigation-yellow');
    improvementEl.className = `mitigation-score-value ${improvementClass}`;
}

// ── Render Recommendations ────────────────────────────────
function renderRecommendations(data) {
    const { fairness, model, groups, dataset } = data;
    const recs = [];

    const spd = fairness.statistical_parity_difference;
    const dir = fairness.disparate_impact_ratio;
    const score = fairness.fairness_score;
    const sensitiveCol = dataset.sensitive_column;

    // Find if sensitive attribute is a top feature
    const sensitiveFeature = model.feature_importance.find(
        f => f.feature.toLowerCase() === sensitiveCol.toLowerCase()
    );
    const sensitiveRank = sensitiveFeature
        ? model.feature_importance.indexOf(sensitiveFeature) + 1
        : -1;

    // 1. Critical: significant bias detected
    if (score < 60) {
        recs.push({
            severity: 'critical',
            icon: '🚨',
            title: 'Significant Bias Detected',
            desc: `The fairness score of ${score}/100 indicates substantial disparities. The model should not be deployed in its current state without mitigation.`
        });
    } else if (score < 80) {
        recs.push({
            severity: 'warning',
            icon: '⚠️',
            title: 'Moderate Bias Detected',
            desc: `The fairness score of ${score}/100 shows room for improvement. Consider applying mitigation techniques before production use.`
        });
    }

    // 2. Sensitive attribute has high importance
    if (sensitiveFeature && sensitiveRank <= 2) {
        recs.push({
            severity: 'critical',
            icon: '🔑',
            title: `Remove or Reduce "${sensitiveCol}" Influence`,
            desc: `The sensitive attribute "${sensitiveCol}" is the #${sensitiveRank} most important feature (coefficient: ${sensitiveFeature.importance.toFixed(3)}). Consider removing it from training features or applying fairness constraints.`
        });
    }

    // 3. DIR below 80% rule
    if (dir < 0.80) {
        recs.push({
            severity: 'critical',
            icon: '⚖️',
            title: 'Fails the 80% Rule (Four-Fifths Rule)',
            desc: `Disparate Impact Ratio of ${dir.toFixed(2)} is below the 0.80 legal threshold. This could constitute adverse impact under employment discrimination law (EEOC Guidelines).`
        });
    }

    // 4. Large SPD
    if (Math.abs(spd) > 0.10) {
        recs.push({
            severity: Math.abs(spd) > 0.30 ? 'critical' : 'warning',
            icon: '📊',
            title: 'Apply Statistical Parity Constraints',
            desc: `SPD of ${spd.toFixed(4)} exceeds the ±0.10 fair range. Consider using in-processing techniques like adversarial debiasing or post-processing calibration to equalize selection rates.`
        });
    }

    // 5. Accuracy gap between groups
    const accGap = Math.abs(groups.accuracy_privileged - groups.accuracy_unprivileged);
    if (accGap > 10) {
        recs.push({
            severity: 'warning',
            icon: '🎯',
            title: 'Address Group Accuracy Disparity',
            desc: `The model accuracy differs by ${accGap.toFixed(1)}% between groups (${groups.accuracy_privileged}% vs ${groups.accuracy_unprivileged}%). Collect more representative training data for the underperforming group.`
        });
    }

    // 6. Data collection recommendation
    if (groups.unprivileged_count < groups.privileged_count * 0.5) {
        recs.push({
            severity: 'warning',
            icon: '📁',
            title: 'Address Data Imbalance',
            desc: `The unprivileged group has ${groups.unprivileged_count} samples vs ${groups.privileged_count} for the privileged group. Use oversampling (SMOTE) or collect more balanced data to improve representation.`
        });
    }

    // 7. General best practices (always show)
    recs.push({
        severity: 'info',
        icon: '🔄',
        title: 'Re-train Without Sensitive Attribute',
        desc: `Try training the model without the "${sensitiveCol}" column directly. Even then, proxy features may encode bias — use fairness-aware algorithms to mitigate indirect discrimination.`
    });

    recs.push({
        severity: 'info',
        icon: '📋',
        title: 'Document and Monitor',
        desc: 'Create a Model Card documenting the bias analysis results. Implement ongoing monitoring to detect fairness drift in production.'
    });

    // Render
    const container = $('#recommendations-list');
    container.innerHTML = recs.map(r => `
        <div class="rec-item">
            <div class="rec-icon rec-${r.severity}">${r.icon}</div>
            <div class="rec-content">
                <div class="rec-title">
                    ${r.title}
                    <span class="rec-severity sev-${r.severity}">${r.severity}</span>
                </div>
                <div class="rec-desc">${r.desc}</div>
            </div>
        </div>
    `).join('');
}

// ── Export Report ─────────────────────────────────────────
function exportReport() {
    showLoading(EXPORT_LOADING_MESSAGE);
    setApiLoading(true);
    try {
        const data = window._lastAnalysisData;
        if (!data) {
            showToast('No analysis data to export', 'error');
            return;
        }

        const report = {
            report_title: 'FairAI Pro - Bias Detection Report',
            generated_at: new Date().toISOString(),
            ...data
        };

        const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'fairai_report_' + Date.now() + '.json';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('Report exported successfully!', 'success');
    } catch (err) {
        showToast('Export failed. Please try again.', 'error');
    } finally {
        hideLoading();
        setApiLoading(false);
        if (window._lastAnalysisData) {
            resultsDashboard.classList.remove('hidden');
        }
    }
}

async function runExplain() {
    const data = window._lastAnalysisData;
    if (!data?.fairness || !data?.dataset || !data?.groups?.details?.length) {
        showToast('Run analysis first to generate explanation.', 'error');
        return;
    }

    showLoading(EXPLAIN_LOADING_MESSAGE);
    setApiLoading(true);
    try {
        const groupStats = {};
        data.groups.details.forEach(g => {
            groupStats[g.group] = g.selection_rate;
        });

        const res = await apiRequest('/explain', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fairness_score: data.fairness.fairness_score,
                sensitive_column: data.dataset.sensitive_column,
                group_stats: groupStats
            })
        }, false, DEFAULT_LOADING_MESSAGE, false);

        if (res?.explanation) {
            data.explanation = res.explanation;
            showToast('AI explanation generated.', 'success');
        } else {
            showToast('No explanation returned by API.', 'error');
        }
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        hideLoading();
        setApiLoading(false);
        if (window._lastAnalysisData) {
            resultsDashboard.classList.remove('hidden');
        }
    }
}
function resetToUpload() {
    configPanel.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    loadingOverlay.classList.add('hidden');
    uploadPanel.classList.remove('hidden');
    fileInput.value = '';
    datasetInfo = null;

    uploadPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Animate Counter ───────────────────────────────────────
function animateCounter(el, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        // Ease out cubic
        const ease = 1 - Math.pow(1 - progress, 3);
        const value = start + diff * ease;
        el.textContent = Math.round(value * 10) / 10;
        if (progress < 1) requestAnimationFrame(update);
    }

    requestAnimationFrame(update);
}

// ── Toast Notifications ───────────────────────────────────
function showToast(message, type = 'info') {
    const container = $('#toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    const icons = {
        info: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
        error: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#FF6B6B" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>',
        success: '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#00E676" stroke-width="2"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>'
    };

    toast.innerHTML = `${icons[type] || icons.info} ${message}`;
    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
