/* ============================================================
 * State + utilities
 * ============================================================ */
const State = {
  experiments: [], checkpoints: [], curvesData: [], charts: {},
  page: 'overview', expSort: { col: 'id', dir: 'desc' },
  trainPoll: null, liveES: null, liveState: null,
  trainLogES: null, trainLogModel: null,
  proxyLatencyHistory: [], drawerExpId: null,
};

const Pages = ['overview','train','live','builder','experiments','curves','checkpoints','predict','cli','settings'];
const PageTitles = {
  overview: 'Overview', train: 'Train', live: 'Live', builder: 'Builder',
  experiments: 'Experiments', curves: 'Curves', checkpoints: 'Checkpoints',
  predict: 'Predict', cli: 'CLI Reference', settings: 'Settings',
};

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      if (j.detail != null) {
        // Preserve structured detail (validation issues etc.) by serializing it,
        // so callers can JSON.parse(e.message) to inspect.
        msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail);
      }
    } catch { try { msg = await res.text(); } catch {} }
    throw new Error(msg);
  }
  return res.json();
}
async function apiPost(path, body) {
  return api(path, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body || {}) });
}
async function apiDelete(path) { return api(path, { method: 'DELETE' }); }

function toast(msg, kind = 'info', ttl = 3500) {
  const wrap = document.getElementById('toasts');
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  const icons = {
    success: '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>',
    error:   '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>',
    info:    '<svg class="toast-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/></svg>',
  };
  el.innerHTML = `${icons[kind] || icons.info}<div class="toast-body">${msg}</div><button class="toast-close" onclick="this.parentElement.remove()">×</button>`;
  wrap.appendChild(el);
  setTimeout(() => { el.classList.add('exit'); setTimeout(() => el.remove(), 220); }, ttl);
}
function confirmDialog(msg) { return Promise.resolve(window.confirm(msg)); }

function fmtDuration(s) {
  if (s == null || isNaN(s)) return '—';
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), sec = Math.floor(s%60);
  if (h) return `${h}h ${m}m ${sec}s`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function fmtTime(iso) { if (!iso) return '—'; return iso.slice(0, 16).replace('T', ' '); }
function fmtNum(n, d = 4) { if (n == null || isNaN(n)) return '—'; return Number(n).toFixed(d); }
function statusBadge(s) {
  const map = { completed: 'b-success', running: 'b-warning', failed: 'b-danger', interrupted: 'b-muted', pending: 'b-accent' };
  return `<span class="badge ${map[s] || 'b-muted'}"><span class="b-dot"></span>${s}</span>`;
}
function tagsList(raw) {
  let tags = [];
  if (Array.isArray(raw)) tags = raw;
  else if (raw) { try { tags = JSON.parse(raw); } catch { tags = []; } }
  return tags.map(t => `<span class="badge b-primary">${escapeHtml(t)}</span>`).join(' ');
}
function escapeHtml(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]); }
function copyText(text, msg = 'Copied') { navigator.clipboard?.writeText(text).then(() => toast(msg, 'success', 1800)); }

/* Navigation */
function navigate(name) {
  if (!Pages.includes(name)) return;
  if (State.page === 'train' && name !== 'train' && State.trainPoll) {
    clearInterval(State.trainPoll); State.trainPoll = null;
  }
  if (State.page === 'train' && name !== 'train' && State.trainLogES) {
    State.trainLogES.close(); State.trainLogES = null;
  }
  if (State.page === 'live' && name !== 'live' && State.liveES) {
    State.liveES.close(); State.liveES = null;
  }
  State.page = name;
  document.querySelectorAll('.page').forEach(p => p.classList.toggle('active', p.id === 'page-' + name));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.page === name));
  document.getElementById('crumb-page').textContent = PageTitles[name];

  if (name === 'overview')    initOverview();
  if (name === 'train')       initTrain();
  if (name === 'live')        initLive();
  if (name === 'builder')     initBuilder();
  if (name === 'experiments') initExperiments();
  if (name === 'curves')      initCurves();
  if (name === 'checkpoints') initCheckpoints();
  if (name === 'predict')     initPredict();
  if (name === 'settings')    initSettings();
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-page]').forEach(el => el.addEventListener('click', () => navigate(el.dataset.page)));
  navigate('overview');
  pollHealth();      setInterval(pollHealth, 8000);
  pollSystemBar();   setInterval(pollSystemBar, 5000);
  pollCounts();      setInterval(pollCounts, 12000);
  pollTopbarTrain(); setInterval(pollTopbarTrain, 4000);
  // Keyboard navigation
  let _gPressed = false; // for g-prefix shortcuts (g+o overview, g+t train, etc.)
  let _gTimer = null;
  const G_MAP = {
    o: 'overview', t: 'train', l: 'live', b: 'builder',
    e: 'experiments', c: 'curves', k: 'checkpoints', p: 'predict', s: 'settings',
  };
  document.addEventListener('keydown', e => {
    // Don't capture inside input fields
    const target = e.target;
    const isInput = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable);

    if ((e.metaKey || e.ctrlKey) && e.key === 'k') { e.preventDefault(); openPalette(); return; }
    if (e.key === 'Escape') { closePalette(); closeDrawer(); closeHFBrowser(); return; }
    if (isInput) return;

    // g-prefix shortcuts
    if (_gPressed && G_MAP[e.key]) {
      e.preventDefault();
      navigate(G_MAP[e.key]);
      _gPressed = false;
      clearTimeout(_gTimer);
      return;
    }
    if (e.key === 'g') {
      _gPressed = true;
      clearTimeout(_gTimer);
      _gTimer = setTimeout(() => { _gPressed = false; }, 1000);
      return;
    }
    // Single-key shortcuts
    if (e.key === '?') { e.preventDefault(); toast(shortcutHelp(), 'info', 8000); }
    if (e.key === 'r' && !e.metaKey && !e.ctrlKey) { e.preventDefault(); refreshAll(); }
  });

  // Predict drag/drop
  const drop = document.getElementById('pg-drop');
  if (drop) {
    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', e => {
      e.preventDefault(); drop.classList.remove('over');
      if (e.dataTransfer.files[0]) pgHandleFile(e.dataTransfer.files[0]);
    });
  }
});

/* Background polls */
async function pollHealth() {
  try {
    const h = await api('/api/health');
    document.getElementById('sb-dot-api').className = 'dot online';
    document.getElementById('sb-uptime').textContent = `up ${fmtDuration(h.uptime)}`;
    document.getElementById('ts-dot').className = 'dot online';
    document.getElementById('ts-text').textContent = 'Healthy';
  } catch {
    document.getElementById('sb-dot-api').className = 'dot error';
    document.getElementById('ts-dot').className = 'dot error';
    document.getElementById('ts-text').textContent = 'Offline';
  }
}
async function pollCounts() {
  try {
    const stats = await api('/api/stats');
    document.getElementById('nav-exp-count').textContent = stats.total_experiments;
    document.getElementById('nav-ckpt-count').textContent = stats.total_checkpoints;
  } catch {}
}
async function pollTopbarTrain() {
  try {
    const s = await api('/api/train/status');
    const dot = document.getElementById('sb-dot-train');
    const text = document.getElementById('sb-train-text');
    document.getElementById('nav-live-pulse').classList.toggle('hidden', !s.running);
    if (s.running) { dot.className = 'dot training'; text.textContent = `${s.experiment || 'training'} — PID ${s.pid}`; }
    else { dot.className = 'dot'; text.textContent = s.returncode != null ? `Exited (${s.returncode})` : 'Idle'; }
  } catch {}
}
async function pollSystemBar() {
  try {
    const sys = await api('/api/system');
    renderSystemStrip('system-strip', sys);
    renderSystemStrip('live-system-strip', sys);
  } catch {}
}
function renderSystemStrip(elId, sys) {
  const el = document.getElementById(elId);
  if (!el) return;
  const cells = [];
  if (sys.cpu_percent != null) {
    const pct = Math.round(sys.cpu_percent);
    cells.push(`<div class="sys-cell">
      <span class="sys-cell-label">CPU</span>
      <div class="sys-bar"><div class="sys-bar-fill" style="width:${pct}%"></div></div>
      <span class="sys-cell-value">${pct}%</span>
    </div>`);
  }
  if (sys.memory) {
    const m = sys.memory;
    cells.push(`<div class="sys-cell">
      <span class="sys-cell-label">RAM</span>
      <div class="sys-bar"><div class="sys-bar-fill" style="width:${m.percent}%;background:var(--accent)"></div></div>
      <span class="sys-cell-value">${m.used_gb} / ${m.total_gb} GB</span>
    </div>`);
  }
  if (sys.gpus && sys.gpus.length) {
    const g = sys.gpus[0];
    // Prefer compute util% when available (NVIDIA), fall back to mem%
    const pct = g.util_percent != null
      ? g.util_percent
      : (g.mem_percent != null ? g.mem_percent : 0);
    let valueText, titleText;
    if (g.kind === 'mps') {
      valueText = 'MPS ready';
      titleText = `${g.name}${g.mem_used_gb != null ? ` · ${g.mem_used_gb} GB allocated` : ''}`;
    } else if (g.util_percent != null && g.mem_used_gb != null) {
      valueText = `${g.util_percent}% · ${g.mem_used_gb}G`;
      titleText = `${g.name} — utilization ${g.util_percent}%, mem ${g.mem_used_gb}/${g.mem_total_gb} GB`
                 + (g.temperature_c ? `, ${g.temperature_c}°C` : '')
                 + (g.power_watts ? `, ${g.power_watts}W` : '');
    } else if (g.mem_used_gb != null) {
      valueText = `${g.mem_used_gb}/${g.mem_total_gb} GB`;
      titleText = g.name;
    } else {
      valueText = g.name.length > 18 ? g.name.slice(0, 16) + '…' : g.name;
      titleText = g.name;
    }
    cells.push(`<div class="sys-cell">
      <span class="sys-cell-label">GPU</span>
      <div class="sys-bar"><div class="sys-bar-fill" style="width:${pct}%;background:var(--violet)"></div></div>
      <span class="sys-cell-value" title="${escapeHtml(titleText)}">${escapeHtml(valueText)}</span>
    </div>`);
  } else {
    const accel = sys.accelerator || 'cpu';
    cells.push(`<div class="sys-cell">
      <span class="sys-cell-label">GPU</span>
      <span class="sys-cell-value text-muted">${accel === 'cpu' ? 'CPU only' : escapeHtml(accel)}</span>
    </div>`);
  }
  if (sys.disk) {
    cells.push(`<div class="sys-cell">
      <span class="sys-cell-label">Disk</span>
      <div class="sys-bar"><div class="sys-bar-fill" style="width:${sys.disk.percent}%;background:var(--warning)"></div></div>
      <span class="sys-cell-value">${sys.disk.used_gb} / ${sys.disk.total_gb} GB</span>
    </div>`);
  }
  el.innerHTML = cells.join('');
}

/* OVERVIEW */
async function initOverview() {
  try {
    const stats = await api('/api/stats');
    document.getElementById('kpi-total').textContent = stats.total_experiments;
    document.getElementById('kpi-done').textContent  = stats.completed;
    document.getElementById('kpi-run').textContent   = stats.running;
    document.getElementById('kpi-ckpt').textContent  = stats.total_checkpoints;
    document.getElementById('kpi-done-meta').textContent = `${stats.failed||0} failed · ${stats.interrupted||0} interrupted`;
    document.getElementById('kpi-run-meta').textContent  = `Active subprocess: ${stats.active_subprocess ? 'yes' : 'no'}`;
    document.getElementById('kpi-ckpt-meta').textContent = `${stats.checkpoints_size_mb} MB on disk`;
  } catch {
    ['kpi-total','kpi-done','kpi-run','kpi-ckpt'].forEach(id => document.getElementById(id).textContent = '—');
  }

  try {
    State.experiments = await api('/api/experiments');
    const recent = State.experiments.slice(0, 6);
    const wrap = document.getElementById('ov-recent');
    if (!recent.length) {
      wrap.innerHTML = `<div class="empty"><div class="empty-icon">🧪</div>No experiments yet — go to <a href="#" onclick="navigate('train');return false;" style="color:var(--primary)">Train</a> to start.</div>`;
    } else {
      wrap.innerHTML = `<table>
        <thead><tr><th>Name</th><th>Status</th><th>Best val</th><th>Created</th></tr></thead>
        <tbody>${recent.map(e => `
          <tr class="clickable" onclick="openDrawer(${e.id})">
            <td><strong>${escapeHtml(e.name)}</strong></td>
            <td>${statusBadge(e.status)}</td>
            <td class="num">${e.best_val_loss != null ? fmtNum(e.best_val_loss) : '—'}</td>
            <td class="muted text-xs">${fmtTime(e.created_at)}</td>
          </tr>`).join('')}
        </tbody></table>`;
    }
    renderOverviewChart(State.experiments);
  } catch (e) {
    document.getElementById('ov-recent').innerHTML = `<div class="empty">Failed to load: ${escapeHtml(e.message)}</div>`;
  }

  try {
    const cks = await api('/api/checkpoints/recent');
    const out = document.getElementById('ov-ckpts');
    if (!cks.length) {
      out.innerHTML = `<div class="empty text-xs">No checkpoints yet</div>`;
    } else {
      out.innerHTML = cks.map(c => `
        <div class="between" style="padding:8px 0;border-bottom:1px solid var(--border)">
          <div>
            <div style="font-weight:500">${escapeHtml(c.experiment)}</div>
            <div class="text-xs text-muted">${escapeHtml(c.name)} · ${escapeHtml(c.model_type) || '—'}</div>
          </div>
          <div class="text-xs text-mono text-muted">${c.size_mb} MB</div>
        </div>`).join('');
    }
  } catch {}
}
function renderOverviewChart(exps) {
  const top = exps.filter(e => e.best_val_loss != null).sort((a, b) => a.best_val_loss - b.best_val_loss).slice(0, 8);
  const ctx = document.getElementById('ov-chart').getContext('2d');
  destroyChart('ov-chart');
  if (!top.length) { return; }
  State.charts['ov-chart'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(e => e.name.length > 14 ? e.name.slice(0, 12) + '…' : e.name),
      datasets: [{
        label: 'Best val loss',
        data: top.map(e => e.best_val_loss),
        backgroundColor: top.map((_, i) => i === 0 ? 'rgba(124,131,255,0.85)' : 'rgba(124,131,255,0.55)'),
        borderColor: 'rgba(124,131,255,1)', borderWidth: 1, borderRadius: 6,
      }],
    },
    options: chartOpts({ noLegend: true }),
  });
}

/* TRAIN */
async function initTrain() {
  await loadConfigs();
  initTrainLogModel();
  await refreshTrainLogs();
  connectTrainLogStream();
  await pollTrainStatus();
  if (State.trainPoll) clearInterval(State.trainPoll);
  State.trainPoll = setInterval(pollTrainStatus, 3000);
}
function initTrainLogModel() {
  State.trainLogModel = {
    rows: [''],
    maxRows: 1200,
    cursorRow: 0,
    ansiMode: 'none', // none | esc | csi | osc
    ansiBuf: '',
    sawCR: false,
  };
}
function processTrainLogChunk(chunk) {
  if (!State.trainLogModel) initTrainLogModel();
  const m = State.trainLogModel;
  const clampCursor = () => {
    if (m.cursorRow < 0) m.cursorRow = 0;
    if (m.cursorRow >= m.rows.length) m.cursorRow = m.rows.length - 1;
  };
  const trimRows = () => {
    if (m.rows.length > m.maxRows) {
      const drop = m.rows.length - m.maxRows;
      m.rows.splice(0, drop);
      m.cursorRow = Math.max(0, m.cursorRow - drop);
    }
  };
  const advanceLine = () => {
    // Terminal newline moves the cursor down; it only creates a new row when
    // we're already at the bottom.
    if (m.cursorRow >= m.rows.length - 1) {
      m.rows.push('');
    } else {
      m.cursorRow += 1;
    }
    trimRows();
  };
  const handleCsi = (seq) => {
    const final = seq.slice(-1);
    const body = seq.slice(0, -1);
    const nums = body.replace(/^[?]/, '').split(';').filter(Boolean).map((n) => parseInt(n, 10));
    const n = Number.isFinite(nums[0]) ? nums[0] : 1;
    if (final === 'A') {
      m.cursorRow -= n;
      clampCursor();
    } else if (final === 'B') {
      m.cursorRow += n;
      clampCursor();
    } else if (final === 'K') {
      m.rows[m.cursorRow] = '';
    } else if (final === 'J') {
      // clear display; for logs, keep only current row to prevent noise
      m.rows = [m.rows[m.cursorRow] || ''];
      m.cursorRow = 0;
    }
  };
  for (const ch of chunk) {
    if (m.ansiMode === 'esc') {
      if (ch === '[') {
        m.ansiMode = 'csi';
        m.ansiBuf = '';
        continue;
      }
      if (ch === ']') {
        m.ansiMode = 'osc';
        m.ansiBuf = '';
        continue;
      }
      m.ansiMode = 'none';
      continue;
    }
    if (m.ansiMode === 'csi') {
      m.ansiBuf += ch;
      const code = ch.charCodeAt(0);
      if (code >= 0x40 && code <= 0x7e) {
        handleCsi(m.ansiBuf);
        m.ansiMode = 'none';
        m.ansiBuf = '';
      }
      continue;
    }
    if (m.ansiMode === 'osc') {
      // OSC ends with BEL or ST (ESC \). Treat BEL as terminator; ESC transitions.
      if (ch === '\x07') {
        m.ansiMode = 'none';
      } else if (ch === '\x1b') {
        m.ansiMode = 'esc';
      }
      continue;
    }

    if (m.sawCR) {
      if (ch === '\n') {
        advanceLine();
        m.sawCR = false;
        continue;
      }
      // Standalone CR means "return to line start / overwrite".
      m.rows[m.cursorRow] = '';
      m.sawCR = false;
    }
    if (ch === '\x1b') {
      m.ansiMode = 'esc';
      continue;
    }
    if (ch === '\r') {
      m.sawCR = true;
      continue;
    }
    if (ch === '\n') {
      advanceLine();
      continue;
    }
    m.rows[m.cursorRow] += ch;
  }
}
function renderTrainLogs() {
  const el = document.getElementById('train-log');
  if (!el || !State.trainLogModel) return;
  const text = State.trainLogModel.rows.join('\n');
  el.textContent = text || 'Subprocess output will appear here after training starts…';
  el.scrollTop = el.scrollHeight;
}
function renderTrainLogsRawFallback(rawText) {
  const el = document.getElementById('train-log');
  if (!el) return;
  if (!rawText) return;
  // If reducer output is unexpectedly empty, still show readable snapshot text.
  el.textContent = rawText.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, '');
  el.scrollTop = el.scrollHeight;
}
function connectTrainLogStream() {
  if (State.trainLogES) {
    State.trainLogES.close();
    State.trainLogES = null;
  }
  const es = new EventSource('/api/train/logs/stream');
  State.trainLogES = es;
  es.addEventListener('chunk', (e) => {
    try {
      const payload = JSON.parse(e.data || '{}');
      if (payload.chunk) {
        processTrainLogChunk(payload.chunk);
        renderTrainLogs();
      }
    } catch {}
  });
}
async function loadConfigs() {
  try {
    const cfgs = await api('/api/configs');
    const sel = document.getElementById('cfg-select');
    const prev = sel.value;
    sel.innerHTML = '<option value="">— pick a config —</option>' +
      cfgs.map(c => `<option value="${escapeHtml(c.path)}"
        data-type="${escapeHtml(c.model_type)}" data-fw="${escapeHtml(c.framework)}"
        data-epochs="${escapeHtml(String(c.num_epochs))}" data-bs="${escapeHtml(String(c.batch_size))}"
        data-lr="${c.lr ?? ''}" data-name="${escapeHtml(c.experiment_name)}">
        ${escapeHtml(c.experiment_name)} · ${escapeHtml(c.model_type)}
      </option>`).join('');
    if (prev) sel.value = prev;
    if (!cfgs.length) sel.innerHTML = '<option value="">No configs found — use Builder or `neural init`</option>';
  } catch (e) { toast('Could not load configs: ' + e.message, 'error'); }
}
function onConfigChange() {
  const sel = document.getElementById('cfg-select');
  const opt = sel.options[sel.selectedIndex];
  const path = sel.value;
  document.getElementById('cfg-path').value = path;
  const chips = document.getElementById('cfg-chips');
  if (!path) { chips.classList.add('hidden'); document.getElementById('train-start').disabled = true; return; }
  chips.classList.remove('hidden');
  const lr = opt.dataset.lr ? parseFloat(opt.dataset.lr).toExponential(1) : '—';
  chips.innerHTML = `<div class="grid grid-5" style="gap:8px">
    ${chipNode('Name', opt.dataset.name)}
    ${chipNode('Model', opt.dataset.type)}
    ${chipNode('Framework', opt.dataset.fw)}
    ${chipNode('Epochs', opt.dataset.epochs)}
    ${chipNode('LR', lr)}
  </div>`;
  document.getElementById('train-start').disabled = false;
}
function chipNode(label, value) {
  return `<div style="background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:8px 10px">
    <div class="text-xs text-muted" style="text-transform:uppercase;letter-spacing:0.06em">${escapeHtml(label)}</div>
    <div class="text-mono mt-2" style="font-size:13px">${escapeHtml(String(value || '—'))}</div>
  </div>`;
}
function onPathInput() {
  const path = document.getElementById('cfg-path').value.trim();
  document.getElementById('train-start').disabled = !path;
  const sel = document.getElementById('cfg-select');
  if (sel.value !== path) sel.value = '';
  document.getElementById('cfg-chips').classList.add('hidden');
}
function addOverride(prefill) {
  const list = document.getElementById('ov-list');
  const row = document.createElement('div');
  row.className = 'row-tight mb-2';
  row.innerHTML = `<input class="input input-mono" type="text" placeholder="training.optimizer.lr=0.001" value="${escapeHtml(prefill || '')}" />
    <button class="btn btn-icon btn-ghost" onclick="this.parentElement.remove()"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>`;
  list.appendChild(row);
  row.querySelector('input').focus();
}
function getOverrides() {
  return [...document.querySelectorAll('#ov-list input')].map(el => el.value.trim()).filter(v => v && v.includes('='));
}
async function startTraining() {
  const path = document.getElementById('cfg-path').value.trim() || document.getElementById('cfg-select').value;
  if (!path) { toast('Pick a config first', 'error'); return; }
  const startBtn = document.getElementById('train-start');
  startBtn.disabled = true;
  startBtn.innerHTML = '<span class="spin"></span> Starting…';
  try {
    const res = await apiPost('/api/train/start', { config_path: path, overrides: getOverrides() });
    document.getElementById('train-pid').textContent = res.pid;
    document.getElementById('train-cmd').textContent = res.cmd;
    document.getElementById('train-proc-info').classList.remove('hidden');
    toast(`Training started (PID ${res.pid})`, 'success');
    setTimeout(refreshTrainLogs, 1200);
  } catch (e) {
    // Try to parse a structured validation payload — the dashboard returns
    // 422 with {detail: {message, issues:[...]}} when pre-flight fails.
    let parsed = null;
    try { parsed = JSON.parse(e.message); } catch {}
    if (parsed && parsed.issues) {
      const errs = parsed.issues.filter(i => i.severity === 'error');
      const warns = parsed.issues.filter(i => i.severity === 'warning');
      const lines = [
        '<strong>' + escapeHtml(parsed.message || 'Config failed validation') + '</strong>',
        '<div class="text-xs text-muted mt-2">',
        ...errs.map(i => `<div style="margin-bottom:6px">✗ <span class="text-mono">${escapeHtml(i.field)}</span>: ${escapeHtml(i.message)}${i.hint ? `<div class="text-faint" style="margin-left:14px">→ ${escapeHtml(i.hint)}</div>` : ''}</div>`),
        ...warns.map(i => `<div style="margin-bottom:6px;color:var(--warning)">⚠ <span class="text-mono">${escapeHtml(i.field)}</span>: ${escapeHtml(i.message)}</div>`),
        '</div>',
      ].join('');
      toast(lines, 'error', 12000);
    } else {
      toast('Failed to start: ' + e.message, 'error', 5000);
    }
  } finally {
    startBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5,3 19,12 5,21"/></svg> Start training`;
    pollTrainStatus();
  }
}
async function stopTraining() {
  if (!await confirmDialog('Stop the current training? This kills the subprocess and all DataLoader workers.')) return;
  const stopBtn = document.getElementById('train-stop');
  stopBtn.disabled = true;
  stopBtn.innerHTML = '<span class="spin"></span> Stopping…';
  try {
    const r = await apiPost('/api/train/stop');
    if (r.stopped) toast(`Stopped (PID ${r.pid})`, 'success');
    else toast(r.reason || 'Nothing to stop', 'info');
  } catch (e) { toast('Stop failed: ' + e.message, 'error'); }
  finally {
    stopBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect x="6" y="6" width="12" height="12"/></svg> Stop`;
    pollTrainStatus();
    setTimeout(refreshTrainLogs, 800);
  }
}
async function pollTrainStatus() {
  try {
    const s = await api('/api/train/status');
    const dot = document.getElementById('train-status-dot');
    const text = document.getElementById('train-status-text');
    const startBtn = document.getElementById('train-start');
    const stopBtn = document.getElementById('train-stop');
    if (s.running) {
      dot.className = 'dot training';
      text.textContent = `Training — PID ${s.pid}`;
      stopBtn.disabled = false;
      startBtn.disabled = true;
    } else {
      dot.className = 'dot';
      const hasCfg = document.getElementById('cfg-path').value.trim() || document.getElementById('cfg-select').value;
      stopBtn.disabled = true;
      startBtn.disabled = !hasCfg;
      if (s.returncode != null) text.textContent = s.returncode === 0 ? 'Completed' : `Exited (${s.returncode})`;
      else text.textContent = 'Idle';
    }
  } catch {}
}
async function refreshTrainLogs() {
  try {
    const data = await api('/api/train/logs?chars=200000');
    initTrainLogModel();
    if (data.text) processTrainLogChunk(data.text);
    renderTrainLogs();
    if (data.text && (!State.trainLogModel || State.trainLogModel.rows.join('').trim() === '')) {
      renderTrainLogsRawFallback(data.text);
    }
  } catch {}
}

/* LIVE */
function newLiveState() {
  return {
    connected: false, running: false, experiment: null,
    totalEpochs: 0, totalBatches: 0, currentEpoch: 0, currentBatch: 0,
    startTs: null, lastBatchTs: null,
    batches: [], epTrain: [], epVal: [], epAcc: [], epValAcc: [],
  };
}
async function initLive() { await connectLive(); }
async function connectLive() {
  if (State.liveES) { State.liveES.close(); State.liveES = null; }
  State.liveState = newLiveState();
  initLiveCharts();
  setLiveStatus('connecting', 'Connecting…');

  try {
    const snap = await api('/api/training/live');
    snap.events.forEach(ev => applyLiveEvent(ev, false));
    if (!snap.is_running && (!State.liveState.experiment)) {
      document.getElementById('live-empty').classList.remove('hidden');
    }
  } catch {}

  const es = new EventSource('/api/events/stream');
  State.liveES = es;
  ['training_start','batch','epoch','checkpoint','early_stop','training_end'].forEach(t => {
    es.addEventListener(t, e => applyLiveEvent(JSON.parse(e.data), true));
  });
  es.addEventListener('connected', () => setLiveStatus('connected', 'Connected — waiting for run'));
  es.onerror = () => setLiveStatus('error', 'Connection lost — retrying…');
}
function setLiveStatus(state, text) {
  const dot = document.getElementById('live-dot');
  const colors = { connected: 'online', connecting: 'training', error: 'error', running: 'training', done: 'online' };
  dot.className = 'dot ' + (colors[state] || '');
  document.getElementById('live-text').textContent = text;
}
function applyLiveEvent(ev, log) {
  const s = State.liveState;
  if (ev.type === 'training_start') {
    Object.assign(s, newLiveState());
    s.running = true;
    s.experiment = ev.experiment;
    s.totalEpochs = ev.total_epochs;
    s.totalBatches = ev.total_batches;
    s.startTs = ev.ts;
    initLiveCharts();
    setLiveStatus('running', `Training: ${ev.experiment}`);
    document.getElementById('live-empty').classList.add('hidden');
    document.getElementById('live-header').classList.remove('hidden');
    document.getElementById('live-stop-btn').classList.remove('hidden');
    document.getElementById('lv-name').textContent = ev.experiment;
    document.getElementById('lv-model').textContent = `${ev.model_type} · ${ev.framework}`;
    document.getElementById('lv-dev').textContent = ev.device;
    document.getElementById('lv-epoch').textContent = `0/${ev.total_epochs}`;
    document.getElementById('lv-epoch-meta').textContent = `${ev.total_batches} batches/epoch`;
    if (log) appendLiveLog(ev.type, `Started "${ev.experiment}" — ${ev.model_type} on ${ev.device}, ${ev.total_epochs} epochs`);
  }
  else if (ev.type === 'batch') {
    s.currentEpoch = ev.epoch; s.currentBatch = ev.batch; s.lastBatchTs = ev.ts;
    const x = (ev.epoch - 1) * s.totalBatches + ev.batch;
    s.batches.push({ x, y: ev.loss });
    while (s.batches.length > 600) s.batches.shift();
    document.getElementById('lv-epoch').textContent = `${ev.epoch}/${ev.total_epochs}`;
    document.getElementById('lv-tl').textContent = ev.loss.toFixed(4);
    document.getElementById('lv-lr').textContent = ev.lr.toExponential(2);
    if (ev.metrics?.accuracy != null) document.getElementById('lv-acc').textContent = (ev.metrics.accuracy * 100).toFixed(1) + '%';
    const epPct = ((ev.epoch - 1) / ev.total_epochs * 100);
    const btPct = (ev.batch / s.totalBatches * 100);
    document.getElementById('lv-ep-bar').style.width = epPct.toFixed(1) + '%';
    document.getElementById('lv-ep-pct').textContent = epPct.toFixed(0) + '%';
    document.getElementById('lv-bt-bar').style.width = btPct.toFixed(1) + '%';
    document.getElementById('lv-bt-pct').textContent = btPct.toFixed(0) + '%';
    document.getElementById('lv-bt-num').textContent = `${ev.batch}/${s.totalBatches}`;
    if (s.startTs) {
      const elapsed = ev.ts - s.startTs;
      document.getElementById('lv-elapsed').textContent = fmtDuration(elapsed);
      const totalSteps = s.totalBatches * s.totalEpochs;
      const stepsDone = (ev.epoch - 1) * s.totalBatches + ev.batch;
      if (stepsDone > 0) {
        const eta = Math.max(0, (elapsed / stepsDone) * (totalSteps - stepsDone));
        document.getElementById('lv-eta').textContent = fmtDuration(eta);
      }
    }
    document.getElementById('lv-batch-stat').textContent = `${s.batches.length} batches`;
    pushChart('ch-batch', s.batches.map(p => p.x), [{ label: 'Batch loss', data: s.batches.map(p => p.y), color: '#7c83ff' }]);
    if (log && ev.batch % 25 === 0) appendLiveLog(ev.type, `Epoch ${ev.epoch} | Batch ${ev.batch}/${s.totalBatches} | loss=${ev.loss.toFixed(4)}`);
  }
  else if (ev.type === 'epoch') {
    s.epTrain.push({ x: ev.epoch, y: ev.train_metrics.loss });
    if (ev.val_metrics?.loss != null) s.epVal.push({ x: ev.epoch, y: ev.val_metrics.loss });
    if (ev.train_metrics.accuracy != null) s.epAcc.push({ x: ev.epoch, y: ev.train_metrics.accuracy });
    if (ev.val_metrics?.accuracy != null) s.epValAcc.push({ x: ev.epoch, y: ev.val_metrics.accuracy });
    if (ev.val_metrics?.loss != null) document.getElementById('lv-vl').textContent = ev.val_metrics.loss.toFixed(4);
    const epPct = (ev.epoch / ev.total_epochs * 100);
    document.getElementById('lv-ep-bar').style.width = epPct.toFixed(1) + '%';
    document.getElementById('lv-ep-pct').textContent = epPct.toFixed(0) + '%';
    document.getElementById('lv-bt-bar').style.width = '100%';
    document.getElementById('lv-bt-pct').textContent = '100%';
    const xs = s.epTrain.map(p => p.x);
    pushChart('ch-epoch', xs, [
      { label: 'Train', data: s.epTrain.map(p => p.y), color: '#7c83ff' },
      { label: 'Val', data: s.epVal.map(p => p.y), color: '#34d399' },
    ]);
    pushChart('ch-acc', xs, [
      { label: 'Train acc', data: s.epAcc.map(p => p.y), color: '#fbbf24' },
      { label: 'Val acc', data: s.epValAcc.map(p => p.y), color: '#22d3ee' },
    ]);
    if (log) {
      let m = `Epoch ${ev.epoch}/${ev.total_epochs} done | train=${ev.train_metrics.loss?.toFixed(4)}`;
      if (ev.val_metrics?.loss != null) m += ` | val=${ev.val_metrics.loss.toFixed(4)}`;
      if (ev.val_metrics?.accuracy != null) m += ` | val_acc=${(ev.val_metrics.accuracy * 100).toFixed(1)}%`;
      appendLiveLog(ev.type, m);
    }
  }
  else if (ev.type === 'checkpoint') {
    if (log) appendLiveLog(ev.type, `Checkpoint saved${ev.is_best ? ' (best)' : ''}: ${ev.path.split('/').pop()}`);
  }
  else if (ev.type === 'early_stop') {
    setLiveStatus('done', `Early stopped @ epoch ${ev.epoch}`);
    if (log) appendLiveLog(ev.type, `Early stopping @ epoch ${ev.epoch} | best=${ev.best_val_loss}`);
  }
  else if (ev.type === 'training_end') {
    s.running = false;
    const status = ev.status;
    const label = { completed: 'Completed', interrupted: 'Stopped', failed: 'Failed' }[status] || status;
    setLiveStatus('done', `${label}${ev.duration ? ' — ' + fmtDuration(ev.duration) : ''}`);
    document.getElementById('live-stop-btn').classList.add('hidden');
    if (log) {
      let m = `Training ${status}`;
      if (ev.total_epochs) m += ` | ${ev.total_epochs} epochs`;
      if (ev.best_val_loss != null) m += ` | best=${ev.best_val_loss.toFixed(4)} @ epoch ${ev.best_epoch}`;
      if (ev.duration) m += ` | ${fmtDuration(ev.duration)}`;
      appendLiveLog(ev.type, m);
    }
    if (State.liveES) { State.liveES.close(); State.liveES = null; }
  }
}
async function stopFromLive() {
  if (!await confirmDialog('Stop the active training?')) return;
  try { await apiPost('/api/train/stop'); toast('Stop signal sent', 'success'); }
  catch (e) { toast('Stop failed: ' + e.message, 'error'); }
}
function appendLiveLog(type, msg) {
  const el = document.getElementById('live-log');
  const ts = new Date().toTimeString().slice(0, 8);
  const div = document.createElement('div');
  div.innerHTML = `<span class="ts">[${ts}]</span> <span class="ev-${type}">${escapeHtml(msg)}</span>`;
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
  while (el.children.length > 250) el.removeChild(el.firstChild);
}
function initLiveCharts() { ['ch-batch', 'ch-epoch', 'ch-acc'].forEach(destroyChart); }

/* BUILDER */
let bMtype = 'mlp';
let bWired = false;
function initBuilder() {
  if (!bWired) {
    bWired = true;
    document.querySelectorAll('#b-mtype-tabs .tab').forEach(t => t.addEventListener('click', () => bSwitchMtype(t.dataset.mtype)));
    if (!document.getElementById('b-mlp-layers').children.length) {
      bAddMlpLayer({ size: 256, activation: 'relu', dropout: 0.2, batch_norm: true });
      bAddMlpLayer({ size: 128, activation: 'relu', dropout: 0.2, batch_norm: true });
    }
    bDataChange();
    bLoadTasks();
    ['b-name','b-desc','b-tags','b-task','b-mlp-in','b-mlp-out','b-cnn-c','b-cnn-h','b-cnn-w','b-cnn-bb','b-cnn-out',
     'b-rnn-var','b-rnn-in','b-rnn-h','b-rnn-l','b-rnn-d','b-rnn-out','b-rnn-bi',
     'b-tx-vocab','b-tx-seq','b-tx-d','b-tx-h','b-tx-l','b-tx-ff','b-tx-out',
     'b-hf-pretrained','b-hf-output','b-hf-freeze',
     'b-epochs','b-bs','b-lr','b-opt','b-sched','b-dev','b-data-source','b-val-split','b-test-split',
    ].forEach(id => { const el = document.getElementById(id); if (el) el.addEventListener('input', bRender); });
  }
  bRender();
}
const B_MTYPE_PANELS = {
  mlp:         'b-mlp',
  cnn:         'b-cnn',
  rnn:         'b-rnn',
  transformer: 'b-tx',
  audio_cnn:   'b-audio',
  tcn:         'b-tcn',
  tabular:     'b-tab',
  video_cnn:   'b-vid',
  hf_pipeline: 'b-hf',
};
/* ----- Task picker (HF pipeline-tag taxonomy) ----- */
let _bTaskCatalog = null;     // {groups: [...], meta: {task: {...}}}

async function bLoadTasks() {
  if (_bTaskCatalog) return _bTaskCatalog;
  try {
    _bTaskCatalog = await api('/api/tasks');
  } catch (e) {
    _bTaskCatalog = { groups: [], meta: {} };
  }
  // Populate the family dropdown
  const groupSel = document.getElementById('b-task-group');
  if (groupSel) {
    groupSel.innerHTML = '<option value="">— pick a family —</option>' +
      _bTaskCatalog.groups.map(g => `<option value="${escapeHtml(g.label)}">${escapeHtml(g.label)}</option>`).join('');
  }
  return _bTaskCatalog;
}

function bRenderTaskList() {
  if (!_bTaskCatalog) return;
  const family = document.getElementById('b-task-group').value;
  const sel = document.getElementById('b-task');
  const group = _bTaskCatalog.groups.find(g => g.label === family);
  if (!group) {
    sel.innerHTML = '<option value="">— pick a family first —</option>';
    return;
  }
  sel.innerHTML = '<option value="">— select a task —</option>' +
    group.tasks.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
}

function bTaskChanged() {
  if (!_bTaskCatalog) return;
  const task = document.getElementById('b-task').value;
  const meta = _bTaskCatalog.meta[task];
  const out = document.getElementById('b-task-meta');
  if (!task || !meta) { out.classList.add('hidden'); return; }
  out.classList.remove('hidden');
  const flags = [];
  if (meta.multimodal) flags.push('<span class="badge b-accent text-xs">multimodal</span>');
  if (meta.generative) flags.push('<span class="badge b-primary text-xs">generative</span>');
  if (meta.requires_pretrained) flags.push('<span class="badge b-warning text-xs">requires pretrained</span>');
  out.innerHTML = `
    <div style="margin-bottom:6px"><strong>${escapeHtml(task)}</strong>  ${flags.join(' ')}</div>
    <div>Inputs: <span class="text-mono">${meta.inputs.map(escapeHtml).join(', ')}</span> →
         Outputs: <span class="text-mono">${meta.outputs.map(escapeHtml).join(', ')}</span></div>
    <div class="mt-2">Suggested architectures:
      ${meta.suggested_models.map(m => `<button class="btn btn-secondary btn-xs" style="margin-right:4px" onclick="bSwitchMtype('${escapeHtml(m)}')">${escapeHtml(m)}</button>`).join('')}
    </div>`;
  // Auto-pick the first suggested architecture (the user can override).
  if (meta.suggested_models.length) bSwitchMtype(meta.suggested_models[0]);
}

function bSwitchMtype(t) {
  bMtype = t;
  document.querySelectorAll('#b-mtype-tabs .tab').forEach(x => x.classList.toggle('active', x.dataset.mtype === t));
  Object.values(B_MTYPE_PANELS).forEach(id => {
    const el = document.getElementById(id); if (el) el.classList.add('hidden');
  });
  const target = document.getElementById(B_MTYPE_PANELS[t] || 'b-mlp');
  if (target) target.classList.remove('hidden');
  bRender();
}
function bAddMlpLayer(prefill = {}) {
  const wrap = document.getElementById('b-mlp-layers');
  const idx = wrap.children.length;
  const el = document.createElement('div');
  el.className = 'layer-card';
  el.innerHTML = `
    <div class="layer-card-h">
      <div class="text-xs text-muted">Layer ${idx + 1}</div>
      <button class="btn btn-icon btn-ghost btn-xs" onclick="this.closest('.layer-card').remove();bRender()"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg></button>
    </div>
    <div class="grid grid-4" style="gap:8px">
      <div><label class="label">Size</label><input class="input" type="number" data-k="size" value="${prefill.size || 128}" /></div>
      <div><label class="label">Activation</label>
        <select class="select" data-k="activation">
          ${['relu','gelu','silu','tanh','sigmoid','none'].map(a => `<option value="${a}" ${prefill.activation === a ? 'selected' : ''}>${a}</option>`).join('')}
        </select>
      </div>
      <div><label class="label">Dropout</label><input class="input" type="number" step="0.05" data-k="dropout" value="${prefill.dropout ?? 0}" /></div>
      <div><label class="label">BN</label>
        <select class="select" data-k="bn">
          <option value="false" ${!prefill.batch_norm ? 'selected' : ''}>off</option>
          <option value="true" ${prefill.batch_norm ? 'selected' : ''}>on</option>
        </select>
      </div>
    </div>`;
  el.querySelectorAll('input,select').forEach(x => x.addEventListener('input', bRender));
  wrap.appendChild(el);
  bRender();
}
let _bSelectedConfig = null;
async function bInspectDataset(config) {
  const name = (document.getElementById('b-data-ds')?.value || '').trim();
  const out = document.getElementById('b-data-inspect');
  if (!name || !out) { toast('Enter a dataset name first', 'error'); return; }
  if (config !== undefined) _bSelectedConfig = config;
  out.classList.remove('hidden');
  out.innerHTML = '<span class="spin"></span> Inspecting…';
  try {
    const url = new URLSearchParams({name});
    if (_bSelectedConfig) url.set('config', _bSelectedConfig);
    const info = await api('/api/hf/inspect?' + url.toString());

    // Multi-config dataset (e.g. superb): render config picker chips and stop.
    if (info.needs_config) {
      const chips = (info.available_configs || []).map(c =>
        `<button class="btn btn-secondary btn-xs" onclick="bInspectDataset('${escapeHtml(c)}')">${escapeHtml(c)}</button>`
      ).join(' ');
      out.innerHTML = `
        <div class="text-xs mb-2"><strong>${escapeHtml(name)}</strong> has multiple configurations — pick one:</div>
        <div class="row-tight">${chips}</div>
        <div class="text-xs text-faint mt-2">The selected config is saved as <code>data.dataset_config</code> in your YAML.</div>`;
      return;
    }

    const schema = info.schema || {};
    const compatible = info.compatible_models || [];
    const mismatch = compatible.length && !compatible.includes(bMtype);
    const detected = info.modality || 'unknown';
    const exp = info.experimental_warning ? ' <span class="badge b-warning text-xs">experimental</span>' : '';
    const cfgChip = info.config ? ` · config: <strong>${escapeHtml(info.config)}</strong>` : '';
    out.innerHTML = `
      <div class="between" style="margin-bottom:6px">
        <span><strong>${escapeHtml(detected)}</strong> dataset · ${schema.columns ? schema.columns.length : 0} columns${cfgChip}</span>
        <span class="text-xs text-muted">${info.splits ? info.splits.join(' · ') : ''}</span>
      </div>
      <div class="text-xs text-muted" style="margin-bottom:6px">
        Suggested model: <strong style="color:var(--primary)">${escapeHtml(info.suggested_model)}</strong>${exp}
        ${compatible.length ? ` · compatible: ${compatible.map(escapeHtml).join(', ')}` : ''}
      </div>
      ${mismatch ? `<div class="text-xs" style="color:var(--warning);margin-bottom:6px">⚠ Current model "${bMtype}" doesn't match this modality. Pick one of the compatible models.</div>` : ''}
      <details style="margin-top:6px">
        <summary class="text-xs text-muted" style="cursor:pointer">Schema details</summary>
        <div class="text-xs" style="margin-top:6px;font-family:var(--font-mono)">
          ${schema.image_columns?.length    ? `image: ${schema.image_columns.join(', ')}<br>`    : ''}
          ${schema.text_columns?.length     ? `text:  ${schema.text_columns.join(', ')}<br>`     : ''}
          ${schema.audio_columns?.length    ? `audio: ${schema.audio_columns.join(', ')}<br>`    : ''}
          ${schema.video_columns?.length    ? `video: ${schema.video_columns.join(', ')}<br>`    : ''}
          ${schema.label_columns?.length    ? `label: ${schema.label_columns.join(', ')}<br>`    : ''}
          ${schema.numeric_columns?.length  ? `numeric: ${schema.numeric_columns.length} cols<br>`: ''}
        </div>
      </details>`;
  } catch (e) {
    out.innerHTML = `<span style="color:var(--danger)">Inspect failed: ${escapeHtml(e.message)}</span>`;
  }
}

/* ============================================================
 * HuggingFace Dataset Browser
 * ============================================================ */
let _hfSearchTimer = null;
const _HF_PAGE_SIZE = 24;   // first page; Load more bumps this up to 50/100/200
const _HF_MAX_LIMIT = 200;  // HF Hub search caps cleanly around here
let _hfState = { q: '', modality: '', limit: _HF_PAGE_SIZE, lastQuery: null };

function openHFBrowser() {
  document.getElementById('hf-scrim').classList.add('show');
  document.getElementById('hf-drawer').classList.add('show');
  // Pre-select the modality based on the model the user has chosen, since
  // searching for "audio" datasets when the model is audio_cnn is the
  // overwhelmingly common case.
  const modSel = document.getElementById('hf-modality');
  const map = {
    audio_cnn: 'audio', cnn: 'image', transformer: 'text',
    tcn: 'time_series', rnn: 'time_series', tabular: 'tabular',
    video_cnn: 'video', mlp: '',
  };
  if (modSel && !modSel.value) modSel.value = map[bMtype] || '';
  // Show curated picks first (no network)
  hfLoadFeatured();
  setTimeout(() => document.getElementById('hf-q')?.focus(), 100);
}

function closeHFBrowser() {
  document.getElementById('hf-scrim').classList.remove('show');
  document.getElementById('hf-drawer').classList.remove('show');
}

function hfDebouncedSearch() {
  clearTimeout(_hfSearchTimer);
  _hfSearchTimer = setTimeout(hfRunSearch, 300);
}

async function hfLoadFeatured() {
  const modality = document.getElementById('hf-modality').value;
  const status = document.getElementById('hf-status');
  const out    = document.getElementById('hf-results');
  status.textContent = modality
    ? `Curated ${modality} datasets to get started — type above to search the full Hub.`
    : 'Curated picks. Type above to search the full Hub.';
  out.innerHTML = '<div class="empty"><span class="spin"></span></div>';
  try {
    const url = modality ? `/api/hf/featured?modality=${encodeURIComponent(modality)}` : '/api/hf/featured';
    const rows = await api(url);
    out.innerHTML = rows.length
      ? rows.map(r => hfCard({...r, downloads: 0, likes: 0, tags: [], modality: r.modality || modality, _featured: true})).join('')
      : `<div class="empty">No curated picks for "${escapeHtml(modality || '')}". Try the search box.</div>`;
  } catch (e) {
    out.innerHTML = `<div class="empty" style="color:var(--danger)">Failed: ${escapeHtml(e.message)}</div>`;
  }
}

async function hfRunSearch() {
  const q = document.getElementById('hf-q').value.trim();
  const modality = document.getElementById('hf-modality').value;
  if (!q) { hfLoadFeatured(); return; }
  // Reset paging state when the query or modality changes
  if (q !== _hfState.q || modality !== _hfState.modality) {
    _hfState = { q, modality, limit: _HF_PAGE_SIZE, lastQuery: null };
  }
  await _hfFetch(true);
}

async function _hfFetch(replace) {
  const status = document.getElementById('hf-status');
  const out    = document.getElementById('hf-results');
  const {q, modality, limit} = _hfState;
  status.textContent = replace
    ? `Searching the Hub for "${q}"…`
    : `Loading more (${limit} results)…`;
  if (replace) out.innerHTML = '<div class="empty"><span class="spin"></span></div>';

  const params = new URLSearchParams({q, limit: String(limit)});
  if (modality) params.set('modality', modality);
  try {
    const rows = await api('/api/hf/search?' + params.toString());
    _hfState.lastQuery = {q, modality, limit, count: rows.length};
    if (!rows.length) {
      out.innerHTML = `<div class="empty">No matches. Try removing the modality filter or a different keyword.</div>`;
      status.textContent = `0 results for "${q}".`;
      return;
    }
    out.innerHTML = rows.map(hfCard).join('') + _hfPaginationFooter(rows.length);
    const totalText = rows.length >= limit ? `${rows.length}+` : `${rows.length}`;
    status.textContent = `${totalText} result${rows.length === 1 ? '' : 's'} for "${q}"${modality ? ` · modality:${modality}` : ''}.`;
  } catch (e) {
    out.innerHTML = `<div class="empty" style="color:var(--danger)">Search failed: ${escapeHtml(e.message)}</div>`;
  }
}

function _hfPaginationFooter(currentCount) {
  // If we got back fewer rows than the limit, there's nothing more to load.
  if (currentCount < _hfState.limit) return '';
  if (_hfState.limit >= _HF_MAX_LIMIT) {
    return `<div class="text-xs text-muted text-center" style="padding:10px">
      Showing the first ${_HF_MAX_LIMIT} matches — narrow the search to find more.
    </div>`;
  }
  const next = Math.min(_HF_MAX_LIMIT, _hfState.limit * 2);
  return `<div class="row" style="justify-content:center;padding:10px">
    <button class="btn btn-secondary btn-sm" onclick="hfLoadMore()">Load more (${next - currentCount}+)</button>
  </div>`;
}

function hfLoadMore() {
  _hfState.limit = Math.min(_HF_MAX_LIMIT, _hfState.limit * 2);
  _hfFetch(false);
}

function hfCard(r) {
  const downloads = r.downloads
    ? (r.downloads >= 1000 ? `${(r.downloads/1000).toFixed(1)}k` : r.downloads)
    : '';
  const modBadge = r.modality && r.modality !== 'unknown'
    ? `<span class="badge b-primary">${escapeHtml(r.modality)}</span>` : '';
  const featured = r._featured ? `<span class="badge b-accent text-xs">curated</span>` : '';
  const gated = r.gated ? `<span class="badge b-warning text-xs" title="Gated — requires HF login">gated</span>` : '';
  return `
    <div class="hf-card">
      <div class="between" style="margin-bottom:6px">
        <div class="row-tight">
          <span class="text-mono" style="color:var(--fg-strong);font-weight:600">${escapeHtml(r.id)}</span>
          ${modBadge} ${featured} ${gated}
        </div>
        <div class="row-tight text-xs text-muted">
          ${downloads ? `↓ ${downloads}` : ''}
          ${r.likes ? ` · ❤ ${r.likes}` : ''}
        </div>
      </div>
      ${r.description ? `<div class="text-xs text-muted" style="margin-bottom:8px;line-height:1.5">${escapeHtml(r.description)}</div>` : ''}
      <div class="row-tight">
        <button class="btn btn-primary btn-xs" onclick="hfPick('${escapeHtml(r.id).replace(/'/g, "\\'")}')">Use this dataset</button>
        <button class="btn btn-ghost btn-xs" onclick="window.open('https://huggingface.co/datasets/' + '${escapeHtml(r.id).replace(/'/g, "\\'")}', '_blank')">View on Hub ↗</button>
      </div>
    </div>`;
}

function hfPick(id) {
  // Pin the dataset name into the Builder, close the drawer, and trigger
  // an inspect so the user immediately sees the schema and suggested model.
  const ds = document.getElementById('b-data-ds');
  if (ds) {
    ds.value = id;
    ds.dispatchEvent(new Event('input', {bubbles: true}));
  }
  _bSelectedConfig = null;        // a fresh dataset always starts unconfigured
  closeHFBrowser();
  setTimeout(() => bInspectDataset(null), 200);
  toast(`Pinned ${id}`, 'success', 2000);
}

function bDataChange() {
  const src = document.getElementById('b-data-source').value;
  const extra = document.getElementById('b-data-extra');
  if (src === 'csv' || src === 'numpy' || src === 'image_folder') {
    extra.innerHTML = `<div class="field"><label class="label">Path</label><input class="input input-mono" id="b-data-path" placeholder="data/file.csv" /></div>
      ${src === 'csv' ? '<div class="field"><label class="label">Target column</label><input class="input" id="b-data-target" placeholder="label" /></div>' : ''}`;
  } else if (src === 'huggingface') {
    extra.innerHTML = `
      <div class="field" style="display:flex;gap:6px;align-items:flex-end">
        <div style="flex:1"><label class="label">Dataset name</label>
          <input class="input" id="b-data-ds" placeholder="imdb" /></div>
        <button class="btn btn-secondary btn-xs" onclick="openHFBrowser()" type="button">Browse ↗</button>
        <button class="btn btn-secondary btn-xs" onclick="bInspectDataset()" type="button">Inspect</button>
      </div>
      <div id="b-data-inspect" class="hidden" style="background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:10px 12px;margin-bottom:10px;font-size:12px"></div>
      <div class="grid grid-2"><div class="field"><label class="label">Text column</label><input class="input" id="b-data-text" placeholder="auto-detect" /></div>
      <div class="field"><label class="label">Label column</label><input class="input" id="b-data-label" placeholder="auto-detect" /></div></div>`;
  } else {
    extra.innerHTML = `<div class="grid grid-3 mb-3">
      <div><label class="label">Samples</label><input class="input" id="b-syn-n" type="number" value="1000" /></div>
      <div><label class="label">Features</label><input class="input" id="b-syn-f" type="number" value="20" /></div>
      <div><label class="label">Classes</label><input class="input" id="b-syn-c" type="number" value="2" /></div>
    </div>`;
  }
  ['b-data-path','b-data-target','b-data-ds','b-data-text','b-data-label','b-syn-n','b-syn-f','b-syn-c']
    .forEach(id => { const el = document.getElementById(id); if (el) el.addEventListener('input', bRender); });
  bRender();
}
function bGetConfig() {
  const name = document.getElementById('b-name').value.trim() || 'my_experiment';
  const desc = document.getElementById('b-desc').value.trim() || null;
  const tags = (document.getElementById('b-tags').value || '').split(',').map(t => t.trim()).filter(Boolean);
  const epochs = +document.getElementById('b-epochs').value;
  const bs = +document.getElementById('b-bs').value;
  const lr = +document.getElementById('b-lr').value;
  const opt = document.getElementById('b-opt').value;
  const sched = document.getElementById('b-sched').value;
  const dev = document.getElementById('b-dev').value;

  const model = { type: bMtype, name, framework: 'pytorch' };
  if (bMtype === 'mlp') {
    const layers = [...document.querySelectorAll('#b-mlp-layers .layer-card')].map(card => {
      const f = (k) => card.querySelector(`[data-k="${k}"]`).value;
      return { size: +f('size'), activation: f('activation'), dropout: +f('dropout'), batch_norm: f('bn') === 'true' };
    });
    model.mlp = {
      input_size: +document.getElementById('b-mlp-in').value,
      hidden_layers: layers,
      output_size: +document.getElementById('b-mlp-out').value,
      output_activation: 'none',
    };
  } else if (bMtype === 'cnn') {
    const bb = document.getElementById('b-cnn-bb').value;
    const cnnCfg = {
      input_channels: +document.getElementById('b-cnn-c').value,
      input_height: +document.getElementById('b-cnn-h').value,
      input_width: +document.getElementById('b-cnn-w').value,
      output_size: +document.getElementById('b-cnn-out').value,
      pretrained: true, freeze_backbone: false,
    };
    if (bb) cnnCfg.backbone = bb;
    else cnnCfg.conv_layers = [
      { out_channels: 32, kernel_size: 3, padding: 1, pool: true, batch_norm: true },
      { out_channels: 64, kernel_size: 3, padding: 1, pool: true, batch_norm: true },
      { out_channels: 128, kernel_size: 3, padding: 1, pool: true, batch_norm: true },
    ];
    model.cnn = cnnCfg;
  } else if (bMtype === 'rnn') {
    model.rnn = {
      variant: document.getElementById('b-rnn-var').value,
      input_size: +document.getElementById('b-rnn-in').value,
      hidden_size: +document.getElementById('b-rnn-h').value,
      num_layers: +document.getElementById('b-rnn-l').value,
      bidirectional: document.getElementById('b-rnn-bi').checked,
      dropout: +document.getElementById('b-rnn-d').value,
      output_size: +document.getElementById('b-rnn-out').value,
      output_mode: 'last',
    };
  } else if (bMtype === 'transformer') {
    model.transformer = {
      vocab_size: +document.getElementById('b-tx-vocab').value,
      max_seq_len: +document.getElementById('b-tx-seq').value,
      d_model: +document.getElementById('b-tx-d').value,
      num_heads: +document.getElementById('b-tx-h').value,
      num_encoder_layers: +document.getElementById('b-tx-l').value,
      num_decoder_layers: 0,
      d_ff: +document.getElementById('b-tx-ff').value,
      dropout: 0.1,
      output_size: +document.getElementById('b-tx-out').value,
      output_mode: 'cls',
      positional_encoding: 'sinusoidal',
    };
  } else if (bMtype === 'audio_cnn') {
    const pre = document.getElementById('b-aud-pre').value.trim();
    model.audio_cnn = {
      sample_rate:    +document.getElementById('b-aud-sr').value,
      duration_secs:  +document.getElementById('b-aud-dur').value,
      use_spectrogram: document.getElementById('b-aud-spec').checked,
      n_mels:         +document.getElementById('b-aud-mels').value,
      n_fft:          +document.getElementById('b-aud-fft').value,
      hop_length:     +document.getElementById('b-aud-hop').value,
      conv_channels:  [32, 64, 128, 256],
      fc_layers:      [{ size: 128, activation: 'relu', dropout: 0.3, batch_norm: false }],
      output_size:    +document.getElementById('b-aud-out').value,
      pretrained:     pre || null,
    };
  } else if (bMtype === 'tcn') {
    const channels = document.getElementById('b-tcn-ch').value
      .split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    const mode = document.getElementById('b-tcn-mode').value;
    model.tcn = {
      input_size:  +document.getElementById('b-tcn-in').value,
      output_size: +document.getElementById('b-tcn-out').value,
      channels:    channels.length ? channels : [64, 64, 64, 64],
      kernel_size: +document.getElementById('b-tcn-k').value,
      dropout:     +document.getElementById('b-tcn-drop').value,
      output_mode: mode,
      pooling:     mode,
    };
  } else if (bMtype === 'tabular') {
    const numeric = document.getElementById('b-tab-num').value
      .split(',').map(s => s.trim()).filter(Boolean);
    const cats = document.getElementById('b-tab-cat').value
      .split('\n').map(line => line.trim()).filter(Boolean).map(line => {
        const parts = line.split(':').map(s => s.trim());
        const out = { name: parts[0], cardinality: parseInt(parts[1] || '10') };
        if (parts[2]) out.embed_dim = parseInt(parts[2]);
        return out;
      });
    model.tabular = {
      numeric_features: numeric,
      categorical_features: cats,
      output_size: +document.getElementById('b-tab-out').value,
      hidden_layers: [
        { size: 256, activation: 'relu', dropout: 0.2, batch_norm: true },
        { size: 128, activation: 'relu', dropout: 0.2, batch_norm: true },
      ],
      output_activation: 'none',
      impute_strategy: document.getElementById('b-tab-imp').value,
    };
  } else if (bMtype === 'video_cnn') {
    model.video_cnn = {
      input_channels: +document.getElementById('b-vid-c').value,
      num_frames:     +document.getElementById('b-vid-t').value,
      input_height:   +document.getElementById('b-vid-h').value,
      input_width:    +document.getElementById('b-vid-w').value,
      conv_layers: [
        { out_channels: 32,  kernel_size: 3, stride: 1, pool: true },
        { out_channels: 64,  kernel_size: 3, stride: 1, pool: true },
        { out_channels: 128, kernel_size: 3, stride: 1, pool: true },
      ],
      fc_layers: [{ size: 256, activation: 'relu', dropout: 0.5, batch_norm: false }],
      output_size: +document.getElementById('b-vid-out').value,
    };
  } else if (bMtype === 'hf_pipeline') {
    const outVal = document.getElementById('b-hf-output').value;
    model.hf_pipeline = {
      pretrained:       (document.getElementById('b-hf-pretrained').value || '').trim(),
      output_size:      outVal ? +outVal : null,
      freeze_backbone:  document.getElementById('b-hf-freeze').value === 'true',
    };
  }

  const dataSource = document.getElementById('b-data-source').value;
  const data = {
    source: dataSource,
    val_split: +document.getElementById('b-val-split').value,
    test_split: +document.getElementById('b-test-split').value,
  };
  if (dataSource === 'synthetic') {
    data.synthetic_n_samples = +document.getElementById('b-syn-n').value;
    data.synthetic_n_features = +document.getElementById('b-syn-f').value;
    data.synthetic_n_classes = +document.getElementById('b-syn-c').value;
  } else if (dataSource === 'csv' || dataSource === 'numpy' || dataSource === 'image_folder') {
    const p = document.getElementById('b-data-path');
    if (p) data.path = p.value || null;
    const t = document.getElementById('b-data-target');
    if (t && t.value) data.target_column = t.value;
  } else if (dataSource === 'huggingface') {
    const ds = document.getElementById('b-data-ds'); if (ds) data.dataset_name = ds.value || null;
    const tx = document.getElementById('b-data-text'); if (tx && tx.value) data.text_column = tx.value;
    const lb = document.getElementById('b-data-label'); if (lb && lb.value) data.label_column = lb.value;
    if (_bSelectedConfig) data.dataset_config = _bSelectedConfig;
  }

  const pipelineTask = (document.getElementById('b-task') || {}).value || null;
  return {
    name, description: desc, tags, output_dir: 'runs',
    model,
    training: {
      task: bMtype === 'transformer' ? 'text_classification' : (bMtype === 'cnn' ? 'image_classification' : 'classification'),
      pipeline_task: pipelineTask,
      loss: 'cross_entropy',
      num_epochs: epochs, batch_size: bs,
      optimizer: { type: opt, lr },
      scheduler: { type: sched, warmup_steps: sched === 'warmup_cosine' ? 100 : 0 },
      device: dev, mixed_precision: false, early_stopping_patience: 10,
    },
    data,
    deploy: { host: '0.0.0.0', port: 8080 },
  };
}
function bRender() {
  try {
    const cfg = bGetConfig();
    document.getElementById('b-yaml').textContent = yamlStringify(cfg);
  } catch (e) { document.getElementById('b-yaml').textContent = '# error: ' + e.message; }
}
async function bSave(thenTrain) {
  const cfg = bGetConfig();
  if (!cfg.name) { toast('Need an experiment name', 'error'); return; }
  try {
    const r = await apiPost('/api/configs/save', { name: cfg.name, config: cfg, overwrite: true });
    toast(`Saved to ${r.path}`, 'success');
    await loadConfigs();
    if (thenTrain) {
      navigate('train');
      setTimeout(() => {
        document.getElementById('cfg-path').value = r.path;
        document.getElementById('cfg-select').value = r.path;
        onConfigChange();
      }, 200);
    }
  } catch (e) { toast('Save failed: ' + e.message, 'error', 5000); }
}
function bReset() {
  document.getElementById('b-name').value = '';
  document.getElementById('b-desc').value = '';
  document.getElementById('b-tags').value = '';
  document.getElementById('b-mlp-layers').innerHTML = '';
  bAddMlpLayer({ size: 128, activation: 'relu', dropout: 0.2 });
  bRender();
}
function yamlStringify(obj, indent = 0) {
  const pad = '  '.repeat(indent);
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'string') return obj.match(/^[a-zA-Z0-9_\-\/\.]+$/) ? obj : `"${obj.replace(/"/g, '\\"')}"`;
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (Array.isArray(obj)) {
    if (!obj.length) return '[]';
    return obj.map(item => {
      if (typeof item === 'object' && item !== null) {
        const inner = yamlStringify(item, indent + 1);
        const lines = inner.split('\n');
        return pad + '- ' + lines[0].trimStart() + (lines.length > 1 ? '\n' + lines.slice(1).map(l => '  ' + l).join('\n') : '');
      }
      return pad + '- ' + yamlStringify(item, indent + 1);
    }).join('\n');
  }
  const lines = [];
  for (const [k, v] of Object.entries(obj)) {
    if (v === null || v === undefined) continue;
    if (typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length > 0) {
      lines.push(`${pad}${k}:`);
      lines.push(yamlStringify(v, indent + 1));
    } else if (Array.isArray(v) && v.length > 0 && typeof v[0] === 'object') {
      lines.push(`${pad}${k}:`);
      lines.push(yamlStringify(v, indent + 1));
    } else {
      lines.push(`${pad}${k}: ${yamlStringify(v, indent + 1)}`);
    }
  }
  return lines.join('\n');
}

/* EXPERIMENTS */
async function initExperiments() {
  try {
    State.experiments = await api('/api/experiments');
    renderExpTable();
    document.querySelectorAll('#exp-table th.sortable').forEach(th => th.onclick = () => sortExp(th.dataset.sort));
  } catch (e) {
    document.getElementById('exp-tbody').innerHTML = `<tr><td colspan="7" class="empty">Failed: ${escapeHtml(e.message)}</td></tr>`;
  }
}
function sortExp(col) {
  if (State.expSort.col === col) State.expSort.dir = State.expSort.dir === 'asc' ? 'desc' : 'asc';
  else { State.expSort.col = col; State.expSort.dir = 'desc'; }
  renderExpTable();
}
function renderExpTable() {
  let exps = [...State.experiments];
  const q = document.getElementById('exp-search').value.toLowerCase();
  const sf = document.getElementById('exp-status').value;
  if (q) exps = exps.filter(e => e.name.toLowerCase().includes(q) || (e.description || '').toLowerCase().includes(q));
  if (sf) exps = exps.filter(e => e.status === sf);
  const { col, dir } = State.expSort;
  exps.sort((a, b) => {
    let va = a[col], vb = b[col];
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
    return (va < vb ? -1 : va > vb ? 1 : 0) * (dir === 'asc' ? 1 : -1);
  });
  document.getElementById('exp-count').textContent = `${exps.length} of ${State.experiments.length}`;

  document.querySelectorAll('#exp-table th.sortable').forEach(th => {
    th.classList.toggle('sorted', th.dataset.sort === col);
    th.textContent = th.textContent.replace(/[↑↓ ]+$/, '');
    if (th.dataset.sort === col) th.textContent += dir === 'asc' ? ' ↑' : ' ↓';
  });

  const tbody = document.getElementById('exp-tbody');
  if (!exps.length) {
    const hasFilter = q || sf;
    tbody.innerHTML = hasFilter
      ? `<tr><td colspan="7" class="empty">No experiments match the current filter. <a href="#" onclick="document.getElementById('exp-search').value='';document.getElementById('exp-status').value='';renderExpTable();return false;" style="color:var(--primary)">Clear filters</a></td></tr>`
      : `<tr><td colspan="7" class="empty">
          <div class="empty-icon">🧪</div>
          <div style="font-size:14px;color:var(--fg);margin-bottom:6px">No experiments yet</div>
          <div class="text-xs text-muted">Build a config in the <a href="#" onclick="navigate('builder');return false;" style="color:var(--primary)">Builder</a>, then start training. <kbd class="kbd">g</kbd> <kbd class="kbd">b</kbd>.</div>
        </td></tr>`;
    return;
  }
  tbody.innerHTML = exps.map(e => `
    <tr class="clickable" onclick="openDrawer(${e.id})">
      <td><span class="badge b-muted">#${e.id}</span></td>
      <td>
        <strong>${escapeHtml(e.name)}</strong>
        ${e.description ? `<div class="text-xs text-muted truncate" style="max-width:340px">${escapeHtml(e.description)}</div>` : ''}
      </td>
      <td>${statusBadge(e.status)}</td>
      <td class="num">${e.best_val_loss != null ? fmtNum(e.best_val_loss) : '<span class="muted">—</span>'}</td>
      <td class="muted text-xs">${fmtTime(e.created_at)}</td>
      <td>${tagsList(e.tags)}</td>
      <td class="nowrap" onclick="event.stopPropagation()">
        <button class="btn btn-ghost btn-xs" onclick="openDrawer(${e.id})">Details</button>
        ${e.status === 'running' ? `<button class="btn btn-danger btn-xs" onclick="quickInterrupt(${e.id})">Stop</button>` : ''}
      </td>
    </tr>`).join('');
}
async function quickInterrupt(id) {
  if (!await confirmDialog('Mark as interrupted?')) return;
  try { await apiPost(`/api/experiments/${id}/interrupt`); toast('Marked as interrupted', 'success'); initExperiments(); }
  catch (e) { toast('Failed: ' + e.message, 'error'); }
}
async function cleanupStale() {
  try {
    const r = await apiPost('/api/train/cleanup');
    toast(r.rows_updated > 0 ? `Cleaned up ${r.rows_updated} stale run(s)` : 'No stale runs found', 'success');
    initExperiments();
    if (State.page === 'overview') initOverview();
  } catch (e) { toast('Cleanup failed: ' + e.message, 'error'); }
}

/* Drawer */
async function openDrawer(expId) {
  State.drawerExpId = expId;
  document.getElementById('scrim').classList.add('show');
  document.getElementById('drawer').classList.add('show');
  document.getElementById('dr-title').textContent = 'Loading…';
  document.getElementById('dr-sub').textContent = '';
  document.getElementById('dr-body').innerHTML = '<div class="empty"><span class="spin"></span> Loading…</div>';
  document.getElementById('dr-interrupt').classList.add('hidden');
  document.getElementById('dr-delete').classList.add('hidden');

  try {
    const [data, metrics] = await Promise.all([
      api(`/api/experiments/${expId}`),
      api(`/api/experiments/${expId}/metrics`),
    ]);
    const exp = data.experiment, runs = data.runs;
    document.getElementById('dr-title').textContent = exp.name;
    document.getElementById('dr-sub').innerHTML = `${statusBadge(exp.status)} · ${tagsList(exp.tags)}`;
    if (exp.status === 'running') document.getElementById('dr-interrupt').classList.remove('hidden');
    document.getElementById('dr-delete').classList.remove('hidden');

    const bestRun = runs.reduce((b, r) => (r.best_val_loss != null && (!b || r.best_val_loss < b.best_val_loss)) ? r : b, null);
    let html = `<div class="grid grid-3 mb-3">
      ${chipNode('Runs', runs.length)}
      ${chipNode('Best val loss', bestRun ? fmtNum(bestRun.best_val_loss) : '—')}
      ${chipNode('Best epoch', bestRun?.best_epoch ?? '—')}
    </div>`;

    if (exp.description) html += `<div class="text-sm mb-3">${escapeHtml(exp.description)}</div>`;

    html += `<div class="card-title mb-2">Runs</div>
      <div class="table-wrap mb-4"><table>
        <thead><tr><th>#</th><th>Framework</th><th>Status</th><th>Best val</th><th>Epoch</th><th>Duration</th></tr></thead>
        <tbody>${runs.length ? runs.map(r => `
          <tr>
            <td><span class="badge b-muted">#${r.run_number}</span></td>
            <td class="text-mono text-xs">${escapeHtml(r.framework || '—')}</td>
            <td>${statusBadge(r.status)}</td>
            <td class="num">${fmtNum(r.best_val_loss, 6)}</td>
            <td class="num">${r.best_epoch ?? '—'}</td>
            <td class="num">${r.duration_secs ? fmtDuration(r.duration_secs) : '—'}</td>
          </tr>`).join('') : '<tr><td colspan="6" class="muted text-center" style="padding:14px">No runs yet</td></tr>'}
        </tbody></table></div>`;

    if (metrics.length) {
      html += `<div class="card-title mb-2">Loss curves</div>
        <div class="chart-wrap"><canvas id="dr-chart"></canvas></div>`;
    }

    html += `<div class="row mt-4">
      <button class="btn btn-secondary btn-sm" onclick="navigate('curves');setTimeout(()=>{document.getElementById('cv-exp').value='${expId}';loadCurves();},120)">View full curves →</button>
      <button class="btn btn-secondary btn-sm" onclick="copyText('neural status ${expId}','Copied')">Copy CLI: neural status ${expId}</button>
    </div>`;

    document.getElementById('dr-body').innerHTML = html;
    if (metrics.length) drawDrawerChart(metrics);
  } catch (e) {
    document.getElementById('dr-body').innerHTML = `<div class="empty" style="color:var(--danger)">Error: ${escapeHtml(e.message)}</div>`;
  }
}
function drawDrawerChart(metrics) {
  const trainPts = metrics.filter(d => d.phase === 'train');
  const valPts = metrics.filter(d => d.phase === 'val');
  const epochs = [...new Set(metrics.map(d => d.epoch))].sort((a, b) => a - b);
  const ctx = document.getElementById('dr-chart').getContext('2d');
  destroyChart('dr-chart');
  State.charts['dr-chart'] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: epochs,
      datasets: [
        { label: 'Train', data: trainPts.map(d => d.metrics.loss), borderColor: '#7c83ff', backgroundColor: 'rgba(124,131,255,0.1)', tension: 0.3, pointRadius: 0, fill: false, borderWidth: 2 },
        { label: 'Val', data: valPts.map(d => d.metrics.loss), borderColor: '#34d399', backgroundColor: 'rgba(52,211,153,0.1)', tension: 0.3, pointRadius: 0, fill: false, borderWidth: 2 },
      ],
    },
    options: chartOpts(),
  });
}
function closeDrawer() {
  document.getElementById('scrim').classList.remove('show');
  document.getElementById('drawer').classList.remove('show');
  State.drawerExpId = null;
}
async function drInterrupt() {
  if (!State.drawerExpId) return;
  if (!await confirmDialog('Mark this experiment as interrupted?')) return;
  try { await apiPost(`/api/experiments/${State.drawerExpId}/interrupt`); toast('Interrupted', 'success'); initExperiments(); openDrawer(State.drawerExpId); }
  catch (e) { toast('Failed: ' + e.message, 'error'); }
}
async function drDelete() {
  if (!State.drawerExpId) return;
  if (!await confirmDialog('Permanently delete this experiment and all its runs/metrics? This cannot be undone.')) return;
  try { await apiDelete(`/api/experiments/${State.drawerExpId}`); toast('Deleted', 'success'); closeDrawer(); initExperiments(); }
  catch (e) { toast('Delete failed: ' + e.message, 'error'); }
}

/* CURVES */
async function initCurves() {
  try {
    const exps = State.experiments.length ? State.experiments : await api('/api/experiments');
    State.experiments = exps;
    const sel = document.getElementById('cv-exp');
    const prev = sel.value;
    sel.innerHTML = '<option value="">— pick an experiment —</option>' +
      exps.map(e => `<option value="${e.id}">${escapeHtml(e.name)} (#${e.id})</option>`).join('');
    if (prev) sel.value = prev;
  } catch (e) { toast('Could not load experiments: ' + e.message, 'error'); }
}
async function loadCurves() {
  const id = document.getElementById('cv-exp').value;
  if (!id) return;
  try { State.curvesData = await api(`/api/experiments/${id}/metrics`); renderCurves(); }
  catch (e) { toast('Failed: ' + e.message, 'error'); }
}
function renderCurves() {
  const metric = document.getElementById('cv-metric').value;
  const scale = document.getElementById('cv-scale').value;
  const trainPts = State.curvesData.filter(d => d.phase === 'train');
  const valPts = State.curvesData.filter(d => d.phase === 'val');
  const trainX = trainPts.map(d => d.epoch);
  const trainY = trainPts.map(d => d.metrics[metric] ?? null);
  const valX = valPts.map(d => d.epoch);
  const valY = valPts.map(d => d.metrics[metric] ?? null);
  const opts = chartOpts({ yScale: scale });
  drawLine('cv-train', trainX, [{ label: `Train ${metric}`, data: trainY, color: '#7c83ff' }], opts);
  drawLine('cv-val', valX, [{ label: `Val ${metric}`, data: valY, color: '#34d399' }], opts);
  drawLine('cv-comb', trainX, [
    { label: `Train ${metric}`, data: trainY, color: '#7c83ff' },
    { label: `Val ${metric}`, data: valY, color: '#34d399' },
  ], opts);
}
function drawLine(id, labels, datasets, opts) {
  const ctx = document.getElementById(id).getContext('2d');
  destroyChart(id);
  State.charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: datasets.map(d => ({
      label: d.label, data: d.data, borderColor: d.color,
      backgroundColor: d.color + '22', tension: 0.3, pointRadius: labels.length > 50 ? 0 : 2,
      borderWidth: 2, fill: datasets.length === 1,
    })) },
    options: opts || chartOpts(),
  });
}

/* CHECKPOINTS */
async function initCheckpoints() {
  try {
    State.checkpoints = await api('/api/checkpoints');
    const tb = document.getElementById('ck-tbody');
    if (!State.checkpoints.length) {
      tb.innerHTML = `<tr><td colspan="7" class="empty">
        <div class="empty-icon">💾</div>
        <div style="font-size:14px;color:var(--fg);margin-bottom:6px">No checkpoints saved yet</div>
        <div class="text-xs text-muted">Checkpoints appear here after a training run completes (or when <code>checkpoint_every</code> fires). Try <a href="#" onclick="navigate('train');return false;" style="color:var(--primary)">launching a run</a>.</div>
      </td></tr>`;
      return;
    }
    tb.innerHTML = State.checkpoints.map((c, i) => {
      const isBest = c.name.includes('best');
      return `<tr class="clickable" onclick="ckOpen(${i})">
        <td><strong>${escapeHtml(c.experiment)}</strong></td>
        <td><span class="badge ${isBest ? 'b-success' : 'b-accent'}">${escapeHtml(c.name)}</span></td>
        <td>${c.model_type ? `<span class="badge b-primary">${escapeHtml(c.model_type)}</span>` : '—'}</td>
        <td class="num">${c.epoch ?? '—'}</td>
        <td class="num">${c.val_loss != null ? fmtNum(c.val_loss) : '—'}</td>
        <td class="num muted">${c.size_mb} MB</td>
        <td onclick="event.stopPropagation()">
          <button class="btn btn-ghost btn-xs" onclick="ckOpen(${i})">Commands</button>
        </td>
      </tr>`;
    }).join('');
  } catch (e) { toast('Could not load checkpoints: ' + e.message, 'error'); }
}
function ckOpen(idx) {
  const c = State.checkpoints[idx];
  const cfg = c.config_path, p = c.path;
  document.getElementById('dr-title').textContent = `${c.experiment} — ${c.name}`;
  document.getElementById('dr-sub').textContent = `Epoch ${c.epoch ?? '—'} · ${c.size_mb} MB · ${c.model_type || '—'}`;
  document.getElementById('dr-interrupt').classList.add('hidden');
  document.getElementById('dr-delete').classList.add('hidden');
  document.getElementById('dr-body').innerHTML = `
    <div class="card-title mb-2">Serve as REST API</div>
    <div class="code-block mb-3">neural serve --config ${cfg} --checkpoint ${p} --port 8080</div>
    <div class="card-title mb-2">Evaluate on val set</div>
    <div class="code-block mb-3">neural evaluate --config ${cfg} --checkpoint ${p}</div>
    <div class="card-title mb-2">Export to ONNX</div>
    <div class="code-block mb-3">neural export --config ${cfg} --checkpoint ${p} --format onnx</div>
    <div class="card-title mb-2">Export to TorchScript</div>
    <div class="code-block mb-3">neural export --config ${cfg} --checkpoint ${p} --format torchscript</div>
    <div class="row mt-4">
      <button class="btn btn-secondary btn-sm" onclick="copyText('neural serve --config ${cfg} --checkpoint ${p} --port 8080','Serve command copied')">Copy serve cmd</button>
      <button class="btn btn-secondary btn-sm" onclick="navigate('predict');document.getElementById('pg-url').value='http://localhost:8080';closeDrawer();">Open Predict →</button>
    </div>`;
  document.getElementById('scrim').classList.add('show');
  document.getElementById('drawer').classList.add('show');
}

/* PREDICT */
let pgType = null, pgImageB64 = null, pgLastReq = null, pgLastRes = null, pgJsonOpen = false;

function initPredict() {
  const grid = document.getElementById('pg-mlp-grid');
  if (!grid.children.length) { for (let i = 0; i < 8; i++) pgAddMlpField(0); }
  initPredictLatencyChart();
}
async function pgConnect() {
  const url = document.getElementById('pg-url').value.trim();
  if (!url) return;
  pgSetStatus('connecting', 'Connecting…');
  try {
    await api(`/api/proxy/health?server_url=${encodeURIComponent(url)}`);
    const info = await api(`/api/proxy/info?server_url=${encodeURIComponent(url)}`);
    pgSetStatus('ok', 'Connected');
    document.getElementById('pg-run').disabled = false;
    pgShowInfo(info);
    const override = document.getElementById('pg-type-override').value;
    pgSetType(override || info.model_type || 'mlp');
    toast('Connected to inference server', 'success');
  } catch (e) {
    pgSetStatus('err', 'Cannot reach server');
    document.getElementById('pg-run').disabled = true;
    toast('Connect failed: ' + e.message, 'error', 5000);
  }
}
function pgSetStatus(state, text) {
  const dot = document.getElementById('pg-dot');
  const colors = { connecting: 'training', ok: 'online', err: 'error' };
  dot.className = 'dot ' + (colors[state] || '');
  document.getElementById('pg-status').textContent = text;
}
function pgShowInfo(info) {
  document.getElementById('pg-info-card').classList.remove('hidden');
  const raw = info.raw || info;
  const fields = [
    ['Name', info.model_name || '—'],
    ['Type', info.model_type || '—'],
    ['Framework', info.framework || '—'],
    ['Parameters', info.parameter_count != null ? Number(info.parameter_count).toLocaleString() : '—'],
    ['Device', info.device || '—'],
    ['Checkpoint', (info.checkpoint_path || '').split('/').pop() || '—'],
  ];
  let html = fields.map(([l, v]) => chipNode(l, v)).join('');
  const classNames = raw.class_names;
  if (Array.isArray(classNames) && classNames.length) {
    const preview = classNames.slice(0, 8).map(c => `<span class="badge b-muted">${escapeHtml(String(c))}</span>`).join(' ');
    const more = classNames.length > 8 ? ` <span class="text-xs text-faint">+${classNames.length - 8} more</span>` : '';
    html += `<div style="grid-column:1 / -1;background:var(--surface-2);border:1px solid var(--border);border-radius:8px;padding:10px 12px">
      <div class="text-xs text-muted mb-2" style="text-transform:uppercase;letter-spacing:0.06em">Class labels (${classNames.length})</div>
      <div class="row-tight">${preview}${more}</div>
    </div>`;
  }
  if (raw.epoch != null || raw.val_loss != null) {
    html += `<div style="grid-column:1 / -1" class="text-xs text-muted">Trained to epoch ${raw.epoch ?? '—'} · val loss ${raw.val_loss != null ? raw.val_loss.toFixed(4) : '—'}</div>`;
  }
  document.getElementById('pg-info').innerHTML = html;
}
function pgSetType(type) {
  if (!type) return;
  pgType = type;
  ['mlp-input', 'cnn-input', 'rnn-input', 'tx-input'].forEach(id => {
    const el = document.getElementById('pg-' + id); if (el) el.classList.add('hidden');
  });
  const map = { mlp: 'pg-mlp-input', cnn: 'pg-cnn-input', rnn: 'pg-rnn-input', transformer: 'pg-tx-input' };
  document.getElementById(map[type] || 'pg-mlp-input').classList.remove('hidden');
}
function pgAddMlpField(val) {
  const grid = document.getElementById('pg-mlp-grid');
  const inp = document.createElement('input');
  inp.type = 'number'; inp.step = 'any'; inp.value = val ?? 0; inp.placeholder = '0.0';
  grid.appendChild(inp);
}
function pgRandomFill() { document.querySelectorAll('#pg-mlp-grid input').forEach(el => el.value = (Math.random() * 2 - 1).toFixed(3)); }
function pgClearFields() { document.querySelectorAll('#pg-mlp-grid input').forEach(el => el.value = 0); }
function pgHandleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const r = new FileReader();
  r.onload = e => {
    pgImageB64 = e.target.result.split(',')[1];
    const img = document.getElementById('pg-img-preview');
    img.src = e.target.result; img.classList.remove('hidden');
  };
  r.readAsDataURL(file);
}
async function pgRun() {
  const url = document.getElementById('pg-url').value.trim();
  const topk = parseInt(document.getElementById('pg-topk').value);
  const btn = document.getElementById('pg-run');
  const errEl = document.getElementById('pg-error');
  errEl.classList.add('hidden');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> Running…';

  const payload = { server_url: url, top_k: topk, return_probabilities: true };
  try {
    if (pgType === 'cnn') {
      if (!pgImageB64) throw new Error('Please select an image first');
      payload.image_b64 = pgImageB64;
    } else if (pgType === 'transformer') {
      const tokens = document.getElementById('pg-tokens')?.value.trim();
      if (!tokens) throw new Error('Enter pre-tokenized IDs (comma-separated)');
      const arr = tokens.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
      if (!arr.length) throw new Error('No valid token IDs found');
      payload.tokens = arr;
    } else if (pgType === 'rnn') {
      const raw = document.getElementById('pg-rnn').value.trim();
      if (!raw) throw new Error('Enter sequence data');
      const rows = raw.split('\n').map(r => r.trim()).filter(Boolean);
      const parsed = rows.map(r => {
        const vals = r.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
        return vals.length === 1 ? vals[0] : vals;
      });
      payload.inputs = parsed;
    } else {
      payload.inputs = [...document.querySelectorAll('#pg-mlp-grid input')].map(el => parseFloat(el.value) || 0);
    }

    pgLastReq = { ...payload };
    const res = await apiPost('/api/proxy/predict', payload);
    pgLastRes = res;

    document.getElementById('pg-latency').textContent =
      `⚡ ${res.latency_ms != null ? res.latency_ms.toFixed(1) : '—'} ms server · ${res.wall_latency_ms != null ? res.wall_latency_ms.toFixed(0) : '—'} ms wall`;
    pgRenderResults(res);
    document.getElementById('pg-json-req').textContent = JSON.stringify(payload, null, 2);
    document.getElementById('pg-json-res').textContent = JSON.stringify(res.raw, null, 2);
    pushLatencyPoint(res.wall_latency_ms);
  } catch (e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5,3 19,12 5,21"/></svg> Run prediction`;
  }
}
function pgRenderResults(res) {
  const empty = document.getElementById('pg-empty');
  const bars = document.getElementById('pg-bars');
  const preds = res.predictions || [];
  if (!preds.length) {
    empty.classList.remove('hidden');
    empty.innerHTML = `<div class="empty-icon">⚠️</div><div>No predictions returned — toggle JSON to inspect.</div>`;
    bars.classList.add('hidden');
    return;
  }
  empty.classList.add('hidden');
  bars.classList.remove('hidden');
  const maxProb = Math.max(...preds.map(p => p.probability ?? 0));
  bars.innerHTML = preds.map((p, i) => {
    const prob = p.probability ?? 0;
    const pct = (prob * 100).toFixed(1);
    const w = maxProb > 0 ? (prob / maxProb * 100).toFixed(1) : 0;
    const top = i === 0;
    // Prefer the human-readable class name if the server sent one
    const human = p.class_name ?? p.label ?? '';
    const showId = (p.class_name && p.label != null && String(p.class_name) !== String(p.label));
    return `
      <div class="pred-row ${top ? 'top' : ''}">
        <div>
          <div class="pred-label">
            <span class="pred-label-rank">#${i + 1}</span>
            <span>${escapeHtml(String(human))}</span>
            ${showId ? `<span class="text-xs text-faint text-mono">[id ${p.label}]</span>` : ''}
            ${top ? '<span class="badge b-primary text-xs">top</span>' : ''}
          </div>
          <div class="pred-track"><div class="pred-fill ${top ? '' : 'muted'}" style="width:${w}%"></div></div>
        </div>
        <div class="pred-prob">${pct}%</div>
      </div>`;
  }).join('');
}
function pgToggleJson() {
  pgJsonOpen = !pgJsonOpen;
  document.getElementById('pg-json').classList.toggle('hidden', !pgJsonOpen);
}
function pgClearResults() {
  document.getElementById('pg-empty').classList.remove('hidden');
  document.getElementById('pg-empty').innerHTML = `<div class="empty-icon">🔮</div><div>Run a prediction to see results here</div>`;
  document.getElementById('pg-bars').classList.add('hidden');
  document.getElementById('pg-json').classList.add('hidden');
  document.getElementById('pg-error').classList.add('hidden');
  document.getElementById('pg-latency').textContent = '';
  pgImageB64 = null; pgLastReq = pgLastRes = null;
  const img = document.getElementById('pg-img-preview'); if (img) { img.src = ''; img.classList.add('hidden'); }
  const t = document.getElementById('pg-text'); if (t) t.value = '';
  const r = document.getElementById('pg-rnn'); if (r) r.value = '';
}
function initPredictLatencyChart() {
  destroyChart('pg-lat-chart');
  const ctx = document.getElementById('pg-lat-chart').getContext('2d');
  State.charts['pg-lat-chart'] = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Wall ms', data: [], borderColor: '#22d3ee', backgroundColor: 'rgba(34,211,238,0.15)', tension: 0.2, pointRadius: 0, fill: true, borderWidth: 2 }] },
    options: chartOpts({ noLegend: true, animation: false }),
  });
}
function pushLatencyPoint(ms) {
  if (ms == null) return;
  State.proxyLatencyHistory.push(ms);
  if (State.proxyLatencyHistory.length > 30) State.proxyLatencyHistory.shift();
  const ch = State.charts['pg-lat-chart']; if (!ch) return;
  ch.data.labels = State.proxyLatencyHistory.map((_, i) => i + 1);
  ch.data.datasets[0].data = State.proxyLatencyHistory;
  ch.update('none');
}

/* SETTINGS */
async function initSettings() {
  try {
    const h = await api('/api/health');
    document.getElementById('set-output').textContent = h.output_dir;
    document.getElementById('set-db').textContent = h.db_exists ? '✓ yes' : '— not yet';
    document.getElementById('set-ver').textContent = h.version;
    document.getElementById('set-uptime').textContent = fmtDuration(h.uptime);
  } catch {}
  try {
    const sys = await api('/api/system');
    const rows = [
      ['Hostname', sys.hostname], ['Platform', sys.platform], ['Python', sys.python],
      ['CPU cores', sys.cpu_count],
      ['Torch', sys.torch_available ? sys.torch_version : 'not installed'],
      ['GPUs', sys.gpus?.length ? sys.gpus.map(g => g.name).join(', ') : 'none'],
    ];
    document.getElementById('set-system').innerHTML = rows.map(([k, v]) => `
      <div class="between" style="padding:8px 0;border-bottom:1px solid var(--border)">
        <span class="text-xs text-muted">${k}</span>
        <span class="text-mono text-sm">${escapeHtml(String(v ?? '—'))}</span>
      </div>`).join('');
  } catch {}
}

/* Chart helpers */
function chartOpts(opts = {}) {
  return {
    responsive: true, maintainAspectRatio: false,
    animation: opts.animation === false ? false : { duration: 350 },
    plugins: {
      legend: { display: !opts.noLegend, labels: { color: '#9499a6', font: { size: 11, family: "'Inter'" }, boxWidth: 10, padding: 12 } },
      tooltip: { mode: 'index', intersect: false,
        backgroundColor: '#0c0e14', borderColor: '#2a2f42', borderWidth: 1,
        titleColor: '#ffffff', bodyColor: '#ecedf2',
        padding: 10, cornerRadius: 6,
        titleFont: { family: "'JetBrains Mono'", size: 11 }, bodyFont: { family: "'JetBrains Mono'", size: 11 } },
    },
    scales: {
      x: { ticks: { color: '#5e6473', font: { size: 10 } }, grid: { color: 'rgba(42,47,66,0.5)', drawBorder: false } },
      y: { type: opts.yScale || 'linear', ticks: { color: '#5e6473', font: { size: 10 } }, grid: { color: 'rgba(42,47,66,0.5)', drawBorder: false } },
    },
    interaction: { mode: 'nearest', axis: 'x', intersect: false },
  };
}
function destroyChart(id) {
  if (State.charts[id]) { try { State.charts[id].destroy(); } catch {} delete State.charts[id]; }
}
function pushChart(id, labels, datasets) {
  const ctx = document.getElementById(id).getContext('2d');
  if (State.charts[id]) {
    const ch = State.charts[id];
    ch.data.labels = labels;
    datasets.forEach((d, i) => {
      if (!ch.data.datasets[i]) {
        ch.data.datasets[i] = { label: d.label, data: d.data, borderColor: d.color,
          backgroundColor: d.color + '22', tension: 0.2, pointRadius: 0, fill: false, borderWidth: 2 };
      } else { ch.data.datasets[i].data = d.data; }
    });
    ch.update('none');
    return;
  }
  State.charts[id] = new Chart(ctx, {
    type: 'line',
    data: { labels, datasets: datasets.map(d => ({ label: d.label, data: d.data, borderColor: d.color, backgroundColor: d.color + '22', tension: 0.2, pointRadius: 0, fill: false, borderWidth: 2 })) },
    options: chartOpts({ animation: false }),
  });
}

/* Command palette */
const PaletteCommands = [
  { label: 'Go to Overview', icon: '📊', action: () => navigate('overview') },
  { label: 'Go to Train', icon: '▶', action: () => navigate('train') },
  { label: 'Go to Live', icon: '📡', action: () => navigate('live') },
  { label: 'Go to Builder', icon: '⚙', action: () => navigate('builder') },
  { label: 'Go to Experiments', icon: '🧪', action: () => navigate('experiments') },
  { label: 'Go to Curves', icon: '📈', action: () => navigate('curves') },
  { label: 'Go to Checkpoints', icon: '💾', action: () => navigate('checkpoints') },
  { label: 'Go to Predict', icon: '🔮', action: () => navigate('predict') },
  { label: 'Go to CLI Reference', icon: '⌨', action: () => navigate('cli') },
  { label: 'Go to Settings', icon: '⚙', action: () => navigate('settings') },
  { label: 'Refresh all data', icon: '↻', action: () => refreshAll() },
  { label: 'Clean up stale runs', icon: '🧹', action: () => cleanupStale() },
];
function openPalette() {
  document.getElementById('palette-scrim').classList.add('show');
  document.getElementById('palette').classList.add('show');
  const inp = document.getElementById('palette-search');
  inp.value = ''; inp.focus();
  renderPalette();
}
function closePalette() {
  document.getElementById('palette-scrim').classList.remove('show');
  document.getElementById('palette').classList.remove('show');
}
function renderPalette() {
  const q = document.getElementById('palette-search').value.toLowerCase();
  const items = PaletteCommands.map((c, i) => ({ c, i })).filter(({c}) => !q || c.label.toLowerCase().includes(q));
  const list = document.getElementById('palette-list');
  if (!items.length) { list.innerHTML = `<div class="palette-empty">Nothing matches "${escapeHtml(q)}"</div>`; return; }
  list.innerHTML = items.map(({c, i}, n) => `
    <div class="palette-item ${n === 0 ? 'active' : ''}" onclick="paletteRun(${i})">
      <span style="width:16px;text-align:center">${c.icon}</span>
      <span>${escapeHtml(c.label)}</span>
    </div>`).join('');
}
function palettenKey(e) {
  const list = document.getElementById('palette-list');
  const items = [...list.querySelectorAll('.palette-item')];
  const cur = items.findIndex(i => i.classList.contains('active'));
  if (e.key === 'ArrowDown') { e.preventDefault(); items[cur]?.classList.remove('active'); items[Math.min(cur + 1, items.length - 1)]?.classList.add('active'); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); items[cur]?.classList.remove('active'); items[Math.max(cur - 1, 0)]?.classList.add('active'); }
  else if (e.key === 'Enter') { e.preventDefault(); items[cur]?.click(); }
}
function paletteRun(idx) {
  const c = PaletteCommands[idx]; if (!c) return;
  closePalette(); c.action();
}

function shortcutHelp() {
  return `<strong>Keyboard shortcuts</strong>
    <div class="text-xs text-muted mt-2" style="line-height:1.7">
      <kbd class="kbd">⌘K</kbd> command palette &nbsp;
      <kbd class="kbd">r</kbd> refresh &nbsp;
      <kbd class="kbd">?</kbd> this help<br>
      <kbd class="kbd">g</kbd> then:
      <kbd class="kbd">o</kbd> overview ·
      <kbd class="kbd">t</kbd> train ·
      <kbd class="kbd">l</kbd> live ·
      <kbd class="kbd">b</kbd> builder ·
      <kbd class="kbd">e</kbd> experiments ·
      <kbd class="kbd">c</kbd> curves ·
      <kbd class="kbd">k</kbd> checkpoints ·
      <kbd class="kbd">p</kbd> predict ·
      <kbd class="kbd">s</kbd> settings
    </div>`;
}

async function refreshAll() {
  toast('Refreshing data', 'info', 1200);
  pollHealth(); pollSystemBar(); pollCounts();
  if (State.page === 'overview') initOverview();
  if (State.page === 'experiments') initExperiments();
  if (State.page === 'checkpoints') initCheckpoints();
  if (State.page === 'curves') initCurves();
}
