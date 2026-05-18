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
    if (e.key === 'Escape') { closePalette(); closeDrawer(); closeHFBrowser(); closeModelBrowser(); return; }
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
  await refreshTrainRuns();          // populate the multi-run table
  await refreshTrainLogs();
  connectTrainLogStream();
  await pollTrainStatus();
  if (State.trainPoll) clearInterval(State.trainPoll);
  State.trainPoll = setInterval(() => {
    pollTrainStatus();
    refreshTrainRuns();             // keep the runs list (and live chips) live
  }, 3000);
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
  const text = sanitizeTrainLogText(State.trainLogModel.rows.join('\n'));
  el.textContent = text || 'Subprocess output will appear here after training starts…';
  el.scrollTop = el.scrollHeight;
}
function sanitizeTrainLogText(text) {
  if (!text) return text;
  const lines = text.split('\n');
  const out = [];
  const interrupted = lines.some((ln) => ln.includes('Training interrupted by user.'));
  for (let i = 0; i < lines.length; i += 1) {
    const ln = lines[i];
    if (!interrupted) {
      out.push(ln);
      continue;
    }
    // Collapse known multiprocessing/DataLoader teardown noise after Ctrl-C.
    if (ln.startsWith('Exception ignored while calling deallocator <function _MultiProcessingDataLoaderIter.__del__')) {
      while (i < lines.length && lines[i].trim() !== '') i += 1;
      out.push('[suppressed DataLoader teardown traceback after interrupt]');
      continue;
    }
    if (ln.startsWith('Traceback (most recent call last):') && i + 1 < lines.length && lines[i + 1].includes('multiprocessing.spawn')) {
      let sawKeyboardInterrupt = false;
      while (i < lines.length) {
        if (lines[i].includes('KeyboardInterrupt')) {
          sawKeyboardInterrupt = true;
          break;
        }
        i += 1;
      }
      out.push(sawKeyboardInterrupt
        ? '[suppressed multiprocessing KeyboardInterrupt traceback after interrupt]'
        : '[suppressed multiprocessing traceback after interrupt]');
      continue;
    }
    out.push(ln);
  }
  return out.join('\n');
}
function renderTrainLogsRawFallback(rawText) {
  const el = document.getElementById('train-log');
  if (!el) return;
  if (!rawText) return;
  // If reducer output is unexpectedly empty, still show readable snapshot text.
  const stripped = rawText.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, '');
  el.textContent = sanitizeTrainLogText(stripped);
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
    // New multi-run endpoint: spawns concurrent runs without 409'ing.
    const res = await apiPost('/api/train/runs/start',
                              { config_path: path, overrides: getOverrides() });
    State.selectedTrainRunId = res.id;
    toast(`Training started: ${res.name} (PID ${res.pid})`, 'success');
    setTimeout(() => { refreshTrainRuns(); refreshTrainLogs(); }, 600);
  } catch (e) {
    // Pre-flight validation — same parsing as before.
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
    startBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5,3 19,12 5,21"/></svg> Start new run`;
    startBtn.disabled = false;
    pollTrainStatus();
  }
}

/* ----- Multi-run training: list + per-run stop + per-run log selection ----- */

State.selectedTrainRunId = null;
State.lastTrainRuns = [];

async function refreshTrainRuns() {
  const list = document.getElementById('train-runs-list');
  if (!list) return;
  try {
    const runs = await api('/api/train/runs');
    State.lastTrainRuns = runs;
    if (!runs.length) {
      list.innerHTML = '<div class="text-faint">No runs tracked yet. Start one above to see it here.</div>';
      // Mirror to live tab
      _renderLiveRunsChips([]);
      return;
    }
    // Default selection: first running, else most recent.
    if (!State.selectedTrainRunId || !runs.find(r => r.id === State.selectedTrainRunId)) {
      const first = runs.find(r => r.status === 'running') || runs[0];
      State.selectedTrainRunId = first.id;
    }
    list.innerHTML = runs.map(r => _renderTrainRunRow(r)).join('');
    // Push the picker chips into the Live tab too — same data
    _renderLiveRunsChips(runs);
    // If the currently-selected run finished, refresh logs once more
    refreshTrainLogs();
  } catch (e) {
    list.innerHTML = `<div class="text-faint">Could not load: ${escapeHtml(e.message || String(e))}</div>`;
  }
}

function _renderTrainRunRow(r) {
  const dotCls = r.status === 'running' ? 'training'
              : r.status === 'starting' ? 'training'
              : r.status === 'failed'  ? 'error'
              : r.status === 'exited'  ? 'online' : '';
  const isSel = State.selectedTrainRunId === r.id;
  const sel = isSel ? 'border:1px solid var(--accent);background:var(--bg-2)' : 'border:1px solid var(--border)';
  return `
    <div style="${sel};border-radius:6px;padding:8px;margin-bottom:6px;display:flex;align-items:center;gap:8px"
         onclick="selectTrainRun('${escapeHtml(r.id)}')" role="button">
      <span class="dot ${dotCls}"></span>
      <div style="min-width:0;flex:1;cursor:pointer">
        <div class="text-mono" style="font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escapeHtml(r.name)}</div>
        <div class="text-faint" style="font-size:11px">
          PID ${r.pid ?? '—'} · ${escapeHtml(r.status)}${r.exit_code != null ? ` (rc=${r.exit_code})` : ''} · started ${fmtAgo(r.started_at)}
        </div>
      </div>
      <button class="btn btn-ghost btn-xs" onclick="event.stopPropagation();liveSwitchToRun('${escapeHtml(r.id)}')" title="Watch in Live tab">▶ Live</button>
      ${r.status === 'running' || r.status === 'starting'
        ? `<button class="btn btn-danger btn-xs" onclick="event.stopPropagation();stopRun('${escapeHtml(r.id)}')">Stop</button>`
        : `<button class="btn btn-ghost btn-xs" onclick="event.stopPropagation();forgetRun('${escapeHtml(r.id)}')" title="Remove from list">✕</button>`}
    </div>`;
}

function selectTrainRun(runId) {
  State.selectedTrainRunId = runId;
  // Re-render so the selection highlight moves.
  refreshTrainRuns();
}

async function stopRun(runId) {
  if (!await confirmDialog('Stop this run? Kills the subprocess and all DataLoader workers.')) return;
  try {
    await apiPost(`/api/train/runs/${encodeURIComponent(runId)}/stop`);
    toast('Stop requested', 'success');
  } catch (e) {
    toast('Stop failed: ' + e.message, 'error');
  } finally {
    setTimeout(refreshTrainRuns, 500);
  }
}

async function forgetRun(runId) {
  try {
    await apiPost(`/api/train/runs/${encodeURIComponent(runId)}/forget`);
    if (State.selectedTrainRunId === runId) State.selectedTrainRunId = null;
  } catch (e) { /* swallow */ }
  refreshTrainRuns();
}

function fmtAgo(ts) {
  if (!ts) return '—';
  const sec = Math.max(0, (Date.now() / 1000) - ts);
  if (sec < 60) return `${Math.round(sec)}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

// Backwards-compat shim: stopTraining() used to stop the (single) running
// process. Now it stops the currently-selected run from the runs list.
async function stopTraining() {
  if (!State.selectedTrainRunId) {
    toast('Pick a run from the Training runs list first', 'info');
    return;
  }
  await stopRun(State.selectedTrainRunId);
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
  // Multi-run aware: when a run is selected we tail its per-run log file.
  // Falls back to the legacy shared log if no run is selected (e.g. on
  // first page load before any spawn).
  const sub = document.getElementById('train-log-sub');
  try {
    let data, label;
    if (State.selectedTrainRunId) {
      data = await api(`/api/train/runs/${encodeURIComponent(State.selectedTrainRunId)}/logs?chars=200000`);
      const r = (State.lastTrainRuns || []).find(x => x.id === State.selectedTrainRunId);
      label = r ? `Tailing ${r.name} · PID ${r.pid ?? '—'} · ${r.status}` : 'Tailing selected run';
    } else {
      data = await api('/api/train/logs?chars=200000');
      label = 'Legacy shared log (no run selected)';
    }
    if (sub) sub.textContent = label;
    initTrainLogModel();
    if (data.text) processTrainLogChunk(data.text);
    renderTrainLogs();
    if (data.text && (!State.trainLogModel || State.trainLogModel.rows.join('').trim() === '')) {
      renderTrainLogsRawFallback(data.text);
    }
  } catch (e) {
    if (sub) sub.textContent = `Log fetch failed: ${e.message || e}`;
  }
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
async function initLive() { await refreshLiveRuns(); await connectLive(); }

State.liveRunId = null;       // currently-watched run id (null = legacy shared stream)
State.lastLiveRuns = [];

async function refreshLiveRuns() {
  // Pull the canonical list from the manager. If at least one run exists
  // and we haven't picked one yet, default to the first running (or first
  // overall) so the chips strip + SSE both have a target.
  try {
    const runs = await api('/api/train/runs');
    State.lastLiveRuns = runs;
    _renderLiveRunsChips(runs);
    if (!State.liveRunId || !runs.find(r => r.id === State.liveRunId)) {
      const first = runs.find(r => r.status === 'running') || runs[0];
      if (first) State.liveRunId = first.id;
    }
  } catch (e) {
    State.lastLiveRuns = [];
  }
}

function _renderLiveRunsChips(runs) {
  const el = document.getElementById('live-runs-chips');
  if (!el) return;
  if (!runs || !runs.length) {
    el.innerHTML = '<span class="text-faint text-xs">No runs tracked. Start one in the Train tab.</span>';
    return;
  }
  el.innerHTML = runs.map(r => {
    const sel = State.liveRunId === r.id;
    const dotCls = r.status === 'running' ? 'training'
                : r.status === 'failed'  ? 'error'
                : r.status === 'exited'  ? 'online' : '';
    return `
      <button class="btn ${sel ? 'btn-primary' : 'btn-secondary'} btn-xs" type="button"
              onclick="liveSwitchToRun('${escapeHtml(r.id)}')"
              title="${escapeHtml(r.status)} · PID ${r.pid ?? '—'}">
        <span class="dot ${dotCls}" style="margin-right:4px"></span>
        ${escapeHtml(r.name)}
      </button>`;
  }).join('');
}

async function liveSwitchToRun(runId) {
  // Switch the live tab to a different run. Tear down the prior SSE
  // connection, reset chart state, then reconnect against the new run's
  // per-run stream.
  State.liveRunId = runId;
  navigate('live');     // make sure user is on the page
  if (State.liveES) { State.liveES.close(); State.liveES = null; }
  await connectLive();
  refreshLiveRuns();    // re-render chips so highlight moves
}

async function connectLive() {
  if (State.liveES) { State.liveES.close(); State.liveES = null; }
  State.liveState = newLiveState();
  initLiveCharts();
  setLiveStatus('connecting', 'Connecting…');

  // Per-run stream when a run is selected; legacy shared stream otherwise
  // (happens once on first page load before refreshLiveRuns has resolved).
  const runId = State.liveRunId;
  let snapUrl, sseUrl;
  if (runId) {
    snapUrl = `/api/train/runs/${encodeURIComponent(runId)}/events`;
    sseUrl  = `/api/train/runs/${encodeURIComponent(runId)}/events/stream`;
  } else {
    snapUrl = '/api/training/live';
    sseUrl  = '/api/events/stream';
  }

  try {
    const snap = await api(snapUrl);
    (snap.events || []).forEach(ev => applyLiveEvent(ev, false));
    if (!snap.is_running && !State.liveState.experiment) {
      const empty = document.getElementById('live-empty');
      if (empty) empty.classList.remove('hidden');
    }
  } catch {}

  const es = new EventSource(sseUrl);
  State.liveES = es;
  ['training_start','batch','epoch','checkpoint','early_stop','training_end'].forEach(t => {
    es.addEventListener(t, e => applyLiveEvent(JSON.parse(e.data), true));
  });
  es.addEventListener('connected', () => {
    const r = (State.lastLiveRuns || []).find(x => x.id === runId);
    setLiveStatus('connected', r ? `Watching ${r.name}` : 'Connected — waiting for run');
  });
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
/* ============================================================
 * Schema-driven Builder (β) — opt-in alternative to the legacy
 * per-family panels. Driven by /api/configs/schema +
 * form_builder.js. Toggle via ?builder=schema in the URL or the
 * button at the top of the Builder tab.
 * ============================================================ */

let _bSchemaHandle = null;         // form_builder render handle
let _bSchemaDescriptor = null;     // cached /api/configs/schema response
let _bSchemaConfig = {};           // current edited config

/** Flip between the legacy hand-rolled Builder and the schema-
 * driven view. Persists the choice on `window.location` so a
 * refresh keeps it. */
function bToggleBuilder(which) {
  const legacy = document.getElementById('b-legacy-root');
  const schema = document.getElementById('b-schema-root');
  const tLegacy = document.getElementById('b-toggle-legacy');
  const tSchema = document.getElementById('b-toggle-schema');
  const useSchema = which === 'schema';
  if (legacy) legacy.classList.toggle('hidden', useSchema);
  if (schema) schema.classList.toggle('hidden', !useSchema);
  if (tLegacy) tLegacy.className = useSchema ? 'btn btn-ghost btn-xs' : 'btn btn-secondary btn-xs';
  if (tSchema) tSchema.className = useSchema ? 'btn btn-secondary btn-xs' : 'btn btn-ghost btn-xs';
  // Mirror to the URL so refreshes stick.
  try {
    const url = new URL(window.location.href);
    if (useSchema) url.searchParams.set('builder', 'schema');
    else           url.searchParams.delete('builder');
    window.history.replaceState({}, '', url);
  } catch (_) {}
  if (useSchema && !_bSchemaHandle) bSchemaRender();
}

/** Fetch /api/configs/schema, mount form_builder.js, wire the
 * onChange callback to refresh the YAML preview live. */
async function bSchemaRender() {
  const host = document.getElementById('b-schema-host');
  const status = document.getElementById('b-schema-yaml-status');
  if (!host) return;
  host.innerHTML = '<div class="text-xs text-faint">Loading schema…</div>';
  if (status) status.textContent = 'fetching schema…';
  try {
    if (!_bSchemaDescriptor) {
      _bSchemaDescriptor = await api('/api/configs/schema');
    }
  } catch (e) {
    host.innerHTML = `<div class="text-xs" style="color:var(--danger)">Schema fetch failed: ${escapeHtml(e.message || String(e))}</div>`;
    return;
  }
  // Seed config with the model_type tab's currently-selected value
  // so the visible_when discriminator kicks in on first render.
  if (!_bSchemaConfig.model) _bSchemaConfig.model = { type: 'mlp' };
  _bSchemaHandle = NF_FORM.render({
    host,
    descriptor:    _bSchemaDescriptor,
    initialConfig: _bSchemaConfig,
    onChange: (path, value, cfg) => {
      _bSchemaConfig = cfg;
      _bSchemaRenderYaml();
    },
    // Per-section cross-field hooks. Each gets the group's DOM node
    // and the form-builder state — we use it for HF Pipeline's
    // 4-bit/8-bit mutual exclusion (Pydantic enforces it too, but
    // surfacing it client-side avoids round-tripping a 422).
    hooks: _bSchemaHooks(),
  });
  if (status) status.textContent = '';
  _bSchemaRenderYaml();
}

/** Hooks for sections that need cross-field behavior beyond what
 * the generic renderer provides. Each function receives the
 * group's DOM container; it can attach extra event listeners. */
function _bSchemaHooks() {
  return {
    // HF Pipeline: 4-bit and 8-bit checkboxes are mutually
    // exclusive. The Pydantic validator catches it on save, but
    // toggling client-side is friendlier.
    'model.hf_pipeline': (node) => {
      const find = (path) => node.querySelector(`.nf-field[data-path="${path}"] input[type="checkbox"]`);
      const b4 = find('model.hf_pipeline.load_in_4bit');
      const b8 = find('model.hf_pipeline.load_in_8bit');
      if (b4 && b8) {
        b4.addEventListener('change', () => { if (b4.checked) b8.checked = false; });
        b8.addEventListener('change', () => { if (b8.checked) b4.checked = false; });
      }
    },
  };
}

/** Serialize the current config to YAML in the side pane. Uses a
 * tiny inline emitter — adequate for the preview; the server's
 * /api/configs/save accepts the JSON shape directly so we don't
 * need a JS YAML library. */
function _bSchemaRenderYaml() {
  const out = document.getElementById('b-schema-yaml');
  if (!out) return;
  out.textContent = _toYaml(_bSchemaConfig);
}

/** Minimal JSON→YAML emitter. Handles strings/numbers/booleans/
 * lists/dicts/nulls — everything Pydantic emits. Not a general
 * YAML library; only used for the preview pane. */
function _toYaml(obj, indent) {
  indent = indent || 0;
  const pad = '  '.repeat(indent);
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'string') {
    // Quote strings with special chars; bare strings otherwise.
    return /[:#&*!|>'"%@`,\[\]{}\n]/.test(obj) || obj === '' ? JSON.stringify(obj) : obj;
  }
  if (typeof obj === 'number' || typeof obj === 'boolean') return String(obj);
  if (Array.isArray(obj)) {
    if (!obj.length) return '[]';
    return obj.map(v => {
      const rendered = _toYaml(v, indent + 1);
      // Inline scalars; block-render dicts/lists.
      if (typeof v === 'object' && v !== null) {
        return `${pad}-\n${rendered.split('\n').map(l => '  ' + l).join('\n')}`;
      }
      return `${pad}- ${rendered}`;
    }).join('\n');
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (!keys.length) return '{}';
    return keys.map(k => {
      const v = obj[k];
      if (v === null || typeof v !== 'object' || (Array.isArray(v) && !v.length)) {
        return `${pad}${k}: ${_toYaml(v, indent + 1)}`;
      }
      return `${pad}${k}:\n${_toYaml(v, indent + 1)}`;
    }).join('\n');
  }
  return String(obj);
}

/** Save the assembled config via /api/configs/save. Surfaces
 * Pydantic validation errors inline. */
async function bSchemaSave() {
  const valEl = document.getElementById('b-schema-validation');
  const saveBtn = document.getElementById('b-schema-save');
  if (valEl) valEl.innerHTML = '<span class="text-faint">Validating…</span>';
  if (saveBtn) saveBtn.disabled = true;
  try {
    const name = _bSchemaConfig.name;
    if (!name) {
      throw new Error('Set an experiment name first (top of the form).');
    }
    // /api/configs/save accepts {name, config, overwrite}. Server
    // runs the same Pydantic validator the CLI uses.
    const res = await apiPost('/api/configs/save', {
      name,
      config: _bSchemaConfig,
      overwrite: true,
    });
    if (valEl) {
      valEl.innerHTML = `<span style="color:var(--success)">✓ Saved to ${escapeHtml(res.path || name)}</span>`;
    }
    toast('Config saved', 'success', 3000);
  } catch (e) {
    if (valEl) {
      valEl.innerHTML = `<span style="color:var(--danger)">${escapeHtml(_pgFormatError(e))}</span>`;
    }
    toast('Save failed: ' + (e.message || e), 'error', 4500);
  } finally {
    if (saveBtn) saveBtn.disabled = false;
  }
}

/** Reset the form to defaults — re-fetches schema in case it
 * changed (e.g. a server-side Pydantic update). */
function bSchemaReset() {
  if (_bSchemaHandle) { _bSchemaHandle.destroy(); _bSchemaHandle = null; }
  _bSchemaDescriptor = null;
  _bSchemaConfig = { model: { type: 'mlp' } };
  bSchemaRender();
}

function initBuilder() {
  if (!bWired) {
    bWired = true;
    // Default the Builder view based on the URL flag — schema if
    // ?builder=schema is present, else legacy. The toggle buttons
    // let the user flip live.
    try {
      const params = new URLSearchParams(window.location.search);
      if (params.get('builder') === 'schema') bToggleBuilder('schema');
    } catch (_) {}
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
  // Browser/inspector wire-up is OUTSIDE the bWired guard so it re-attaches
  // on every initBuilder call. Each call is idempotent — we tag the element
  // with a `dataset.bound` flag and skip if already attached. This survives
  // a stale `bWired=true` from a cached app.js that didn't include the
  // browser code.
  bWireModelBrowser();
  bRender();
}

/* ----- Idempotent wire-up for the HF model inspector + browser drawer ----- */
function bWireModelBrowser() {
  // Auto-inspect on pretrained id / task / dataset changes (debounced).
  const pretrainedInput = document.getElementById('b-hf-pretrained');
  if (pretrainedInput && !pretrainedInput.dataset.boundInspect) {
    pretrainedInput.dataset.boundInspect = '1';
    pretrainedInput.addEventListener('input', _bScheduleInspect);
  }
  const taskInput = document.getElementById('b-task');
  if (taskInput && !taskInput.dataset.boundInspect) {
    taskInput.dataset.boundInspect = '1';
    taskInput.addEventListener('change', _bScheduleInspect);
  }
  const dsInput = document.getElementById('b-data-ds');
  if (dsInput && !dsInput.dataset.boundInspect) {
    dsInput.dataset.boundInspect = '1';
    dsInput.addEventListener('change', _bScheduleInspect);
  }
  // Populate the drawer's source dropdown if empty (idempotent)
  const sourceSel = document.getElementById('model-source');
  if (sourceSel && !sourceSel.options.length) bLoadModelSources();
  // Populate the drawer's task dropdown
  const taskSel = document.getElementById('model-task');
  if (taskSel && taskSel.options.length <= 1) _bPopulateTaskFilter(taskSel);
}

function _bPopulateTaskFilter(sel) {
  // Reuse the task taxonomy already loaded for the Builder.
  bLoadTasks().then(catalog => {
    const meta = (catalog && catalog.meta) || {};
    const tasks = Object.keys(meta).sort();
    if (!tasks.length) return;
    sel.innerHTML = '<option value="">Any task</option>' +
      tasks.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join('');
  }).catch(() => { /* offline — leave the default */ });
}

/* ----- Model Browser drawer (pluggable model source) ----- */

let _bSourceList = null;
let _bInspectTimer = null;
let _modelSearchTimer = null;

async function bLoadModelSources() {
  const sel = document.getElementById('model-source');
  if (!sel) return;
  try {
    _bSourceList = await api('/api/models/sources');
  } catch (e) {
    _bSourceList = [{ name: 'huggingface' }];
  }
  sel.innerHTML = _bSourceList
    .map(s => `<option value="${escapeHtml(s.name)}">${escapeHtml(s.name)}</option>`)
    .join('');
}

function openModelBrowser() {
  document.getElementById('model-scrim').classList.add('show');
  document.getElementById('model-drawer').classList.add('show');
  // Make sure the source/task selects are populated.
  bWireModelBrowser();
  // Pre-fill query from whatever the user has typed in the pretrained box
  const pretrained = (document.getElementById('b-hf-pretrained') || {}).value || '';
  const q = document.getElementById('model-q');
  if (q && !q.value) q.value = pretrained.split('/').pop() || '';
  // Pre-select the task currently chosen in the Builder, since searching
  // for models matching the current task is the common case.
  const taskSel = document.getElementById('model-task');
  const builderTask = (document.getElementById('b-task') || {}).value || '';
  if (taskSel && builderTask && !taskSel.value) {
    setTimeout(() => { taskSel.value = builderTask; modelRunSearch(); }, 50);
  } else {
    modelRunSearch();
  }
  setTimeout(() => document.getElementById('model-q')?.focus(), 100);
}

function closeModelBrowser() {
  document.getElementById('model-scrim').classList.remove('show');
  document.getElementById('model-drawer').classList.remove('show');
}

function modelDebouncedSearch() {
  clearTimeout(_modelSearchTimer);
  _modelSearchTimer = setTimeout(modelRunSearch, 300);
}

async function modelRunSearch() {
  const out = document.getElementById('model-results');
  const status = document.getElementById('model-status');
  if (!out) return;
  const q = (document.getElementById('model-q') || {}).value.trim();
  const source = (document.getElementById('model-source') || {}).value || 'huggingface';
  const task = (document.getElementById('model-task') || {}).value || '';
  const sort = (document.getElementById('model-sort') || {}).value || 'downloads';

  status.textContent = q ? `Searching ${source} for "${q}"…` : `Top ${source} models${task ? ` for ${task}` : ''}…`;
  out.innerHTML = '<div class="empty"><span class="spin"></span></div>';

  const params = new URLSearchParams({ source, sort, limit: '36' });
  if (q) params.set('q', q);
  if (task) params.set('task', task);
  try {
    const cards = await api('/api/models/search?' + params.toString());
    if (!cards.length) {
      out.innerHTML = `<div class="empty">No matches. Broaden the query or clear the task filter.</div>`;
      status.textContent = `0 results from ${source}.`;
      return;
    }
    out.innerHTML = cards.map(modelCard).join('');
    status.textContent = `${cards.length}${cards.length >= 36 ? '+' : ''} result${cards.length === 1 ? '' : 's'}${task ? ` · task:${task}` : ''}.`;
  } catch (e) {
    out.innerHTML = `<div class="empty" style="color:var(--danger)">Search failed: ${escapeHtml(e.message || String(e))}</div>`;
  }
}

function modelCard(c) {
  const downloads = c.downloads
    ? (c.downloads >= 1000 ? `${(c.downloads/1000).toFixed(1)}k` : c.downloads)
    : '';
  const tag = c.pipeline_tag ? `<span class="badge b-accent">${escapeHtml(c.pipeline_tag)}</span>` : '';
  const mod = c.modality && c.modality !== 'unknown' ? `<span class="badge b-primary">${escapeHtml(c.modality)}</span>` : '';
  const gated = c.gated ? `<span class="badge b-warning text-xs" title="Gated — needs HF token + access">gated</span>` : '';
  const lib = c.library ? `<span class="text-faint text-xs" title="library_name">${escapeHtml(c.library)}</span>` : '';
  const idEsc = escapeHtml(c.id).replace(/'/g, "\\'");
  return `
    <div class="hf-card">
      <div class="between" style="margin-bottom:6px">
        <div class="row-tight">
          <span class="text-mono" style="color:var(--fg-strong);font-weight:600">${escapeHtml(c.id)}</span>
          ${tag} ${mod} ${gated} ${lib}
        </div>
        <div class="row-tight text-xs text-muted">
          ${downloads ? `↓ ${downloads}` : ''}
          ${c.likes ? ` · ❤ ${c.likes}` : ''}
        </div>
      </div>
      ${c.description ? `<div class="text-xs text-muted" style="margin-bottom:8px;line-height:1.5">${escapeHtml(c.description)}</div>` : ''}
      <div class="row-tight">
        <button class="btn btn-primary btn-xs" onclick="modelPick('${idEsc}')">Use this model</button>
        <button class="btn btn-ghost btn-xs" onclick="window.open('https://huggingface.co/' + '${idEsc}', '_blank')">View on Hub ↗</button>
      </div>
    </div>`;
}

function modelPick(id) {
  // Pin the id into the Builder's pretrained input, close the drawer, and
  // fire the inspector so compatibility + resource fit show immediately.
  const inp = document.getElementById('b-hf-pretrained');
  if (inp) {
    inp.value = id;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
  }
  closeModelBrowser();
  setTimeout(() => bInspectCurrentModel(), 200);
  toast(`Pinned ${id}`, 'success', 2000);
}

function _bScheduleInspect() {
  if (_bInspectTimer) clearTimeout(_bInspectTimer);
  _bInspectTimer = setTimeout(() => bInspectCurrentModel(), 600);
}

async function bInspectCurrentModel() {
  const out = document.getElementById('b-hf-inspect');
  if (!out) return;
  const pretrained = (document.getElementById('b-hf-pretrained') || {}).value.trim();
  if (!pretrained || bMtype !== 'hf_pipeline') {
    out.classList.add('hidden');
    return;
  }
  const task = (document.getElementById('b-task') || {}).value || '';
  const dataset = (document.getElementById('b-data-ds') || {}).value || '';
  out.classList.remove('hidden');
  out.innerHTML = '<div class="text-muted">Inspecting model…</div>';
  try {
    const params = new URLSearchParams({ source: 'huggingface', id: pretrained });
    if (task) params.set('task', task);
    if (dataset) params.set('dataset', dataset);
    const report = await api('/api/models/inspect?' + params.toString());

    // Resource fit (best-effort)
    let fitReport = null;
    try {
      const fparams = new URLSearchParams({ source: 'huggingface', id: pretrained, purpose: 'training' });
      if (dataset) fparams.set('dataset', dataset);
      fitReport = await api('/api/models/fit?' + fparams.toString());
    } catch (_) { /* offline / unauth — skip */ }

    out.innerHTML = bRenderInspectReport(report, fitReport);
  } catch (e) {
    out.innerHTML = `<div class="text-muted">Inspect failed: ${escapeHtml(e.message || String(e))}</div>`;
  }
}

function bRenderInspectReport(report, fit) {
  const issues = (report.issues || []).map(i => {
    const cls = i.severity === 'error' ? 'b-warning' : (i.severity === 'warning' ? 'b-accent' : '');
    return `
      <div style="padding:6px 0;border-top:1px solid var(--border)">
        <span class="badge ${cls} text-xs">${escapeHtml(i.severity)}</span>
        <span style="margin-left:6px">${escapeHtml(i.message)}</span>
        ${i.hint ? `<div class="text-faint mt-1">→ ${escapeHtml(i.hint)}</div>` : ''}
      </div>`;
  }).join('');
  const okBanner = report.ok
    ? `<span class="badge text-xs" style="background:#1f3a1f;color:#9af09a">compatible</span>`
    : `<span class="badge b-warning text-xs">incompatible</span>`;
  const summary = `
    <div>
      <strong>${escapeHtml(report.model_id)}</strong>
      ${okBanner}
      <span class="text-faint" style="margin-left:6px">declared: ${escapeHtml(report.detected_pipeline || '?')}</span>
      ${report.intended_task ? `<span class="text-faint" style="margin-left:6px">intended: ${escapeHtml(report.intended_task)}</span>` : ''}
    </div>`;
  let fitBlock = '';
  if (fit && fit.estimate) {
    const est = fit.estimate;
    const host = fit.host || {};
    // Three-state badge: fits / won't fit / unknown (no params reported).
    let fits;
    if (fit.estimate_known === false) {
      fits = `<span class="badge b-warning text-xs">unknown — no params reported</span>`;
    } else if (fit.fits) {
      fits = `<span class="badge text-xs" style="background:#1f3a1f;color:#9af09a">fits</span>`;
    } else {
      fits = `<span class="badge b-warning text-xs">won't fit</span>`;
    }
    fitBlock = `
      <div class="mt-2 pt-2" style="border-top:1px solid var(--border)">
        <div><strong>Resource fit:</strong> ${fits}</div>
        <div class="text-faint mt-1">
          weights ${_h(est.model_weight_b)} · grads ${_h(est.gradients_b)} · optim ${_h(est.optimizer_b)} ·
          activations ${_h(est.activations_b)} · runtime ${_h(est.runtime_total_b)}
        </div>
        <div class="text-faint">
          download ${_h(est.download_total_b)}
          ${est.dataset_disk_b ? `(incl. dataset ${_h(est.dataset_disk_b)})` : ''}
          · host ${escapeHtml(host.accelerator || 'cpu')}
          ${host.vram_total_b ? `· VRAM ${_h(host.vram_total_b)}` : ''}
          ${host.ram_total_b ? `· RAM ${_h(host.ram_total_b)}` : ''}
          ${host.disk_free_b ? `· free disk ${_h(host.disk_free_b)}` : ''}
        </div>
      </div>`;
    if (fit.issues && fit.issues.length) {
      fitBlock += fit.issues.map(i => `
        <div style="padding:6px 0;border-top:1px solid var(--border)">
          <span class="badge ${i.severity === 'error' ? 'b-warning' : 'b-accent'} text-xs">${escapeHtml(i.severity)}</span>
          <span style="margin-left:6px">${escapeHtml(i.message || '')}</span>
          ${i.hint ? `<div class="text-faint mt-1">→ ${escapeHtml(i.hint)}</div>` : ''}
        </div>`).join('');
    }
  }
  return summary + (issues || '') + fitBlock;
}

function _h(n) {
  if (n === null || n === undefined) return '?';
  const u = ['B','KB','MB','GB','TB'];
  let i = 0, f = Number(n) || 0;
  while (f >= 1024 && i < u.length - 1) { f /= 1024; i++; }
  return f.toFixed(1) + ' ' + u[i];
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
  initPredictLatencyChart();
  ifRefresh();
}

/* ----- Managed inference servers (lifecycle UI) ----- */

async function ifRefresh() {
  const out = document.getElementById('if-list');
  if (!out) return;
  try {
    const servers = await api('/api/inference/list');
    if (!servers.length) {
      out.innerHTML = '<div class="text-faint">No managed servers running. Click <strong>Launch new ↗</strong> to start one from a saved config.</div>';
      return;
    }
    out.innerHTML = servers.map(ifRenderRow).join('');
  } catch (e) {
    out.innerHTML = `<div class="text-faint">Could not load: ${escapeHtml(e.message || String(e))}</div>`;
  }
}

function ifRenderRow(s) {
  const dotClass = s.status === 'running' ? 'online'
                : s.status === 'starting' ? 'training'
                : s.status === 'failed'   ? 'error'   : '';
  const errLine = s.last_error
    ? `<div class="text-faint mt-1">last log: ${escapeHtml(s.last_error)}</div>` : '';
  // Source badge: 'HF' for HuggingFace launches, blank for checkpoint-backed
  // servers (the existing flow). Helps the user spot which subprocesses are
  // pulling weights from the Hub vs. their own runs.
  const sourceBadge = s.source === 'huggingface'
    ? `<span class="badge b-accent text-xs" title="Launched from HuggingFace${s.model_id ? ' — ' + s.model_id : ''}">HF</span>`
    : '';
  const sourceSub = (s.source === 'huggingface' && s.model_id)
    ? `<div class="text-faint text-xs mt-1">HF: <code>${escapeHtml(s.model_id)}</code></div>`
    : '';
  return `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;padding:8px 0;border-top:1px solid var(--border)">
      <div style="min-width:0;flex:1">
        <div class="row-tight">
          <span class="dot ${dotClass}"></span>
          <span class="text-mono" style="font-weight:600">${escapeHtml(s.name)}</span>
          <span class="badge text-xs" style="margin-left:4px">${escapeHtml(s.status)}</span>
          ${s.model_type ? `<span class="badge b-accent text-xs">${escapeHtml(s.model_type)}</span>` : ''}
          ${sourceBadge}
          <span class="text-faint text-xs">port ${s.port}</span>
        </div>
        ${sourceSub}
        ${errLine}
      </div>
      <div class="row-tight">
        <button class="btn btn-secondary btn-xs" onclick="ifPickServer('${escapeHtml(s.id)}', ${s.port})">Use</button>
        <button class="btn btn-ghost btn-xs" onclick="ifStop('${escapeHtml(s.id)}')">Stop</button>
      </div>
    </div>`;
}

function ifPickServer(serverId, port) {
  // Point the connect URL at the managed server. The proxy handles auth.
  const inp = document.getElementById('pg-url');
  if (inp) {
    inp.value = `managed://${serverId}`;
    pgConnect();
  }
}

async function ifStop(serverId) {
  try {
    await apiPost(`/api/inference/${encodeURIComponent(serverId)}/stop`, {});
    toast('Stopped', 'success');
    ifRefresh();
  } catch (e) {
    toast('Stop failed: ' + (e.message || e), 'error', 4000);
  }
}

/* Currently-selected source for the Launch drawer: 'config' or 'hf'.
   Module-level so ifSwitchSource can update tab styling without juggling
   classes through dataset attributes. */
let _ifSource = 'config';

async function ifOpenLaunch() {
  document.getElementById('if-scrim').classList.add('show');
  document.getElementById('if-drawer').classList.add('show');
  // Default back to the config tab on each open so the drawer doesn't
  // remember stale state from a prior failed HF launch.
  ifSwitchSource('config');
  // Reset quantization checkboxes — leaking 4-bit between launches is
  // a surprise (the user picks a different model + accidentally
  // quantizes it because the toggle stuck on).
  ['if-hf-4bit', 'if-hf-8bit', 'if-hf-trust'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.checked = false;
  });
  const dt = document.getElementById('if-hf-bnb-dtype');
  if (dt) dt.value = '';
  // Populate the config dropdown from the existing /api/configs endpoint
  const sel = document.getElementById('if-cfg-select');
  if (sel) {
    sel.innerHTML = '<option value="">Loading…</option>';
    try {
      const cfgs = await api('/api/configs');
      sel.innerHTML = cfgs.length
        ? cfgs.map(c => `<option value="${escapeHtml(c.path)}">${escapeHtml(c.name || c.path)}</option>`).join('')
        : '<option value="">(no configs found — create one in Builder first)</option>';
    } catch (e) {
      sel.innerHTML = `<option value="">Failed to load: ${escapeHtml(e.message)}</option>`;
    }
  }
}

function ifSwitchSource(src) {
  // Toggle the segmented control + show/hide the matching pane.
  // Source values are 'config' (existing flow) and 'hf' (HuggingFace launcher).
  _ifSource = src;
  const cfgTab = document.getElementById('if-tab-config');
  const hfTab  = document.getElementById('if-tab-hf');
  const cfgPane = document.getElementById('if-pane-config');
  const hfPane  = document.getElementById('if-pane-hf');
  if (!cfgTab || !hfTab) return;
  if (src === 'hf') {
    hfTab.classList.remove('btn-ghost');
    hfTab.classList.add('btn-secondary');
    cfgTab.classList.remove('btn-secondary');
    cfgTab.classList.add('btn-ghost');
    if (cfgPane) cfgPane.classList.add('hidden');
    if (hfPane)  hfPane.classList.remove('hidden');
  } else {
    cfgTab.classList.remove('btn-ghost');
    cfgTab.classList.add('btn-secondary');
    hfTab.classList.remove('btn-secondary');
    hfTab.classList.add('btn-ghost');
    if (cfgPane) cfgPane.classList.remove('hidden');
    if (hfPane)  hfPane.classList.add('hidden');
  }
  const errEl = document.getElementById('if-launch-error');
  if (errEl) errEl.classList.add('hidden');
}

function ifCloseLaunch() {
  document.getElementById('if-scrim').classList.remove('show');
  document.getElementById('if-drawer').classList.remove('show');
}

/** Mutual-exclusion for the 4-bit / 8-bit checkboxes in the HF launch
 * drawer. Toggling one off if the user enables the other so we never
 * send both flags to the server (which would 422 the launch). */
function ifSyncQuantization(which) {
  const b4 = document.getElementById('if-hf-4bit');
  const b8 = document.getElementById('if-hf-8bit');
  if (!b4 || !b8) return;
  if (which === '4bit' && b4.checked) b8.checked = false;
  else if (which === '8bit' && b8.checked) b4.checked = false;
}

async function ifInspectHF() {
  // Run the existing model inspector before launch so the user sees task /
  // resource fit issues without burning a subprocess slot on a model that
  // would have been blocked anyway.
  const id = document.getElementById('if-hf-id').value.trim();
  const task = document.getElementById('if-hf-task').value;
  const out = document.getElementById('if-hf-inspect');
  if (!id) {
    out.classList.remove('hidden');
    out.style.color = 'var(--danger)';
    out.textContent = 'Paste an HF model id first.';
    return;
  }
  out.classList.remove('hidden');
  out.style.color = 'var(--fg-muted)';
  out.textContent = 'Inspecting…';
  try {
    // The inspect endpoint expects ?source=&id=&task= — note the param name
    // is `id`, not `model_id` (the Builder uses the same shape).
    const params = new URLSearchParams({ source: 'huggingface', id, task });
    const rep = await api(`/api/models/inspect?${params.toString()}`);
    const issues = rep.issues || [];
    const errs = issues.filter(i => i.severity === 'error');
    const warns = issues.filter(i => i.severity === 'warning');
    if (!issues.length) {
      out.style.color = 'var(--success)';
      out.innerHTML = `✓ Inspector found no issues. Task <code>${escapeHtml(task)}</code> looks compatible with <code>${escapeHtml(id)}</code>.`;
    } else {
      const fmt = (i) => `${i.severity === 'error' ? '✗' : '⚠'} <strong>${escapeHtml(i.code || i.severity)}</strong>: ${escapeHtml(i.message || '')}`;
      out.style.color = errs.length ? 'var(--danger)' : 'var(--warning)';
      out.innerHTML = [...errs, ...warns].map(fmt).join('<br>');
    }
  } catch (e) {
    out.style.color = 'var(--danger)';
    out.textContent = 'Inspect failed: ' + (e.message || String(e));
  }
}

async function ifLaunch() {
  const errEl = document.getElementById('if-launch-error');
  errEl.classList.add('hidden');
  const name = document.getElementById('if-name').value.trim() || null;

  try {
    let info;
    if (_ifSource === 'hf') {
      // HuggingFace launch — synthesizes config + spawns serve --no-checkpoint.
      const id = document.getElementById('if-hf-id').value.trim();
      const task = document.getElementById('if-hf-task').value;
      const trust = document.getElementById('if-hf-trust').checked;
      if (!id) {
        errEl.textContent = 'Paste an HF model id first.';
        errEl.classList.remove('hidden');
        return;
      }
      // Quantization knobs — mutually exclusive between 4-bit / 8-bit;
      // the Pydantic validator on the server also enforces this, but
      // we surface the error in the UI before round-tripping.
      const load4 = document.getElementById('if-hf-4bit')?.checked || false;
      const load8 = document.getElementById('if-hf-8bit')?.checked || false;
      const bnbDtype = document.getElementById('if-hf-bnb-dtype')?.value || null;
      if (load4 && load8) {
        errEl.textContent = 'load_in_4bit and load_in_8bit are mutually exclusive — pick one.';
        errEl.classList.remove('hidden');
        return;
      }
      info = await apiPost('/api/inference/start_hf', {
        hf_model_id: id,
        pipeline_task: task,
        name,
        trust_remote_code: trust,
        load_in_4bit: load4,
        load_in_8bit: load8,
        bnb_compute_dtype: bnbDtype || null,
      });
    } else {
      // Existing config-driven launch.
      const cfgPath = document.getElementById('if-cfg-select').value;
      const ckpt = document.getElementById('if-ckpt').value.trim() || null;
      if (!cfgPath) {
        errEl.textContent = 'Pick a config first.';
        errEl.classList.remove('hidden');
        return;
      }
      info = await apiPost('/api/inference/start',
                            { config_path: cfgPath, checkpoint: ckpt, name });
    }
    toast(`Launched ${info.name} on port ${info.port}`, 'success');
    ifCloseLaunch();
    setTimeout(ifRefresh, 600);
    setTimeout(ifRefresh, 2000);   // re-poll once it's likely loaded
  } catch (e) {
    errEl.textContent = e.message || String(e);
    errEl.classList.remove('hidden');
  }
}
function _pgManagedId(url) {
  // Recognize the `managed://<server_id>` form and return the id.
  // For these, we route all traffic through the dashboard's secure proxy
  // (`/api/inference/<id>/...`) — no token needed from the browser.
  // Plain `http://...` URLs go through the public proxy and may need an
  // explicit bearer token (paste in the Token field).
  if (typeof url !== 'string') return null;
  const m = url.match(/^managed:\/\/([A-Za-z0-9_-]+)$/);
  return m ? m[1] : null;
}

// In-memory only. Never persisted to localStorage / sessionStorage so a
// reload clears it and the token can't be exfiltrated by a same-origin
// scripts that read storage. The actual transmit happens via the
// Authorization header through the dashboard's proxy.
let _pgBearerToken = null;

function _pgReadToken() {
  const el = document.getElementById('pg-token');
  const v = el ? el.value.trim() : '';
  // Cache in JS variable so subsequent requests work even if the user
  // clears the field (the variable is the source of truth — we re-sync
  // from the field on each call so the user can update it).
  if (v) _pgBearerToken = v;
  return _pgBearerToken;
}

function pgClearToken() {
  _pgBearerToken = null;
  const el = document.getElementById('pg-token');
  if (el) el.value = '';
  toast('Token cleared', 'success', 1500);
}

function pgToggleToken() {
  const el = document.getElementById('pg-token');
  if (!el) return;
  el.type = el.type === 'password' ? 'text' : 'password';
}

function _pgAuthHeaders() {
  // Build the headers object used for every proxy call. The token is
  // sent ONLY in the Authorization header — never as a URL param, never
  // in JSON bodies — so it doesn't end up in browser history, server
  // access logs, or referer headers.
  const t = _pgReadToken();
  return t ? { 'Authorization': 'Bearer ' + t } : {};
}

async function pgConnect() {
  const url = document.getElementById('pg-url').value.trim();
  if (!url) return;
  pgSetStatus('connecting', 'Connecting…');
  try {
    const managed = _pgManagedId(url);
    let info;
    if (managed) {
      // Managed server — the dashboard manager attaches its own token
      // server-side; the browser doesn't need to handle one.
      info = await api(`/api/inference/${encodeURIComponent(managed)}/info`);
    } else {
      const headers = _pgAuthHeaders();
      await api(`/api/proxy/health?server_url=${encodeURIComponent(url)}`, { headers });
      info = await api(`/api/proxy/info?server_url=${encodeURIComponent(url)}`, { headers });
    }
    pgSetStatus('ok', 'Connected');
    document.getElementById('pg-run').disabled = false;
    pgShowInfo(info);
    // Stash architecture + generation defaults from /info before
    // applying the UI hint — pgApplyHint reads them to prefill
    // placeholders on max_new_tokens / temperature and to render the
    // "Model accepts up to N tokens" subtitle.
    _pgModelDefaults = info.model_defaults || null;
    // Drive the universal input panel from the server's spec hint.
    // /info now carries a ui_hint dict per the connected model's task —
    // we toggle visibility instead of switching across hardcoded panels.
    _pgServerHint = info.ui_hint || _pgFallbackHintFor(info.model_type);
    pgApplyHint(_pgServerHint);
    // A new connection always starts in auto-detect mode — clear any
    // force-override the user had selected from a prior session.
    const _typeOverrideEl = document.getElementById('pg-type-override');
    if (_typeOverrideEl) _typeOverrideEl.value = '';
    // Chat surface — visible whenever the loaded tokenizer ships a
    // chat_template. Replaces the universal input panel for these
    // models since multi-turn is the natural way to use them.
    pgSetChatMode(!!info.has_chat_template);
    // Refresh the "Recent" sidebar so users immediately see their
    // prior history for this server.
    pgHistRefresh();
    toast('Connected to inference server', 'success');
  } catch (e) {
    pgSetStatus('err', 'Cannot reach server');
    document.getElementById('pg-run').disabled = true;
    // Surface the 401 case clearly: the user knows to paste a token.
    const msg = (e.message || String(e));
    if (/401|unauthor|bearer/i.test(msg)) {
      toast('Server requires a bearer token — paste it in the Token field.',
            'error', 6000);
    } else {
      toast('Connect failed: ' + msg, 'error', 5000);
    }
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
/**
 * Universal input panel — driven by the connected model's `ui_hint`
 * (returned by /info). Applies visibility + placeholders + accept type
 * to one set of fields. The legacy `pgSetType()` per-type switch was
 * replaced with this; `pgType` still tracks the high-level model type
 * for compat with existing helpers (e.g. results rendering).
 */
function pgApplyHint(hint) {
  hint = hint || {};
  pgType = hint.primary_field || 'text';
  // Persist the hint on the module so pgRun() can consult it without
  // re-fetching /info every keystroke.
  _pgHint = hint;

  const set = (id, on) => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('hidden', !on);
  };
  set('pg-text-row',     !!hint.show_text);
  set('pg-context-row',  !!hint.show_context);
  set('pg-labels-row',   !!hint.show_candidate_labels);
  set('pg-file-row',     !!hint.show_file);
  set('pg-gen-row',      !!hint.show_generation_knobs);
  // Streaming toggle: visible only for generative tasks, where the
  // /predict/stream endpoint is meaningful. The default 'on' state is
  // intentional — for ASR / chat / summarization, watching tokens
  // arrive is dramatically better than waiting for the full response.
  set('pg-stream-toggle', !!hint.show_generation_knobs);

  // Update placeholders / labels.
  const text = document.getElementById('pg-text');
  if (text) text.placeholder = hint.text_placeholder || 'Input text';
  const drop = document.getElementById('pg-drop-label');
  if (drop) drop.textContent = hint.file_placeholder || 'Drop a file here';
  const acceptHint = document.getElementById('pg-drop-hint');
  if (acceptHint) {
    acceptHint.textContent = hint.accept
      ? `Accepts: ${hint.accept.replace(/\*/g, '*')}`
      : '';
  }
  const fileInput = document.getElementById('pg-file');
  if (fileInput && hint.accept) fileInput.accept = hint.accept;
  // Hint banner above the panel — short summary of what this model accepts.
  const banner = document.getElementById('pg-hint');
  if (banner) {
    banner.textContent = hint.summary || '';
    banner.style.display = hint.summary ? '' : 'none';
  }
  // Reset any previously-loaded file when the panel reconfigures —
  // wrong b64 type would otherwise leak between connections.
  pgImageB64 = null;
  _pgAudioB64 = null;
  _pgVideoB64 = null;
  _pgFileMime = null;
  const img = document.getElementById('pg-img-preview');
  if (img) { img.src = ''; img.classList.add('hidden'); }
  const fileInfo = document.getElementById('pg-file-info');
  if (fileInfo) { fileInfo.textContent = ''; fileInfo.classList.add('hidden'); }

  // Prefill generation-knob placeholders + length hint from
  // info.model_defaults. We touch `placeholder`, never `value` — a
  // user-typed override must not be clobbered when we reconnect.
  _pgApplyModelDefaults();
}

/** Apply info.model_defaults to the universal panel: placeholder
 *  values on generation knobs, and a "model accepts up to N tokens"
 *  subtitle under the text input when max_position_embeddings is known.
 */
function _pgApplyModelDefaults() {
  const d = _pgModelDefaults || {};
  const gen = d.generation || {};

  const setPh = (id, v) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.placeholder = (v != null) ? `(default: ${v})` : '(model default)';
  };
  setPh('pg-gen-max', gen.max_new_tokens != null ? gen.max_new_tokens : gen.max_length);
  setPh('pg-gen-temp', gen.temperature);

  // Length hint under the text input. Approximate token count as
  // chars/4 — wrong for non-English / heavy punctuation but matches
  // what users see in HF playgrounds, and it's a *hint*, not a hard
  // gate. Server-side runtime errors stay the source of truth for
  // overflow detection (different tokenizers, different ratios).
  const hint = document.getElementById('pg-text-hint');
  if (hint) {
    const maxTok = d.max_position_embeddings;
    if (maxTok && maxTok > 0) {
      hint.classList.remove('hidden');
      hint.dataset.maxTokens = String(maxTok);
      hint.textContent = `Model accepts up to ${maxTok.toLocaleString()} tokens (≈${Math.round(maxTok * 4).toLocaleString()} characters)`;
      // Re-run length check in case the user already typed before
      // reconnecting to a smaller-context model.
      pgValidateLength();
    } else {
      hint.classList.add('hidden');
      hint.textContent = '';
      delete hint.dataset.maxTokens;
    }
  }
}

/** Soft length warning: when the text input crosses the model's
 *  approximate token budget, paint the hint amber so the user notices
 *  before they hit a runtime tokenizer error. Token count is estimated
 *  via chars/4 — rough but matches HF playground heuristics. */
function pgValidateLength() {
  const hint = document.getElementById('pg-text-hint');
  if (!hint || !hint.dataset.maxTokens) return;
  const maxTok = parseInt(hint.dataset.maxTokens, 10);
  if (!maxTok) return;
  const text = (document.getElementById('pg-text')?.value || '');
  const ctx  = (document.getElementById('pg-context')?.value || '');
  const approxTokens = Math.ceil((text.length + ctx.length) / 4);
  if (approxTokens > maxTok) {
    hint.textContent = `⚠ Input ≈${approxTokens.toLocaleString()} tokens exceeds the model's ${maxTok.toLocaleString()}-token context. Tokenizer will truncate.`;
    hint.style.color = 'var(--warning, #f59e0b)';
  } else {
    hint.textContent = `Model accepts up to ${maxTok.toLocaleString()} tokens (≈${Math.round(maxTok * 4).toLocaleString()} characters)`;
    hint.style.color = '';
  }
}

/**
 * Fallback hint for older inference servers (or non-HF model types)
 * that don't return a ui_hint. Uses the model_type to pick a sensible
 * default panel layout — same logic the server-side _native_ui_hint()
 * uses, mirrored here so the UI doesn't break if /info doesn't ship
 * the hint yet (rolling deploys, custom forks).
 */
function _pgFallbackHintFor(modelType) {
  const t = modelType || 'mlp';
  const base = {
    show_text: false, show_context: false, show_file: false,
    show_candidate_labels: false, show_generation_knobs: false,
    accept: '', primary_field: 'inputs',
    text_placeholder: 'Input value',
    file_placeholder: 'Drop a file here',
    summary: '',
  };
  if (t === 'mlp' || t === 'tabular' || t === 'tcn' || t === 'rnn') {
    return { ...base, show_text: true, primary_field: 'inputs',
             text_placeholder: 'Numeric features (JSON list or comma-separated)' };
  }
  if (t === 'cnn') return { ...base, show_file: true, accept: 'image/*',
                             primary_field: 'image_b64',
                             file_placeholder: 'Drop an image' };
  if (t === 'audio_cnn') return { ...base, show_file: true, accept: 'audio/*',
                                    primary_field: 'audio_b64',
                                    file_placeholder: 'Drop an audio file' };
  if (t === 'video_cnn') return { ...base, show_file: true, accept: 'video/*',
                                    primary_field: 'video_b64',
                                    file_placeholder: 'Drop a video file' };
  if (t === 'transformer') return { ...base, show_text: true, primary_field: 'text',
                                      text_placeholder: 'Type the text to classify' };
  if (t === 'hf_pipeline') return { ...base, show_text: true, show_file: true,
                                      accept: 'image/*,audio/*,video/*',
                                      primary_field: 'text',
                                      text_placeholder: 'Type input or drop a file' };
  return { ...base, show_text: true, primary_field: 'text' };
}

/**
 * Universal file handler. Detects the MIME type of the dropped/picked
 * file and routes the base64 payload into the right *_b64 field on
 * the request. Single drop zone, three possible destinations — image,
 * audio, video. The server-side decoder picks the right pipeline based
 * on the connected model's task; this function just makes sure the
 * right field is populated.
 */
function pgHandleFile(file) {
  if (!file) return;
  const mime = file.type || '';
  const r = new FileReader();
  r.onload = e => {
    const dataUrl = e.target.result;
    const b64 = dataUrl.split(',')[1] || '';
    // Reset all binary fields, then route by MIME family.
    pgImageB64 = null; _pgAudioB64 = null; _pgVideoB64 = null;
    _pgFileMime = mime;
    if (mime.startsWith('image/')) {
      pgImageB64 = b64;
      const img = document.getElementById('pg-img-preview');
      if (img) { img.src = dataUrl; img.classList.remove('hidden'); }
    } else if (mime.startsWith('audio/')) {
      _pgAudioB64 = b64;
    } else if (mime.startsWith('video/')) {
      _pgVideoB64 = b64;
    } else {
      // Unknown MIME — best-effort guess by file extension.
      const ext = (file.name.split('.').pop() || '').toLowerCase();
      const audioExts = ['wav', 'flac', 'mp3', 'ogg', 'm4a', 'opus'];
      const videoExts = ['mp4', 'mov', 'avi', 'mkv', 'webm'];
      const imageExts = ['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'];
      if (audioExts.includes(ext))      _pgAudioB64 = b64;
      else if (videoExts.includes(ext)) _pgVideoB64 = b64;
      else if (imageExts.includes(ext)) {
        pgImageB64 = b64;
        const img = document.getElementById('pg-img-preview');
        if (img) { img.src = dataUrl; img.classList.remove('hidden'); }
      } else {
        toast(`Unrecognized file type "${ext}". Sending as raw image_b64.`, 'warn', 4000);
        pgImageB64 = b64;
      }
    }
    // Surface what was loaded — file size + type, so users see something
    // happened. Important for audio/video where there's no preview.
    const info = document.getElementById('pg-file-info');
    if (info) {
      const kb = (file.size / 1024).toFixed(1);
      info.textContent = `${file.name} · ${mime || ext || '?'} · ${kb} KB`;
      info.classList.remove('hidden');
    }
  };
  r.readAsDataURL(file);
}

// Module-level state for the universal input panel.
let _pgHint       = null;   // last hint object applied (drives pgRun's payload assembly)
let _pgServerHint = null;   // hint returned by /info on last connect — used to restore auto-detect
let _pgAudioB64 = null;   // base64 of a dropped audio file
let _pgVideoB64 = null;   // base64 of a dropped video file
let _pgFileMime = null;   // MIME hint forwarded with the b64 field
let _pgStreamCtl = null;  // AbortController for an in-flight streaming request
let _pgModelDefaults = null;   // info.model_defaults — drives prefills + length hint

// Chat-mode state (only used when info.has_chat_template is true).
// _pgChatMessages is the in-memory transcript; clearing it is "+ New chat".
// _pgChatPendingImage is set by pgChatAttach and consumed on the next send.
let _pgChatMessages = [];
let _pgChatPendingImage = null;
let _pgChatActive = false;

// Legacy MLP grid helpers — kept as no-ops so any cached HTML calling
// them doesn't throw. The new universal panel doesn't render the grid.

// Synthetic hint objects for "Force input type" dropdown overrides.
// Each entry drives pgApplyHint exactly as a server /info ui_hint would.
const _OVERRIDE_HINTS = {
  mlp:        { primary_field: 'inputs',    show_text: false, show_file: false,
                show_generation_knobs: false,
                summary: 'Override: MLP / Tabular — supply raw feature vector in the Inputs field' },
  cnn:        { primary_field: 'image_b64', show_text: false, show_file: true,
                show_generation_knobs: false,
                accept: 'image/*', file_placeholder: 'Drop an image',
                summary: 'Override: CNN / Image' },
  rnn:        { primary_field: 'inputs',    show_text: true,  show_file: false,
                show_generation_knobs: false,
                text_placeholder: 'Comma-separated sequence values',
                summary: 'Override: RNN / Sequence' },
  transformer:{ primary_field: 'text',      show_text: true,  show_file: false,
                show_generation_knobs: false,
                text_placeholder: 'Input text',
                summary: 'Override: Transformer / Text' },
  audio_cnn:  { primary_field: 'audio_b64', show_text: false, show_file: true,
                show_generation_knobs: false,
                accept: 'audio/*', file_placeholder: 'Drop an audio file',
                summary: 'Override: Audio CNN' },
  tcn:        { primary_field: 'inputs',    show_text: true,  show_file: false,
                show_generation_knobs: false,
                text_placeholder: 'Comma-separated sequence values',
                summary: 'Override: TCN / Sequence' },
  tabular:    { primary_field: 'inputs',    show_text: false, show_file: false,
                show_generation_knobs: false,
                summary: 'Override: Tabular (named features) — supply raw inputs JSON' },
  video_cnn:  { primary_field: 'video_b64', show_text: false, show_file: true,
                show_generation_knobs: false,
                accept: 'video/*', file_placeholder: 'Drop a video file',
                summary: 'Override: Video CNN' },
  hf_pipeline:{ primary_field: 'text',      show_text: true,  show_file: false,
                show_generation_knobs: true,
                text_placeholder: 'Input text',
                summary: 'Override: HF Pipeline (universal)' },
};

function pgSetType(type) {
  if (!type) {
    // '' = Auto-detect: restore whatever the last pgConnect() returned.
    if (_pgServerHint) pgApplyHint(_pgServerHint);
    return;
  }
  const override = _OVERRIDE_HINTS[type];
  if (override) pgApplyHint(override);
}
function pgAddMlpField(val) { /* noop */ }
function pgRandomFill()    { /* noop */ }
function pgClearFields()   { /* noop */ }
async function pgRun() {
  const url = document.getElementById('pg-url').value.trim();
  const topk = parseInt(document.getElementById('pg-topk').value);
  const btn = document.getElementById('pg-run');
  const errEl = document.getElementById('pg-error');
  errEl.classList.add('hidden');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> Running…';

  const hint = _pgHint || {};
  const payload = { server_url: url, top_k: topk, return_probabilities: true };
  try {
    // Universal payload assembly — read whatever the visible fields hold
    // and pack them into the request shape the server already understands.
    // Server-side spec dispatch + decoders pick the right path. This is
    // the same code regardless of whether the model is text / image /
    // audio / video / multimodal.
    const text = document.getElementById('pg-text')?.value.trim() || '';
    const context = document.getElementById('pg-context')?.value.trim() || '';
    const labels = document.getElementById('pg-labels')?.value.trim() || '';
    if (text)     payload.text = text;
    if (context)  payload.context = context;
    if (labels) {
      const arr = labels.split(',').map(s => s.trim()).filter(Boolean);
      if (arr.length) payload.candidate_labels = arr;
    }
    if (pgImageB64)   payload.image_b64 = pgImageB64;
    if (_pgAudioB64)  payload.audio_b64 = _pgAudioB64;
    if (_pgVideoB64)  payload.video_b64 = _pgVideoB64;
    if (_pgFileMime)  payload.file_mime = _pgFileMime;

    // Generation knobs (only meaningful for tasks where show_generation_knobs
    // is true; we still let users send them even when hidden — the server
    // ignores extras for non-generative tasks).
    const gMax  = parseInt(document.getElementById('pg-gen-max')?.value);
    const gTemp = parseFloat(document.getElementById('pg-gen-temp')?.value);
    const gSamp = document.getElementById('pg-gen-sample')?.checked;
    if (!isNaN(gMax))  payload.max_new_tokens = gMax;
    if (!isNaN(gTemp)) payload.temperature = gTemp;
    if (gSamp != null) payload.do_sample = !!gSamp;

    // Advanced escape hatch — raw `inputs` and pre-tokenized `tokens`.
    // These take precedence over the universal fields when populated,
    // for users who need exact control (RNN sequences, MLP feature
    // vectors, pre-tokenized text).
    const rawInputs = document.getElementById('pg-raw-inputs')?.value.trim();
    const rawTokens = document.getElementById('pg-raw-tokens')?.value.trim();
    if (rawInputs) {
      try { payload.inputs = JSON.parse(rawInputs); }
      catch (_) {
        // Tolerate comma-separated floats as a convenience.
        const arr = rawInputs.split(/[\s,]+/).map(parseFloat).filter(v => !isNaN(v));
        if (!arr.length) throw new Error('Raw `inputs` must be valid JSON or a comma-separated list of numbers.');
        payload.inputs = arr;
      }
    }
    if (rawTokens) {
      try { payload.tokens = JSON.parse(rawTokens); }
      catch (_) {
        const arr = rawTokens.split(/[\s,]+/).map(s => parseInt(s)).filter(n => !isNaN(n));
        if (!arr.length) throw new Error('Pre-tokenized IDs must be valid JSON or a comma-separated list of integers.');
        payload.tokens = arr;
      }
    }

    // Sanity: make sure we actually have something to send. The hint
    // tells us the primary field; complain if it's missing AND no
    // fallback fields were supplied.
    const hasContent = !!(payload.text || payload.image_b64
        || payload.audio_b64 || payload.video_b64
        || payload.inputs !== undefined || payload.tokens
        || payload.candidate_labels);
    if (!hasContent) {
      const need = hint.primary_field || 'an input';
      const human = {
        text: 'text', image_b64: 'an image', audio_b64: 'an audio file',
        video_b64: 'a video file', inputs: 'raw inputs',
      }[need] || need;
      throw new Error(`Provide ${human} before running.`);
    }

    pgLastReq = { ...payload };
    // For managed servers, route through the dashboard's secure proxy so
    // the per-server bearer token never reaches the browser. The body is
    // the same PredictRequest shape, minus `server_url` (which is implied
    // by the path). For external servers, attach the token via the
    // Authorization header (never in the body / URL).
    const managedId = _pgManagedId(url);

    // Streaming branch — generative tasks with the toggle on. SSE pumps
    // tokens into _pgRenderGeneratedText incrementally and the regular
    // bar-chart path is skipped. Falls back to the normal /predict on
    // any pre-flight failure (non-generative task → 400).
    const wantsStream = (
      hint.show_generation_knobs
      && document.getElementById('pg-stream')?.checked
    );
    if (wantsStream) {
      const ok = await pgRunStream({ payload, url, managedId, btn });
      if (ok) return;   // streamed successfully; finally-block restores button
      // fall through to the regular path on stream failure
    }

    let res;
    if (managedId) {
      const t0 = performance.now();
      const raw = await apiPost(
        `/api/inference/${encodeURIComponent(managedId)}/predict`,
        { ...payload, server_url: undefined }
      );
      const wall = performance.now() - t0;
      res = {
        ...raw,
        wall_latency_ms: wall,
        latency_ms: raw.latency_ms,
        raw,
      };
    } else {
      res = await api('/api/proxy/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', ..._pgAuthHeaders() },
        body: JSON.stringify(payload),
      });
    }
    pgLastRes = res;

    document.getElementById('pg-latency').textContent =
      `⚡ ${res.latency_ms != null ? res.latency_ms.toFixed(1) : '—'} ms server · ${res.wall_latency_ms != null ? res.wall_latency_ms.toFixed(0) : '—'} ms wall`;
    pgRenderResults(res);
    document.getElementById('pg-json-req').textContent = JSON.stringify(payload, null, 2);
    document.getElementById('pg-json-res').textContent = JSON.stringify(res.raw, null, 2);
    pushLatencyPoint(res.wall_latency_ms);
    // Refresh the "Recent" sidebar — the server logged this call.
    pgHistRefresh();
  } catch (e) {
    errEl.textContent = _pgFormatError(e);
    errEl.classList.remove('hidden');
    // Also surface the request payload so the user can see exactly what
    // was sent — the previous "input: []" 422s were impossible to debug
    // without this.
    try {
      document.getElementById('pg-json-req').textContent = JSON.stringify(payload, null, 2);
    } catch (_) {}
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-with="2.5"><polygon points="5,3 19,12 5,21"/></svg> Run prediction`;
    // Hide the Stop button + drop the abort controller — only relevant
    // while a stream was in flight, but cheap to do unconditionally.
    const stopBtn = document.getElementById('pg-stop');
    if (stopBtn) stopBtn.classList.add('hidden');
    _pgStreamCtl = null;
  }
}

/**
 * Stream tokens from a generative model via SSE. Returns true on a
 * clean stream, false if the caller should fall back to /predict.
 * Handles cancellation via _pgStreamCtl (the Stop button).
 */
async function pgRunStream({ payload, url, managedId, btn }) {
  const stopBtn = document.getElementById('pg-stop');
  const empty   = document.getElementById('pg-empty');
  const richTxt = document.getElementById('pg-rich-text');
  const errEl   = document.getElementById('pg-error');

  // Pre-clear the UI: hide everything else, show the text pane that
  // will receive incremental updates.
  ['pg-bars', 'pg-rich-image', 'pg-rich-spans'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
  });
  empty.classList.add('hidden');
  richTxt.classList.remove('hidden');
  richTxt.textContent = '';
  if (stopBtn) stopBtn.classList.remove('hidden');

  _pgStreamCtl = new AbortController();
  const t0 = performance.now();
  let acc = '';   // accumulated decoded text
  let finalPayload = null;

  // Pick the URL: managed servers route through the dashboard's
  // streaming proxy (token attached server-side). External servers go
  // through their own /predict/stream with the bearer header.
  let endpoint, init;
  if (managedId) {
    endpoint = `/api/inference/${encodeURIComponent(managedId)}/predict/stream`;
    init = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body:    JSON.stringify({ ...payload, server_url: undefined }),
      signal:  _pgStreamCtl.signal,
    };
  } else {
    // External-server streaming routes through the dashboard's SSE proxy
    // so the bearer token stays server-side and CORS is not an issue.
    endpoint = '/api/proxy/predict/stream';
    init = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body:    JSON.stringify(payload),   // payload already contains server_url
      signal:  _pgStreamCtl.signal,
    };
  }

  let resp;
  try {
    resp = await fetch(endpoint, init);
  } catch (e) {
    if (e.name === 'AbortError') {
      richTxt.textContent += '\n\n[stopped]';
      return true;   // stream was cancelled cleanly; nothing to render via /predict
    }
    return false;    // network-level failure → caller falls back to /predict
  }
  if (!resp.ok || !resp.body) {
    return false;    // server rejected (e.g. 400 because task isn't generative)
  }

  // SSE is a text/event-stream of `event: …\ndata: …\n\n` records.
  // We don't pull in a library — a small line-buffered parser is enough.
  const reader  = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let evType = '', evData = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Split off complete events on the blank-line delimiter.
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        evType = ''; evData = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) evType = line.slice(6).trim();
          else if (line.startsWith('data:')) {
            evData += (evData ? '\n' : '') + line.slice(5).trim();
          }
        }
        if (evType === 'token' && evData) {
          // tokens come as JSON-encoded strings
          let piece = '';
          try { piece = JSON.parse(evData); }
          catch (_) { piece = evData; }
          acc += piece;
          richTxt.textContent = acc;
        } else if (evType === 'done') {
          try { finalPayload = JSON.parse(evData); }
          catch (_) {}
        } else if (evType === 'error') {
          let detail = evData;
          try { detail = JSON.parse(evData).detail || detail; }
          catch (_) {}
          errEl.textContent = detail;
          errEl.classList.remove('hidden');
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      errEl.textContent = 'Stream interrupted: ' + (e.message || String(e));
      errEl.classList.remove('hidden');
    }
  }

  const wall = performance.now() - t0;
  document.getElementById('pg-latency').textContent =
    `⚡ ${finalPayload?.latency_ms != null ? finalPayload.latency_ms.toFixed(1) : '—'} ms server · ${wall.toFixed(0)} ms wall`;
  pushLatencyPoint(wall);
  pgLastRes = {
    predictions: [{ class_name: acc, label: 0, probability: null,
                     score: null, metadata: null }],
    result_kind: 'generated_text',
    latency_ms:  finalPayload?.latency_ms ?? null,
    wall_latency_ms: wall,
    raw: finalPayload,
  };
  document.getElementById('pg-json-req').textContent = JSON.stringify(payload, null, 2);
  document.getElementById('pg-json-res').textContent = JSON.stringify(finalPayload || { tokens: acc }, null, 2);
  return true;
}

function pgStopStream() {
  if (_pgStreamCtl) {
    try { _pgStreamCtl.abort(); }
    catch (_) {}
    _pgStreamCtl = null;
  }
}

/* ----- Recent predictions sidebar ----- */

/**
 * Returns the currently-connected server id (for managed servers) or
 * the raw connect URL (external). Used to filter history.
 */
function _pgCurrentServerKey() {
  const url = document.getElementById('pg-url')?.value.trim();
  if (!url) return null;
  return _pgManagedId(url) || url;
}

async function pgHistRefresh() {
  const key = _pgCurrentServerKey();
  if (!key) return;
  const list  = document.getElementById('pg-hist-list');
  const empty = document.getElementById('pg-hist-empty');
  if (!list || !empty) return;
  try {
    const params = new URLSearchParams();
    // Only filter by server_id for managed servers — external connect
    // URLs aren't unique keys (port + host can repeat across runs).
    if (_pgManagedId(document.getElementById('pg-url').value.trim())) {
      params.set('server_id', key);
    }
    params.set('limit', '50');
    const rows = await api(`/api/predict/history?${params.toString()}`);
    if (!rows.length) {
      list.classList.add('hidden');
      empty.classList.remove('hidden');
      return;
    }
    empty.classList.add('hidden');
    list.classList.remove('hidden');
    list.innerHTML = rows.map(_pgHistRow).join('');
  } catch (e) {
    empty.textContent = 'Could not load history: ' + (e.message || e);
  }
}

function _pgHistRow(r) {
  // Build a short summary string: the response's first prediction's
  // class_name, or a placeholder when generated text is empty.
  const preds = (r.response && r.response.predictions) || [];
  const top = preds[0] || {};
  const label = top.class_name || top.label || '(no result)';
  const labelShort = String(label).length > 60
    ? String(label).slice(0, 57) + '…'
    : String(label);
  const ts = (r.created_at || '').replace('T', ' ').slice(0, 19);
  const kind = r.result_kind || 'logits';
  const lat = r.latency_ms != null ? `${Number(r.latency_ms).toFixed(0)} ms` : '—';

  // Distinguish input modality with a small emoji prefix so users can
  // scan the list without reading every row.
  const req = r.request || {};
  const modIcon =
      req.audio_b64 ? '🎵 '
    : req.video_b64 ? '🎬 '
    : req.image_b64 ? '🖼️ '
    : (req.messages && req.messages.length) ? '💬 '
    : '';

  return `
    <div class="hist-row" onclick="pgHistReplay(${r.id})"
         style="display:flex;justify-content:space-between;gap:8px;padding:6px 8px;border:1px solid var(--border);border-radius:6px;cursor:pointer;background:var(--bg-elev)">
      <div style="min-width:0;flex:1">
        <div class="text-xs" style="font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
          ${modIcon}${escapeHtml(labelShort)}
        </div>
        <div class="text-xs text-faint">
          <span class="badge b-muted">${escapeHtml(kind)}</span>
          <span class="ml-1">${lat}</span>
          <span class="text-faint" style="margin-left:6px">${escapeHtml(ts)}</span>
        </div>
      </div>
      <button class="btn btn-icon btn-ghost btn-xs" onclick="event.stopPropagation();pgHistDelete(${r.id})" title="Forget this row">✕</button>
    </div>`;
}

async function pgHistReplay(id) {
  // Fetch the row from the cache rendered above (the bulk list is
  // already in memory) — simplest is to re-fetch by id, but the list
  // endpoint already returned the row we need. Re-pull to be safe.
  try {
    const rows = await api(`/api/predict/history?limit=200`);
    const row = rows.find(r => r.id === id);
    if (!row) {
      toast('History entry not found', 'error', 3000);
      return;
    }
    const req = row.request || {};
    // Reset the universal panel and re-apply text-shaped fields. We
    // intentionally don't re-attach b64 blobs (they were stripped on
    // storage); the user can re-drop the file if they want a true rerun.
    const set = (id, v) => {
      const el = document.getElementById(id);
      if (el) el.value = v ?? '';
    };
    set('pg-text', req.text);
    set('pg-context', req.context);
    set('pg-labels', (req.candidate_labels || []).join(', '));
    set('pg-gen-max', req.max_new_tokens);
    set('pg-gen-temp', req.temperature);
    const samp = document.getElementById('pg-gen-sample');
    if (samp && req.do_sample != null) samp.checked = !!req.do_sample;
    // Note when a binary attachment is required to re-run accurately.
    const had = (req.audio_b64 || '').startsWith('<stripped:') ? 'audio'
              : (req.video_b64 || '').startsWith('<stripped:') ? 'video'
              : (req.image_b64 || '').startsWith('<stripped:') ? 'image'
              : null;
    if (had) {
      toast(`Loaded text fields. Re-attach an ${had} file to rerun exactly.`,
             'warn', 5000);
    } else {
      toast('Replay loaded — edit and Run.', 'success', 2500);
    }
  } catch (e) {
    toast('Replay failed: ' + (e.message || e), 'error', 4000);
  }
}

async function pgHistDelete(id) {
  try {
    await api(`/api/predict/history/${id}`, { method: 'DELETE' });
    pgHistRefresh();
  } catch (e) {
    toast('Delete failed: ' + (e.message || e), 'error', 3000);
  }
}

async function pgHistClear() {
  if (!confirm('Clear history for this server?')) return;
  const key = _pgCurrentServerKey();
  if (!key) return;
  try {
    const params = new URLSearchParams();
    if (_pgManagedId(document.getElementById('pg-url').value.trim())) {
      params.set('server_id', key);
    }
    await api(`/api/predict/history?${params.toString()}`, { method: 'DELETE' });
    pgHistRefresh();
  } catch (e) {
    toast('Clear failed: ' + (e.message || e), 'error', 3000);
  }
}

/* ----- Chat mode (multi-turn for any-to-any / chat-template models) ----- */

/**
 * Toggle chat mode on/off. When on, the chat transcript pane is
 * shown and the regular universal input is hidden. The Run button
 * is also hidden — chat sends via the chat composer's enter key.
 */
function pgSetChatMode(on) {
  _pgChatActive = !!on;
  const chat   = document.getElementById('pg-chat');
  const univ   = document.getElementById('pg-universal');
  const runBtn = document.getElementById('pg-run');
  if (chat)   chat.classList.toggle('hidden', !on);
  if (univ)   univ.classList.toggle('hidden', !!on);
  if (runBtn) runBtn.classList.toggle('hidden', !!on);
  if (on) {
    pgChatNew();
  } else {
    _pgChatMessages = [];
    _pgChatPendingImage = null;
  }
}

/** Reset the chat history (used on connect + by the New Chat button). */
function pgChatNew() {
  _pgChatMessages = [];
  _pgChatPendingImage = null;
  _pgChatRender();
  pgClearResults();
  const att = document.getElementById('pg-chat-attach');
  if (att) att.textContent = '';
}

/** Stash a base64 image to attach to the next user message. */
function pgChatAttach(file) {
  if (!file) return;
  const r = new FileReader();
  r.onload = e => {
    _pgChatPendingImage = (e.target.result || '').split(',')[1] || null;
    const att = document.getElementById('pg-chat-attach');
    if (att) att.textContent = `📎 ${file.name} attached to next message`;
  };
  r.readAsDataURL(file);
}

/** Send the current composer text (+ optional pending image) as a
 * user turn, then stream the assistant's response into a placeholder
 * bubble that updates in place. */
async function pgChatSend() {
  const ta = document.getElementById('pg-chat-input');
  const txt = (ta?.value || '').trim();
  if (!txt && !_pgChatPendingImage) return;

  // Build the user message — single text part, or text + image when an
  // attachment is pending.
  const userMsg = { role: 'user' };
  if (_pgChatPendingImage) {
    userMsg.content = [
      { type: 'image', image_b64: _pgChatPendingImage },
      ...(txt ? [{ type: 'text', text: txt }] : []),
    ];
  } else {
    userMsg.content = txt;
  }
  _pgChatMessages.push(userMsg);

  // Clear composer state immediately — the user expects the bubble to
  // appear and the box to empty.
  if (ta) ta.value = '';
  _pgChatPendingImage = null;
  const att = document.getElementById('pg-chat-attach');
  if (att) att.textContent = '';

  // Push a placeholder assistant message that the streamer fills in.
  const assistantMsg = { role: 'assistant', content: '' };
  _pgChatMessages.push(assistantMsg);
  _pgChatRender();

  // Stream the response.
  const url = document.getElementById('pg-url').value.trim();
  const managedId = _pgManagedId(url);
  const payload = {
    server_url: url,
    messages: _pgChatMessages.slice(0, -1),  // exclude the placeholder
    return_probabilities: false,
  };
  // Forward generation knobs from the universal panel even when hidden;
  // makes them tunable for chat too.
  const gMax  = parseInt(document.getElementById('pg-gen-max')?.value);
  const gTemp = parseFloat(document.getElementById('pg-gen-temp')?.value);
  const gSamp = document.getElementById('pg-gen-sample')?.checked;
  if (!isNaN(gMax))  payload.max_new_tokens = gMax;
  if (!isNaN(gTemp)) payload.temperature = gTemp;
  if (gSamp != null) payload.do_sample = !!gSamp;

  const stopBtn = document.getElementById('pg-stop');
  if (stopBtn) stopBtn.classList.remove('hidden');
  _pgStreamCtl = new AbortController();

  let endpoint, init;
  if (managedId) {
    endpoint = `/api/inference/${encodeURIComponent(managedId)}/predict/stream`;
    init = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json',
                  'Accept': 'text/event-stream' },
      body: JSON.stringify({ ...payload, server_url: undefined }),
      signal: _pgStreamCtl.signal,
    };
  } else {
    // Route through the dashboard's SSE proxy — keeps the bearer token
    // server-side and avoids CORS issues with external inference servers.
    endpoint = '/api/proxy/predict/stream';
    init = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
      body:    JSON.stringify(payload),   // payload already contains server_url
      signal:  _pgStreamCtl.signal,
    };
  }

  try {
    const resp = await fetch(endpoint, init);
    if (!resp.ok || !resp.body) {
      assistantMsg.content = `(server returned ${resp.status})`;
      _pgChatRender();
      return;
    }
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        let evType = '', evData = '';
        for (const line of block.split('\n')) {
          if (line.startsWith('event:'))      evType = line.slice(6).trim();
          else if (line.startsWith('data:')) evData += (evData ? '\n' : '') + line.slice(5).trim();
        }
        if (evType === 'token' && evData) {
          let piece = '';
          try { piece = JSON.parse(evData); } catch (_) { piece = evData; }
          assistantMsg.content += piece;
          _pgChatRender();
        } else if (evType === 'error') {
          let detail = evData;
          try { detail = JSON.parse(evData).detail || detail; } catch (_) {}
          assistantMsg.content = `(error) ${detail}`;
          _pgChatRender();
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      assistantMsg.content += `\n[stream interrupted: ${e.message || e}]`;
      _pgChatRender();
    }
  } finally {
    if (stopBtn) stopBtn.classList.add('hidden');
    _pgStreamCtl = null;
    // Refresh the Recent sidebar — the chat turn was logged server-side
    // by the streaming proxy.
    pgHistRefresh();
  }
}

/** Render the chat transcript into #pg-chat-transcript. */
function _pgChatRender() {
  const host = document.getElementById('pg-chat-transcript');
  if (!host) return;
  const bubble = (m, idx) => {
    const role = m.role || 'user';
    const isUser = role === 'user';
    const isSystem = role === 'system';
    const align = isUser ? 'flex-end' : 'flex-start';
    const bg = isUser ? 'var(--accent-soft)'
              : isSystem ? 'transparent'
              : 'var(--surface-2)';
    const border = isSystem ? '1px dashed var(--border)' : '1px solid var(--border)';
    let body;
    if (typeof m.content === 'string') {
      body = escapeHtml(m.content) || '<span class="text-faint">…</span>';
    } else if (Array.isArray(m.content)) {
      body = m.content.map(part => {
        if (part.type === 'text')
          return escapeHtml(part.text || '');
        if (part.type === 'image' && part.image_b64)
          return `<img src="data:image/*;base64,${part.image_b64}" style="max-width:200px;border-radius:6px;display:block;margin:4px 0">`;
        if (part.type === 'audio')
          return `<span class="text-xs text-faint">🎵 audio attached</span>`;
        return '';
      }).join('');
    } else {
      body = '';
    }
    return `
      <div style="align-self:${align};max-width:80%;background:${bg};border:${border};border-radius:10px;padding:8px 12px">
        <div class="text-xs text-faint" style="text-transform:uppercase;letter-spacing:0.04em">${role}</div>
        <div style="white-space:pre-wrap;line-height:1.45;font-size:14px">${body}</div>
      </div>`;
  };
  host.innerHTML = _pgChatMessages.map(bubble).join('');
  // Auto-scroll to the bottom on each render so streaming tokens stay visible.
  host.scrollTop = host.scrollHeight;
  const count = document.getElementById('pg-chat-count');
  if (count) count.textContent = `${_pgChatMessages.length} message(s)`;
}

function _pgFormatError(e) {
  // FastAPI/Pydantic errors surface as JSON in `e.message`. Render them as
  // a readable bullet list instead of dumping raw JSON like
  // `[{"type":"float_type","loc":[...],"input":[]}]` (which is what bit
  // the user). Falls back to the raw message for non-JSON errors.
  const raw = (e && e.message) || String(e);
  let parsed = null;
  try { parsed = JSON.parse(raw); } catch (_) { return raw; }
  const arr = Array.isArray(parsed) ? parsed
            : (parsed && Array.isArray(parsed.detail) ? parsed.detail : null);
  if (arr) {
    return arr.slice(0, 12).map(d => {
      const loc = Array.isArray(d.loc)
        ? d.loc.filter(x => x !== 'body').join('.')
        : '?';
      const got = d.input !== undefined
        ? `  (got: ${JSON.stringify(d.input).slice(0, 80)})` : '';
      return `• ${loc}: ${d.msg}${got}`;
    }).join('\n');
  }
  if (parsed && parsed.detail) return String(parsed.detail);
  return raw;
}
function pgRenderResults(res) {
  // Reset all renderer panes; the dispatch below re-shows whichever
  // ones are relevant for this result_kind.
  const empty   = document.getElementById('pg-empty');
  const bars    = document.getElementById('pg-bars');
  const richImg = document.getElementById('pg-rich-image');
  const richTxt = document.getElementById('pg-rich-text');
  const richSpn = document.getElementById('pg-rich-spans');
  [bars, richImg, richTxt, richSpn].forEach(el => {
    if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
  });

  const preds = res.predictions || [];
  if (!preds.length) {
    empty.classList.remove('hidden');
    empty.innerHTML = `<div class="empty-icon">⚠️</div><div>No predictions returned — toggle JSON to inspect.</div>`;
    return;
  }
  empty.classList.add('hidden');

  // Dispatch on result_kind so each task renders the most useful surface.
  // Older servers (pre-0.4.5) won't ship result_kind — default to "logits".
  const kind = res.result_kind || 'logits';
  switch (kind) {
    case 'boxes':         _pgRenderBoxes(preds);             break;
    case 'depth':         _pgRenderDepth(preds);             break;
    case 'masks':         _pgRenderMasks(preds);             break;
    case 'qa_spans':      _pgRenderQA(preds);                break;
    case 'generated_text':_pgRenderGeneratedText(preds);     break;
    case 'token_spans':   _pgRenderTokenSpans(preds);        break;
    case 'embedding':     _pgRenderEmbedding(preds);         break;
    default:              _pgRenderLogitsBars(preds);
  }
}

/**
 * Embedding renderer — feature-extraction / sentence-similarity /
 * any model whose pipeline_task resolves to the AutoModel fallback.
 * The server already pooled across tokens and shipped dim + L2 +
 * preview, so we render a small "info card" with a sparkline of the
 * first N dims. Full vector isn't on the wire — for now it's a
 * preview + stats; if users want the raw bytes we can add a
 * /predict/embed endpoint later.
 */
function _pgRenderEmbedding(preds) {
  const host = document.getElementById('pg-rich-text');
  host.classList.remove('hidden');
  const p = preds[0] || {};
  const md = p.metadata || {};
  const dim = md.dim;
  const preview = md.preview || [];
  const stats = md.stats || {};
  // Render a small SVG sparkline of the first N dims. Scaling: map
  // [stats.min, stats.max] to the vertical extent of the chart.
  const W = 360, H = 60, pad = 4;
  const lo = stats.min ?? Math.min(...preview, 0);
  const hi = stats.max ?? Math.max(...preview, 1);
  const span = (hi - lo) || 1;
  const pts = preview.map((v, i) => {
    const x = pad + (preview.length <= 1 ? 0 :
                      (i / (preview.length - 1)) * (W - 2 * pad));
    const y = H - pad - ((v - lo) / span) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const zeroY = H - pad - ((0 - lo) / span) * (H - 2 * pad);
  host.innerHTML = `
    <div class="text-xs text-muted mb-2" style="text-transform:uppercase;letter-spacing:0.06em">Embedding vector</div>
    <div class="row-tight mb-2" style="gap:14px">
      <span class="text-mono"><strong>${dim ?? '—'}</strong>-dim</span>
      <span class="text-mono">‖v‖₂ = ${md.l2_norm != null ? Number(md.l2_norm).toFixed(4) : '—'}</span>
      <span class="text-faint text-xs">min ${stats.min ?? '—'} · max ${stats.max ?? '—'} · μ ${stats.mean ?? '—'} · σ ${stats.std ?? '—'}</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:${W}px;height:${H}px;background:var(--bg-elev);border:1px solid var(--border);border-radius:6px">
      <line x1="0" y1="${zeroY}" x2="${W}" y2="${zeroY}" stroke="var(--border)" stroke-dasharray="2 3" />
      <polyline points="${pts}" fill="none" stroke="var(--accent, #22d3ee)" stroke-width="1.5" />
      ${preview.map((v, i) => {
        const x = pad + (preview.length <= 1 ? 0 :
                          (i / (preview.length - 1)) * (W - 2 * pad));
        const y = H - pad - ((v - lo) / span) * (H - 2 * pad);
        return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2" fill="var(--accent, #22d3ee)" />`;
      }).join('')}
    </svg>
    <div class="text-xs text-faint mt-2">Showing first ${preview.length} dim(s). Full vector not in response — use a downstream endpoint for raw embeddings.</div>`;
}

/* ----- renderers — one per result_kind ----- */

function _pgRenderLogitsBars(preds) {
  const bars = document.getElementById('pg-bars');
  bars.classList.remove('hidden');
  const maxProb = Math.max(...preds.map(p => p.probability ?? 0));
  bars.innerHTML = preds.map((p, i) => {
    const prob = p.probability ?? 0;
    const pct = (prob * 100).toFixed(1);
    const w = maxProb > 0 ? (prob / maxProb * 100).toFixed(1) : 0;
    const top = i === 0;
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

/**
 * Object-detection: overlay each predicted bbox on the image the user
 * dropped. Boxes come back in xyxy_norm coordinates (0..1) so scaling
 * to the rendered <img> is just `coord * naturalWidth`.
 */
function _pgRenderBoxes(preds) {
  const host = document.getElementById('pg-rich-image');
  const bars = document.getElementById('pg-bars');
  bars.classList.remove('hidden');   // also show top-K bar list alongside
  host.classList.remove('hidden');

  const palette = ['#22d3ee', '#f59e0b', '#a78bfa', '#34d399',
                   '#f87171', '#fb923c', '#60a5fa', '#fbbf24'];

  // Render image + box overlay on a stacked canvas. We use the
  // user-dropped image (still in pgImageB64) as the base; if it's
  // missing the model didn't actually have an image input.
  if (!pgImageB64) {
    host.innerHTML = `<div class="text-xs text-faint">No image preview available — drop an image and re-run to see overlays.</div>`;
  } else {
    host.innerHTML = `
      <div style="position:relative;display:inline-block;max-width:100%">
        <img id="pg-box-img" src="data:image/*;base64,${pgImageB64}"
             style="max-width:100%;display:block;border-radius:6px" />
        <canvas id="pg-box-canvas" style="position:absolute;top:0;left:0;pointer-events:none"></canvas>
      </div>
      <div class="text-xs text-faint mt-2">${preds.length} detection(s) · top score ${
        preds[0]?.probability != null
          ? (preds[0].probability * 100).toFixed(1) + '%'
          : '—'}</div>`;
    const img = document.getElementById('pg-box-img');
    const canvas = document.getElementById('pg-box-canvas');
    const draw = () => {
      const w = img.clientWidth, h = img.clientHeight;
      // Use displayed (CSS) size for canvas drawing surface so coords
      // align with what the user sees, regardless of the natural res.
      canvas.width = w; canvas.height = h;
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
      const ctx = canvas.getContext('2d');
      ctx.lineWidth = 2;
      ctx.font = 'bold 12px ui-sans-serif, system-ui, -apple-system, sans-serif';
      preds.forEach((p, i) => {
        const md = p.metadata || {};
        const bb = md.bbox;
        if (!Array.isArray(bb) || bb.length !== 4) return;
        const color = palette[i % palette.length];
        const [x1, y1, x2, y2] = bb;
        const px = x1 * w, py = y1 * h;
        const pw = (x2 - x1) * w, ph = (y2 - y1) * h;
        ctx.strokeStyle = color;
        ctx.strokeRect(px, py, pw, ph);
        // Label with score, on a small filled background.
        const score = p.probability != null ? (p.probability * 100).toFixed(0) + '%' : '';
        const label = `${p.class_name || p.label} ${score}`.trim();
        const tw = ctx.measureText(label).width + 8;
        ctx.fillStyle = color;
        ctx.fillRect(px, Math.max(0, py - 16), tw, 16);
        ctx.fillStyle = '#0d0e10';
        ctx.fillText(label, px + 4, Math.max(12, py - 4));
      });
    };
    if (img.complete) draw(); else img.addEventListener('load', draw);
    window.addEventListener('resize', draw, { once: false });
  }

  // Also show the top-K bar list — useful when the boxes are small or
  // overlap; the bars give a clear ranking by score.
  _pgRenderLogitsBars(preds);
}

/**
 * Depth estimation: render the colormapped PNG thumbnail the server
 * shipped, plus the depth-stat summary as a caption.
 */
function _pgRenderDepth(preds) {
  const host = document.getElementById('pg-rich-image');
  host.classList.remove('hidden');
  const p = preds[0] || {};
  const md = p.metadata || {};
  if (!md.image_b64) {
    host.innerHTML = `<div class="text-xs text-faint">${escapeHtml(p.class_name || '(no depth output)')}</div>`;
    return;
  }
  const minVal = md.min ?? '—';
  const maxVal = md.max ?? '—';
  host.innerHTML = `
    <div style="text-align:center">
      <img src="data:${escapeHtml(md.image_mime || 'image/png')};base64,${md.image_b64}"
           alt="depth map" style="max-width:100%;border-radius:6px" />
      <div class="text-xs text-faint mt-2">
        ${escapeHtml(String(p.class_name || ''))}<br>
        Range [${minVal}, ${maxVal}] · viridis colormap (closer = brighter)
      </div>
    </div>`;
}

/**
 * Segmentation: gallery of per-class mask thumbnails. Each carries a
 * coverage percentage so users can spot which classes are present.
 */
function _pgRenderMasks(preds) {
  const host = document.getElementById('pg-rich-image');
  host.classList.remove('hidden');
  if (!preds.length || !preds[0].metadata?.image_b64) {
    host.innerHTML = `<div class="text-xs text-faint">No mask thumbnails returned.</div>`;
    return;
  }
  const cards = preds.map((p) => {
    const md = p.metadata || {};
    const cov = md.coverage != null ? (md.coverage * 100).toFixed(1) + '%' : '';
    return `
      <div style="border:1px solid var(--border);border-radius:6px;padding:6px;background:var(--bg-elev);text-align:center">
        <img src="data:${escapeHtml(md.image_mime || 'image/png')};base64,${md.image_b64}"
             style="max-width:100%;border-radius:4px;image-rendering:pixelated" />
        <div class="text-xs mt-1">${escapeHtml(p.class_name || ('class_' + p.label))}</div>
        <div class="text-xs text-faint">${cov}</div>
      </div>`;
  }).join('');
  host.innerHTML = `
    <div class="text-xs text-muted mb-2">Top ${preds.length} class mask(s) by area</div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px">${cards}</div>`;
}

/**
 * Question-answering: highlight the answer span in the user's context
 * (when present) plus show the decoded text + confidence.
 */
function _pgRenderQA(preds) {
  const host = document.getElementById('pg-rich-text');
  host.classList.remove('hidden');
  const p = preds[0] || {};
  const md = p.metadata || {};
  const answer = p.class_name || '(no answer)';
  const conf = p.probability != null ? (p.probability * 100).toFixed(1) + '%' : '—';

  // If the user supplied context, highlight the decoded answer inside
  // it. Otherwise just show the answer.
  const contextEl = document.getElementById('pg-context');
  const context = contextEl ? contextEl.value : '';
  let body;
  if (context && answer && answer !== '(no answer)') {
    const idx = context.toLowerCase().indexOf(answer.toLowerCase());
    if (idx >= 0) {
      body = escapeHtml(context.slice(0, idx))
        + `<mark style="background:var(--accent-soft);color:var(--fg)">`
        +   escapeHtml(context.slice(idx, idx + answer.length))
        + `</mark>`
        + escapeHtml(context.slice(idx + answer.length));
    } else {
      body = escapeHtml(context);
    }
  } else {
    body = `<strong>${escapeHtml(answer)}</strong>`;
  }
  host.innerHTML = `
    <div class="text-xs text-muted mb-2">Answer · confidence ${conf}
      ${md.start_idx != null ? ` · tokens ${md.start_idx}–${md.end_idx}` : ''}</div>
    <div>${body}</div>`;
}

/** ASR / summarization / translation / image-to-text / chat — render the
 * decoded text in a roomy card. */
function _pgRenderGeneratedText(preds) {
  const host = document.getElementById('pg-rich-text');
  host.classList.remove('hidden');
  const text = preds.map(p => p.class_name || '').filter(Boolean).join('\n\n')
                || '(empty generation)';
  host.textContent = text;
}

/** Token-classification (NER, etc.) — list each tagged span with
 * its label and confidence. */
function _pgRenderTokenSpans(preds) {
  const host = document.getElementById('pg-rich-spans');
  host.classList.remove('hidden');
  const palette = ['#22d3ee', '#f59e0b', '#a78bfa', '#34d399',
                   '#f87171', '#fb923c', '#60a5fa', '#fbbf24'];
  // Group consecutive tokens with the same class — that's roughly how
  // BIO tags would render. Loose grouping; fine for a preview.
  const items = preds.map((p, i) => {
    const color = palette[(p.label ?? 0) % palette.length];
    const conf = p.probability != null ? (p.probability * 100).toFixed(0) + '%' : '';
    const tok = p.metadata?.token || `tok#${p.metadata?.token_idx ?? '?'}`;
    return `
      <div style="display:inline-flex;gap:6px;align-items:center;padding:4px 8px;border-radius:6px;border:1px solid ${color};background:rgba(255,255,255,0.02);margin:2px">
        <span class="text-mono text-xs">${escapeHtml(String(tok))}</span>
        <span class="text-xs" style="color:${color};font-weight:600">${escapeHtml(p.class_name)}</span>
        <span class="text-xs text-faint">${conf}</span>
      </div>`;
  }).join('');
  host.innerHTML = `
    <div class="text-xs text-muted mb-2">${preds.length} tagged token(s)</div>
    <div>${items}</div>`;
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
  // Clear the structured-output renderers (boxes/depth/masks/qa/spans).
  ['pg-rich-image', 'pg-rich-text', 'pg-rich-spans'].forEach(id => {
    const el = document.getElementById(id);
    if (el) { el.classList.add('hidden'); el.innerHTML = ''; }
  });
  // Clear universal-input state — text + context + labels + every b64 field.
  pgImageB64 = null; _pgAudioB64 = null; _pgVideoB64 = null;
  _pgFileMime = null; pgLastReq = pgLastRes = null;
  const img = document.getElementById('pg-img-preview');
  if (img) { img.src = ''; img.classList.add('hidden'); }
  const fileInfo = document.getElementById('pg-file-info');
  if (fileInfo) { fileInfo.textContent = ''; fileInfo.classList.add('hidden'); }
  ['pg-text', 'pg-context', 'pg-labels',
   'pg-gen-max', 'pg-gen-temp',
   'pg-raw-inputs', 'pg-raw-tokens'].forEach(id => {
    const el = document.getElementById(id); if (el) el.value = '';
  });
  const samp = document.getElementById('pg-gen-sample');
  if (samp) samp.checked = false;
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
