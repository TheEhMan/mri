/* webapp_v3 — script.js (3-section viewer + Q&A) */
"use strict";

// ─── Theme Toggle ─────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const toggleBtn = document.getElementById('theme-toggle');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      document.body.classList.toggle('light-theme');
    });
  }
});

// ─── Q&A Delays ───────────────────────────────────────────────────────────────
const RESPONSE_DELAY = { volume:1000, region:1500, morphology:2800, clinical:2400 };
function getDelay(cat) { return Math.max(700, (RESPONSE_DELAY[cat]||1500) + Math.floor(Math.random()*400)-150); }

// ─── Questions ────────────────────────────────────────────────────────────────
const QUESTIONS = {
  volume:[
    {id:'volume_total',     label:'What is the total tumor volume?'},
    {id:'volume_necrotic',  label:'What is the volume of the necrotic core?'},
    {id:'volume_edema',     label:'What is the volume of peritumoral edema?'},
    {id:'volume_enhancing', label:'What is the volume of the enhancing tumor?'},
    {id:'volume_diameter',  label:'What is the maximum diameter of the tumor?'},
  ],
  region:[
    {id:'region_location',   label:'Where is the tumor located?'},
    {id:'region_enhancing',  label:'Which regions are affected by the enhancing tumor?'},
    {id:'region_edema',      label:'Which regions are affected by edema?'},
    {id:'region_hemisphere', label:'Which hemisphere is the tumor primarily in?'},
    {id:'region_midline',    label:'Does the tumor cross the midline?'},
    {id:'region_eloquent',   label:'Is the tumor near any eloquent cortex?'},
    {id:'region_motor',      label:'Is the tumor near the motor cortex?'},
    {id:'region_language',   label:"Is the tumor near Broca's or Wernicke's area?"},
  ],
  morphology:[
    {id:'morph_margins', label:'Does the tumor have well-defined or infiltrative margins?'},
    {id:'morph_mass',    label:'Is there significant mass effect?'},
    {id:'morph_hetero',  label:'How heterogeneous is the tumor signal on T1c?'},
    {id:'morph_cystic',  label:'Is there evidence of cystic degeneration?'},
  ],
  clinical:[
    {id:'clin_location', label:'Is this concerning or manageable given the location?'},
    {id:'clin_deficits', label:'What functional deficits might this patient be experiencing?'},
  ],
};
const CAT_LABELS = { volume:'Volume', region:'Region / Anatomy', morphology:'Shape / Morphology', clinical:'Clinical Implications' };

// ─── NIfTI state — per-panel ──────────────────────────────────────────────────
const VIEWS = ['axial','coronal','sagittal'];
const panelState = {
  axial:    { slice: 0, max: 100 },
  coronal:  { slice: 0, max: 100 },
  sagittal: { slice: 0, max: 100 },
};
let activeView    = 'axial';   // which panel the slider controls
let sliceInfo     = null;
let sliceCache    = {};
let autoPlayTimer = null;

// ─── App state ────────────────────────────────────────────────────────────────
let currentPid    = null;
let currentImages = {};
let askedIds      = new Set();
let isAnswering   = false;

// ─── DOM ──────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ─── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSubjects();
  setupNiftiCells();
  setupSlider();
  setupRegSliders();
  setupCategoryButtons();
  $('back-to-categories').addEventListener('click', showCategories);
});


// ─── Subjects ────────────────────────────────────────────────────────────────
async function loadSubjects() {
  try {
    const subjects = await (await fetch('/api/subjects')).json();
    const list = $('subject-cards');
    list.innerHTML = '';
    subjects.forEach((s, i) => {
      const card = document.createElement('div');
      card.className = `subject-card${s.ready ? '' : ' not-ready'}`;
      card.dataset.pid = s.pid;
      card.innerHTML = `
        <div class="card-rank">${i + 1}</div>
        <div class="card-body">
          <div class="card-pid">${s.pid}</div>
          <div class="card-meta">${capitalize(s.hemisphere)} · ${s.primary_lobe} lobe · ${s.total_volume} mL</div>
        </div>
        <div class="card-accuracy">${s.accuracy}%</div>
      `;
      if (s.ready) card.addEventListener('click', () => selectSubject(s.pid, card));
      list.appendChild(card);
    });
  } catch { $('subject-cards').innerHTML='<div class="loading-placeholder" style="color:#f43f5e">Failed</div>'; }
}

async function selectSubject(pid, cardEl) {
  if (pid === currentPid) return;

  document.querySelectorAll('.subject-card').forEach(c => c.classList.remove('active'));
  cardEl.classList.add('active');
  currentPid = pid;
  askedIds.clear();
  $('chat-messages').innerHTML = '';
  showCategories();
  clearAutoPlay();

  try {
    const data    = await (await fetch(`/api/subject/${pid}`)).json();
    currentImages = data.images || {};

    updateStats(data.meta);
    renderReport(data.report, data.meta);
    showSubjectUI();
    renderStaticImages();

    // NIfTI native-space 3-panel viewer
    if (data.has_nifti) {
      await initAllPanels(pid);
      $('sect-nifti').classList.remove('hidden');
    }

    // Registered-space 2-panel viewer
    await initRegPanels(pid);

  } catch (e) { console.error(e); }
}


// ─── Stats (animated pipeline simulation) ───────────────────────────────────
async function updateStats(meta) {
  if (!meta) return;

  // Header + badge appear immediately (pipeline has subject loaded)
  $('stats-pid').textContent      = meta.pid;
  $('stats-accuracy').textContent = `${meta.accuracy_pct}%`;
  $('stats-accuracy').classList.add('badge-pulse');

  // Show the panel in skeleton state (dashes + shimmer)
  setStatsSkeleton(true);
  $('stats-panel').classList.remove('hidden');

  // ── Phase 1: Simulate pipeline computation (800 ms) ────────────────────────
  await sleep(820);

  // ── Phase 2: Volume values count up one by one ────────────────────────────
  await countUp('stat-total-vol', meta.total_volume_ml);
  await sleep(160);
  await countUp('stat-ncr',       meta.ncr_volume_ml);
  await sleep(160);
  await countUp('stat-ed',        meta.ed_volume_ml);
  await sleep(160);
  await countUp('stat-et',        meta.et_volume_ml);

  // ── Phase 3: Meta rows fade in sequentially ──────────────────────────────
  await sleep(200);
  await revealMetaRow('meta-hemi',    capitalize(meta.hemisphere));
  await sleep(90);
  await revealMetaRow('meta-lobe',    `${meta.primary_lobe} lobe`);
  await sleep(90);
  const ml = $('meta-midline');
  ml.textContent = meta.crosses_midline ? 'Yes' : 'No';
  ml.className   = 'meta-val meta-anim ' + (meta.crosses_midline ? 'val-warn' : 'val-ok');
  await sleep(90);
  await revealMetaRow('meta-regions', `${meta.n_regions} regions`);

  // ── Phase 4: Legend fades in ──────────────────────────────────────────────
  await sleep(150);
  $('stats-panel').querySelectorAll('.legend-row').forEach((row, i) => {
    setTimeout(() => row.classList.add('legend-anim'), i * 80);
  });
  $('stats-accuracy').classList.remove('badge-pulse');
}

function setStatsSkeleton(on) {
  ['stat-total-vol','stat-ncr','stat-ed','stat-et'].forEach(id => {
    const el = $(id);
    if (on) { el.textContent = '—'; el.closest('.stat-card').classList.add('stat-scanning'); }
    else     { el.closest('.stat-card').classList.remove('stat-scanning'); }
  });
  ['meta-hemi','meta-lobe','meta-midline','meta-regions'].forEach(id => {
    const el = $(id); if (el) { el.textContent = '—'; el.className = 'meta-val'; }
  });
  $('stats-panel').querySelectorAll('.legend-row').forEach(r => r.classList.remove('legend-anim'));
}

async function countUp(id, value) {
  const el     = $(id);
  const card   = el.closest('.stat-card');
  card.classList.remove('stat-scanning');
  card.classList.add('stat-pop');               // flash on reveal
  setTimeout(() => card.classList.remove('stat-pop'), 400);

  const target = parseFloat(value);
  if (!value || isNaN(target) || target === 0) { el.textContent = '—'; return; }

  const DURATION_MS = 550;
  const INTERVAL_MS = 16;
  const steps       = DURATION_MS / INTERVAL_MS;
  let step = 0;

  return new Promise(resolve => {
    const iv = setInterval(() => {
      step++;
      const t        = step / steps;
      const eased    = 1 - Math.pow(1 - t, 3);  // ease-out cubic
      el.textContent = (target * eased).toFixed(1);
      if (step >= steps) {
        clearInterval(iv);
        el.textContent = target.toFixed(1);
        resolve();
      }
    }, INTERVAL_MS);
  });
}

async function revealMetaRow(id, text) {
  const el = $(id);
  if (!el) return;
  el.textContent = text;
  el.classList.add('meta-anim');
  await sleep(20);   // give browser a paint frame
}

function fmt(v) { return (v==null||v===0)?'—':parseFloat(v).toFixed(1); }
function capitalize(s) { return s ? s[0].toUpperCase()+s.slice(1) : '—'; }


// ─── Static images ────────────────────────────────────────────────────────────
function renderStaticImages() {
  const pairs = [
    ['img-orthoview', 'orthoview'],
    ['img-seg',       'segmentation'],
    ['img-reg',       'registration'],
  ];
  let any = false;
  pairs.forEach(([elId, key]) => {
    const el  = $(elId);
    const url = currentImages[key];
    if (url) { el.src = url; el.classList.remove('hidden'); any = true; }
    else { el.classList.add('hidden'); el.closest('.img-cell').style.display='none'; }
  });
  if (any) $('sect-overview').classList.remove('hidden');
}

// ─── Registered-space panel state ─────────────────────────────────────────────
const regState = {
  brain:      { slice: 0, max: 100, cache: {} },
  projection: { slice: 0, max: 100, cache: {} },
};

async function initRegPanels(pid) {
  // Reset state
  regState.brain.cache      = {};
  regState.projection.cache = {};

  // Get dimensions of the registered T1
  const info = await fetch(`/api/slice_info/${pid}?space=registered`).then(r => r.json()).catch(() => null);
  if (!info) return;

  const mid = info.mid_axial;
  regState.brain.slice      = mid;
  regState.brain.max        = info.axial_slices - 1;
  regState.projection.slice = mid;
  regState.projection.max   = info.axial_slices - 1;

  // Set sliders
  ['brain', 'projection'].forEach(p => {
    const sl  = $(`reg-slider-${p}`);
    sl.max    = String(regState[p].max);
    sl.value  = String(regState[p].slice);
  });

  // Load initial slices in parallel
  await Promise.all([
    loadRegSlice('brain',      mid),
    loadRegSlice('projection', mid),
  ]);

  $('sect-registered').classList.remove('hidden');
}

async function loadRegSlice(panel, idx) {
  if (!currentPid) return;
  const isBrain = panel === 'brain';
  const params  = isBrain
    ? `space=registered&seg=0`
    : `space=registered&viz=projection`;
  const key     = `${currentPid}_reg_${panel}_${idx}`;
  const imgEl   = $(`reg-slice-${panel}`);
  const labelEl = $(`reg-label-${panel}`);

  let url = regState[panel].cache[key];
  if (!url) {
    try {
      const ts = Date.now();
      const resp = await fetch(`/api/slice/${currentPid}/axial/${idx}?${params}&_t=${ts}`);
      if (!resp.ok) return;
      url = URL.createObjectURL(await resp.blob());
      regState[panel].cache[key] = url;
    } catch { return; }
  }

  imgEl.src          = url;
  const st           = regState[panel];
  labelEl.textContent = `Slice ${idx + 1} / ${st.max + 1}`;
}

function setupRegSliders() {
  // Brain panel
  $('reg-slider-brain').addEventListener('input', async () => {
    const idx = parseInt($('reg-slider-brain').value, 10);
    regState.brain.slice = idx;
    await loadRegSlice('brain', idx);
  });
  $('reg-brain-prev').addEventListener('click', async () => {
    regState.brain.slice = Math.max(0, regState.brain.slice - 1);
    $('reg-slider-brain').value = String(regState.brain.slice);
    await loadRegSlice('brain', regState.brain.slice);
  });
  $('reg-brain-next').addEventListener('click', async () => {
    regState.brain.slice = Math.min(regState.brain.max, regState.brain.slice + 1);
    $('reg-slider-brain').value = String(regState.brain.slice);
    await loadRegSlice('brain', regState.brain.slice);
  });

  // Projection panel
  $('reg-slider-projection').addEventListener('input', async () => {
    const idx = parseInt($('reg-slider-projection').value, 10);
    regState.projection.slice = idx;
    await loadRegSlice('projection', idx);
  });
  $('reg-proj-prev').addEventListener('click', async () => {
    regState.projection.slice = Math.max(0, regState.projection.slice - 1);
    $('reg-slider-projection').value = String(regState.projection.slice);
    await loadRegSlice('projection', regState.projection.slice);
  });
  $('reg-proj-next').addEventListener('click', async () => {
    regState.projection.slice = Math.min(regState.projection.max, regState.projection.slice + 1);
    $('reg-slider-projection').value = String(regState.projection.slice);
    await loadRegSlice('projection', regState.projection.slice);
  });

  // Wheel scroll on each reg panel
  ['brain', 'projection'].forEach(panel => {
    $(`reg-slice-${panel}`)?.parentElement?.addEventListener('wheel', async (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 1 : -1;
      regState[panel].slice = Math.max(0, Math.min(regState[panel].max, regState[panel].slice + delta));
      $(`reg-slider-${panel}`).value = String(regState[panel].slice);
      await loadRegSlice(panel, regState[panel].slice);
    }, { passive: false });
  });
}


// ─── Show right panel UI ──────────────────────────────────────────────────────
function showSubjectUI() {
  $('viewer-placeholder').classList.add('hidden');
  $('right-placeholder').classList.add('hidden');
  $('report-section').classList.remove('hidden');
  $('qa-divider').classList.remove('hidden');
  $('chat-window').classList.remove('hidden');
  $('qa-controls').classList.remove('hidden');
}


// ─── Report ──────────────────────────────────────────────────────────────────
function renderReport(report, meta) {
  if (meta) $('report-pid').textContent = meta.pid;
  if (!report) return;
  setSection('report-morphology',  report.morphology);
  setSection('report-anatomy',     report.anatomy);
  setSection('report-correlation', report.correlation);

  const btnView = $('btn-view-report');
  if (btnView && meta && meta.pid) {
    btnView.href = `/report/${meta.pid}`;
    btnView.style.opacity = '1';
    btnView.style.pointerEvents = 'auto';
  }
}
function setSection(id, text) {
  const el = $(id); if (!el) return;
  if (!text || text.startsWith('[DRY')) { el.closest('.report-block').style.display='none'; return; }
  el.innerHTML = markdownLite(text);
}


// ─── NIfTI 3-panel viewer ─────────────────────────────────────────────────────
function setupNiftiCells() {
  VIEWS.forEach(view => {
    const cell = $(`ncell-${view}`);
    if (!cell) return;

    // Click to set active
    cell.addEventListener('click', () => setActiveView(view));

    // Wheel scroll on each panel independently
    cell.addEventListener('wheel', async (e) => {
      e.preventDefault();
      clearAutoPlay();
      setActiveView(view);
      const delta = e.deltaY > 0 ? 1 : -1;
      stepSlice(view, delta);
    }, { passive: false });
  });
}

function setActiveView(view) {
  activeView = view;
  VIEWS.forEach(v => {
    $(`ncell-${v}`).classList.toggle('active', v === view);
  });
  // Sync slider to active panel
  const st = panelState[view];
  const sl = $('slice-slider');
  sl.max   = String(st.max);
  sl.value = String(st.slice);
  updateStatusLabel();
}

function updateStatusLabel() {
  const st  = panelState[activeView];
  $('slice-status-label').textContent = `Slice ${st.slice + 1} / ${st.max + 1} · ${activeView.toUpperCase()}`;
}

async function stepSlice(view, delta) {
  const st = panelState[view];
  st.slice  = Math.max(0, Math.min(st.max, st.slice + delta));
  if (view === activeView) {
    $('slice-slider').value = String(st.slice);
    updateStatusLabel();
  }
  await loadPanelSlice(view, st.slice);
}

async function loadPanelSlice(view, idx) {
  if (!currentPid) return;
  const key   = `${currentPid}_${view}_${idx}`;
  const imgEl = $(`nslice-${view}`);

  let url = sliceCache[key];
  if (!url) {
    try {
      const ts = Date.now();
      const resp = await fetch(`/api/slice/${currentPid}/${view}/${idx}?_t=${ts}`);
      if (!resp.ok) return;
      url = URL.createObjectURL(await resp.blob());
      sliceCache[key] = url;
    } catch { return; }
  }

  imgEl.style.opacity = '0.6';
  imgEl.src = url;
  imgEl.style.opacity = '1';
}

function setupSlider() {
  const slider = $('slice-slider');

  slider.addEventListener('input', async () => {
    clearAutoPlay();
    const idx = parseInt(slider.value, 10);
    panelState[activeView].slice = idx;
    updateStatusLabel();
    await loadPanelSlice(activeView, idx);
  });

  $('slice-prev').addEventListener('click', async () => {
    clearAutoPlay();
    await stepSlice(activeView, -1);
    $('slice-slider').value = String(panelState[activeView].slice);
  });
  $('slice-next').addEventListener('click', async () => {
    clearAutoPlay();
    await stepSlice(activeView, 1);
    $('slice-slider').value = String(panelState[activeView].slice);
  });

  // Arrow keys
  document.addEventListener('keydown', async (e) => {
    if (!currentPid) return;
    if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      clearAutoPlay(); await stepSlice(activeView, -1);
      $('slice-slider').value = String(panelState[activeView].slice);
    }
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      clearAutoPlay(); await stepSlice(activeView, 1);
      $('slice-slider').value = String(panelState[activeView].slice);
    }
  });
}



async function initAllPanels(pid) {
  sliceCache = {};

  // Fetch volume dimensions
  try {
    sliceInfo = await (await fetch(`/api/slice_info/${pid}`)).json();
  } catch { return; }

  // Set mid-slices for each view
  panelState.axial.slice    = sliceInfo.mid_axial;
  panelState.axial.max      = sliceInfo.axial_slices - 1;
  panelState.coronal.slice  = sliceInfo.mid_coronal;
  panelState.coronal.max    = sliceInfo.coronal_slices - 1;
  panelState.sagittal.slice = sliceInfo.mid_sagittal;
  panelState.sagittal.max   = sliceInfo.sagittal_slices - 1;

  // Set active panel to axial initially
  setActiveView('axial');

  // Load all 3 mid-slices in parallel
  await Promise.all(VIEWS.map(v => loadPanelSlice(v, panelState[v].slice)));

  // Scroll through a few coronal slices as "wow" animation
  startAutoPlay(pid);
}

function startAutoPlay(pid) {
  const mid   = sliceInfo.mid_coronal;
  const range = Math.min(20, Math.floor(sliceInfo.coronal_slices * 0.12));
  const start = Math.max(0, mid - range);
  const end   = Math.min(panelState.coronal.max, mid + range);
  let idx = start, dir = 1, loops = 0;

  setActiveView('coronal');

  autoPlayTimer = setInterval(async () => {
    if (!currentPid) { clearAutoPlay(); return; }
    panelState.coronal.slice = idx;
    $('slice-slider').value  = String(idx);
    updateStatusLabel();
    await loadPanelSlice('coronal', idx);
    idx += dir * 2;
    if (idx >= end)   { dir = -1; idx = end;  loops++; }
    if (idx <= start) { dir =  1; idx = start; loops++; }
    if (loops >= 2) {
      clearAutoPlay();
      panelState.coronal.slice = mid;
      await loadPanelSlice('coronal', mid);
      $('slice-slider').value = String(mid);
      updateStatusLabel();
    }
  }, 70);
}
function clearAutoPlay() {
  if (autoPlayTimer) { clearInterval(autoPlayTimer); autoPlayTimer = null; }
}


// ─── Q&A ─────────────────────────────────────────────────────────────────────
function setupCategoryButtons() {
  document.querySelectorAll('.cat-btn').forEach(btn => {
    btn.addEventListener('click', () => { if (!isAnswering) showQuestions(btn.dataset.cat); });
  });
}
function showCategories() { $('category-menu').classList.remove('hidden'); $('question-menu').classList.add('hidden'); }
function showQuestions(cat) {
  $('category-menu').classList.add('hidden'); $('question-menu').classList.remove('hidden');
  $('question-prompt').textContent = `${CAT_LABELS[cat]}:`;
  $('question-btns').innerHTML = '';
  QUESTIONS[cat].forEach(q => {
    const btn = document.createElement('button');
    btn.className   = 'q-btn' + (askedIds.has(q.id) ? ' asked' : '');
    btn.textContent = q.label;
    btn.dataset.qid = q.id;
    btn.addEventListener('click', () => { if (!isAnswering) askQuestion(q.id, q.label, cat); });
    $('question-btns').appendChild(btn);
  });
}

async function askQuestion(qId, qLabel, cat) {
  if (!currentPid || isAnswering) return;
  isAnswering = true;
  askedIds.add(qId);
  document.querySelector(`[data-qid="${qId}"]`)?.classList.add('asked');

  appendUserBubble(qLabel, cat);
  const typingId = appendTypingIndicator(cat);

  const [resp] = await Promise.all([
    fetch(`/api/qa/${currentPid}/${qId}`).then(r=>r.json()).catch(()=>({answer:'Error.'})),
    sleep(getDelay(cat)),
  ]);

  removeTypingIndicator(typingId);
  appendAIBubble(resp.answer || 'Answer not available.');
  $('chat-window').scrollTop = $('chat-window').scrollHeight;
  isAnswering = false;
  setTimeout(showCategories, 300);
}


// ─── Chat ─────────────────────────────────────────────────────────────────────
const CAT_COLORS = { volume:'var(--teal)', region:'var(--purple)', morphology:'var(--amber)', clinical:'var(--rose)' };

function appendUserBubble(text, cat) {
  const color = CAT_COLORS[cat]||'var(--accent)';
  const el = document.createElement('div');
  el.className = 'chat-bubble user-bubble';
  el.innerHTML = `
    <div class="bubble-icon user-icon" style="background:${color}18;border-color:${color}40">
      <svg viewBox="0 0 20 20" fill="${color}"><path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"/></svg>
    </div>
    <div class="bubble-content">
      <div class="bubble-meta" style="color:${color}">${CAT_LABELS[cat]}</div>
      <div class="bubble-text user" style="border-color:${color}25">${escapeHtml(text)}</div>
    </div>`;
  $('chat-messages').appendChild(el);
  $('chat-window').scrollTop = $('chat-window').scrollHeight;
}

function appendAIBubble(text) {
  const el = document.createElement('div');
  el.className = 'chat-bubble ai-bubble';
  const textEl = document.createElement('div');
  textEl.className = 'bubble-text ai-text';
  el.innerHTML = `
    <div class="bubble-icon ai-icon">
      <svg viewBox="0 0 20 20" fill="white"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
    </div>
    <div class="bubble-content">
      <div class="bubble-meta ai-meta"><span class="ai-badge">MedGemma</span> Atlas-Verified Analysis</div>
    </div>`;
  el.querySelector('.bubble-content').appendChild(textEl);
  $('chat-messages').appendChild(el);
  typewriterEffect(textEl, markdownLite(text));
}

function appendTypingIndicator(cat) {
  const id = `typing-${Date.now()}`;
  const msgs = { volume:'Querying volumetric data…', region:'Mapping atlas regions…', morphology:'Analysing MRI scan…', clinical:'Evaluating clinical context…' };
  const el = document.createElement('div');
  el.id = id; el.className = 'chat-bubble ai-bubble';
  el.innerHTML = `
    <div class="bubble-icon ai-icon">
      <svg viewBox="0 0 20 20" fill="white"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
    </div>
    <div class="bubble-content">
      <div class="bubble-meta ai-meta"><span class="ai-badge">MedGemma</span></div>
      <div class="bubble-text ai-text thinking-bubble">
        <div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div>
        <span class="thinking-label">${msgs[cat]||'Processing…'}</span>
      </div>
    </div>`;
  $('chat-messages').appendChild(el);
  $('chat-window').scrollTop = $('chat-window').scrollHeight;
  return id;
}
function removeTypingIndicator(id) { const el=$(id); if(el) el.remove(); }

function typewriterEffect(el, html) {
  const plain = html.replace(/<[^>]+>/g,'');
  let i = 0;
  el.textContent = '';
  const iv = setInterval(() => {
    i += 4;
    if (i >= plain.length) { clearInterval(iv); el.innerHTML = html; $('chat-window').scrollTop=$('chat-window').scrollHeight; }
    else el.textContent = plain.slice(0,i);
  }, 10);
}

// ─── AMBIENT DOT-MATRIX SCANNER WAVE ─────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  window.addEventListener("resize", () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  let time = 0;
  const SPACING = 32; // Wider grid for a smoother, less dense appearance

  function animate() {
    ctx.clearRect(0, 0, width, height);
    time += 0.025; // Speed of the sweeping wave
    
    // Smooth infinite diagonal drift effect
    let offsetX = (time * 12) % SPACING;
    let offsetY = (time * 12) % SPACING;
    const isLight = document.body.classList.contains('light-theme');

    for (let x = -SPACING; x < width + SPACING; x += SPACING) {
      for (let y = -SPACING; y < height + SPACING; y += SPACING) {
        
        let trueX = x + offsetX;
        let trueY = y - offsetY; // Drifts up and right
        
        // Calculate the linear bounding line of the wave scanner
        let wavePhase = (trueX * 0.003) + (trueY * 0.005) - time * 1.5;
        let wave = (Math.sin(wavePhase) + 1) / 2; // Normalize 0 to 1
        
        // Exponentiate the wave so it looks like a sharp laser scanner line
        let intenseGlow = Math.pow(wave, 8); 
        let subtleGlow = Math.pow(wave, 3) * 0.2;
        
        // Base visibility
        let opacity = isLight ? 0.05 : 0.06;
        opacity += subtleGlow + (intenseGlow * 0.35);
        if (opacity > 1) opacity = 1;
        
        // Dots gently expand as the wave hits them
        let radius = 1.0 + (intenseGlow * 0.6);
        
        ctx.beginPath();
        ctx.arc(trueX, trueY, radius, 0, Math.PI * 2);
        
        // The wave dynamically paints the grid, maintaining smooth monochromatic colors
        if (intenseGlow > 0.2) {
             ctx.shadowBlur = 4 * intenseGlow;
             ctx.shadowColor = isLight ? 'rgba(37, 99, 235, 0.4)' : 'rgba(172, 189, 242, 0.4)';
        } else {
             ctx.shadowBlur = 0;
        }

        // Use ONLY the peaceful '#acbdf2' color mathematically for dark mode!
        ctx.fillStyle = isLight ? `rgba(37, 99, 235, ${opacity})` : `rgba(172, 189, 242, ${opacity})`;
        
        ctx.fill();
      }
    }
    requestAnimationFrame(animate);
  }
  animate();
});


// ─── Util ──────────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r=>setTimeout(r,ms)); }
function escapeHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function markdownLite(text) {
  if (!text) return '';
  return text.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>').replace(/^/,'<p>').replace(/$/, '</p>');
}
