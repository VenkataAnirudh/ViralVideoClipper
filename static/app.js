/**
 * ViralClipper — Frontend Application
 * State machine: input → processing → results
 */

const App = {
  state: 'input', // 'input' | 'processing' | 'results'
  jobId: null,
  config: null,
  apiKeyValid: false,
  eventSource: null,
  logInterval: null,

  async init() {
    await this.loadConfig();
    this.bindEvents();
    this.showState('input');
  },

  async loadConfig() {
    try {
      const res = await fetch('/api/config');
      this.config = await res.json();
      this.populateForm();
    } catch (e) {
      console.error('Failed to load config:', e);
    }
  },

  populateForm() {
    const c = this.config;
    if (!c) return;

    // Provider toggle
    document.querySelectorAll('.provider-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.provider === c.default_provider);
    });

    // Models
    this.updateModelDropdown(c.default_provider);

    // Defaults
    const clipSlider = document.getElementById('clipCount');
    const minSlider = document.getElementById('minDuration');
    const maxSlider = document.getElementById('maxDuration');
    if (clipSlider) { clipSlider.value = c.default_clip_count; this.updateSliderLabel('clipCount'); }
    if (minSlider) { minSlider.value = c.default_min_duration; this.updateSliderLabel('minDuration'); }
    if (maxSlider) { maxSlider.value = c.default_max_duration; this.updateSliderLabel('maxDuration'); }

    // Caption style
    const styleSelect = document.getElementById('captionStyle');
    if (styleSelect && c.caption_styles) {
      styleSelect.innerHTML = '';
      for (const [key, name] of Object.entries(c.caption_styles)) {
        const opt = document.createElement('option');
        opt.value = key; opt.textContent = name;
        if (key === c.default_caption_style) opt.selected = true;
        styleSelect.appendChild(opt);
      }
    }

    // Aspect ratio
    document.querySelectorAll('.aspect-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.ratio === c.default_aspect_ratio);
    });

    // API key hint
    const provider = this.getProvider();
    if (
      (provider === 'featherless' && c.has_featherless_key) ||
      (provider === 'claude' && c.has_anthropic_key) ||
      (provider === 'gemini' && c.has_google_key) ||
      (provider === 'openrouter' && c.has_openrouter_key)
    ) {
      document.getElementById('apiKey').placeholder = '••••••••  (using key from .env file)';
    }
  },

  updateModelDropdown(provider) {
    const select = document.getElementById('modelSelect');
    if (!select || !this.config) return;
    select.innerHTML = '';
    const modelMap = {
      featherless: this.config.featherless_models,
      claude: this.config.claude_models,
      gemini: this.config.gemini_models,
      openrouter: this.config.openrouter_models,
    };
    const defaultMap = {
      featherless: this.config.default_featherless_model,
      claude: this.config.default_claude_model,
      gemini: this.config.default_gemini_model,
      openrouter: this.config.default_openrouter_model,
    };
    const models = modelMap[provider] || this.config.featherless_models || [];
    const defaultModel = defaultMap[provider] || models[0];
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      if (m === defaultModel) opt.selected = true;
      select.appendChild(opt);
    });
  },

  bindEvents() {
    // Provider toggle
    document.querySelectorAll('.provider-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.provider-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        this.updateModelDropdown(btn.dataset.provider);
        this.apiKeyValid = false;
        this.updateProcessBtn();
        document.getElementById('testResult').className = 'test-result';
        const apiInput = document.getElementById('apiKey');
        const c = this.config;
        const p = btn.dataset.provider;
        if (
          (p === 'featherless' && c.has_featherless_key) ||
          (p === 'claude' && c.has_anthropic_key) ||
          (p === 'gemini' && c.has_google_key) ||
          (p === 'openrouter' && c.has_openrouter_key)
        ) {
          apiInput.placeholder = '••••••••  (using key from .env file)';
        } else {
          apiInput.placeholder = 'Enter your API key...';
        }
      });
    });

    // Aspect ratio toggle
    document.querySelectorAll('.aspect-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.aspect-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      });
    });

    // Sliders
    ['clipCount', 'minDuration', 'maxDuration'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('input', () => this.updateSliderLabel(id));
    });

    // Test API key
    document.getElementById('btnTestKey')?.addEventListener('click', () => this.testApiKey());

    // Process button
    document.getElementById('btnProcess')?.addEventListener('click', () => this.startProcessing());

    // Log toggle
    document.getElementById('logToggle')?.addEventListener('click', () => {
      document.getElementById('logViewer')?.classList.toggle('open');
    });

    // Process another
    document.getElementById('btnNewJob')?.addEventListener('click', () => {
      this.showState('input');
    });
  },

  updateSliderLabel(id) {
    const slider = document.getElementById(id);
    const label = document.getElementById(id + 'Val');
    if (slider && label) {
      label.textContent = slider.value + (id.includes('Duration') ? 's' : '');
    }
  },

  getProvider() {
    return document.querySelector('.provider-btn.active')?.dataset.provider || 'featherless';
  },

  async testApiKey() {
    const btn = document.getElementById('btnTestKey');
    const result = document.getElementById('testResult');
    const provider = this.getProvider();
    const apiKey = document.getElementById('apiKey').value.trim();
    const model = document.getElementById('modelSelect').value;

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Testing...';
    result.className = 'test-result';

    try {
      const res = await fetch('/api/test-key', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ provider, api_key: apiKey, model }),
      });
      const data = await res.json();

      if (data.success) {
        result.className = 'test-result show success';
        result.textContent = data.message;
        btn.className = 'btn btn-test success';
        this.apiKeyValid = true;
      } else {
        result.className = 'test-result show error';
        result.textContent = data.message;
        btn.className = 'btn btn-test error';
        this.apiKeyValid = false;
      }
    } catch (e) {
      result.className = 'test-result show error';
      result.textContent = '❌ Network error: ' + e.message;
      this.apiKeyValid = false;
    }

    btn.disabled = false;
    btn.innerHTML = '🔑 Test API Key';
    this.updateProcessBtn();
  },

  updateProcessBtn() {
    const btn = document.getElementById('btnProcess');
    if (btn) btn.disabled = !this.apiKeyValid;
  },

  async startProcessing() {
    const url = document.getElementById('videoUrl').value.trim();
    if (!url) { alert('Please enter a YouTube URL'); return; }

    const body = {
      url,
      ai_provider: this.getProvider(),
      ai_model: document.getElementById('modelSelect').value,
      api_key: document.getElementById('apiKey').value.trim(),
      clip_count: document.getElementById('clipCount').value,
      min_duration: document.getElementById('minDuration').value,
      max_duration: document.getElementById('maxDuration').value,
      caption_style: document.getElementById('captionStyle').value,
      aspect_ratio: document.querySelector('.aspect-btn.active')?.dataset.ratio || '9:16',
      blog_post_enabled: !!document.getElementById('blogPostEnabled')?.checked,
    };

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) { alert(data.error); return; }
      this.jobId = data.job_id;
      this.showState('processing');
      this.startProgressPolling();
    } catch (e) {
      alert('Failed to start processing: ' + e.message);
    }
  },

  startProgressPolling() {
    // SSE
    if (this.eventSource) this.eventSource.close();
    this.eventSource = new EventSource(`/api/progress/${this.jobId}`);

    this.eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.updateProgress(data);
        if (data.status === 'done') {
          this.eventSource.close();
          this.loadResults();
        } else if (data.status === 'error') {
          this.eventSource.close();
          this.showError(data.message);
        }
      } catch (e) { console.error('SSE parse error:', e); }
    };

    this.eventSource.onerror = () => {
      // Fallback to polling if SSE fails
      this.eventSource.close();
      this.startFallbackPolling();
    };

    // Poll logs
    this.logInterval = setInterval(() => this.fetchLogs(), 3000);
  },

  startFallbackPolling() {
    const poll = async () => {
      try {
        const res = await fetch(`/api/jobs`);
        const jobs = await res.json();
        const job = jobs.find(j => j.id === this.jobId);
        if (job) {
          this.updateProgress(job);
          if (job.status === 'done') { this.loadResults(); return; }
          if (job.status === 'error') { this.showError(job.message); return; }
        }
      } catch (e) { console.error('Poll error:', e); }
      setTimeout(poll, 2000);
    };
    poll();
  },

  updateProgress(data) {
    const pct = Math.max(0, Math.min(100, data.progress || 0));
    const fill = document.getElementById('progressFill');
    const pctEl = document.getElementById('progressPct');
    const stageEl = document.getElementById('progressStage');

    if (fill) fill.style.width = pct + '%';
    if (pctEl) pctEl.textContent = pct + '%';
    if (stageEl) stageEl.textContent = data.message || data.stage || '';

    // Update stage list
    const stages = ['downloading','transcribing','analyzing','selecting','extracting','captioning','copywriting','packaging'];
    const currentIdx = stages.indexOf(data.stage);
    stages.forEach((s, i) => {
      const el = document.getElementById('stage-' + s);
      if (!el) return;
      el.classList.remove('active', 'done');
      if (i < currentIdx) el.classList.add('done');
      else if (i === currentIdx) el.classList.add('active');

      const icon = el.querySelector('.stage-icon');
      if (i < currentIdx) icon.textContent = '✅';
      else if (i === currentIdx) icon.innerHTML = '<span class="spinner"></span>';
      else icon.textContent = '⏳';
    });
  },

  async fetchLogs() {
    if (!this.jobId) return;
    try {
      const res = await fetch(`/api/logs/${this.jobId}?lines=30`);
      const data = await res.json();
      const logEl = document.getElementById('logContent');
      if (logEl && data.log) {
        logEl.textContent = data.log;
        logEl.scrollTop = logEl.scrollHeight;
      }
    } catch (e) { /* ignore */ }
  },

  showError(msg) {
    if (this.logInterval) clearInterval(this.logInterval);
    const errEl = document.getElementById('errorBanner');
    if (errEl) { errEl.textContent = '❌ Error: ' + msg; errEl.classList.add('show'); }
  },

  async loadResults() {
    if (this.logInterval) clearInterval(this.logInterval);
    try {
      const res = await fetch(`/api/results/${this.jobId}`);
      const results = await res.json();
      this.renderResults(results);
      this.showState('results');
    } catch (e) {
      console.error('Failed to load results:', e);
    }
  },

  renderResults(results) {
    const title = document.getElementById('resultsVideoTitle');
    const summary = document.getElementById('resultsSummary');
    const grid = document.getElementById('clipsGrid');

    if (title) title.textContent = results.video_title || 'Video Processed';
    if (summary) summary.textContent = results.summary || '';

    if (!grid) return;
    grid.innerHTML = '';

    const aspect = document.querySelector('.aspect-btn.active')?.dataset.ratio || '9:16';

    results.clips.forEach(clip => {
      const card = document.createElement('div');
      card.className = 'clip-card';
      const videoFile = clip.has_captioned ? clip.files.captioned : clip.files.raw;
      const videoClass = aspect === '16:9' ? 'clip-video clip-video-landscape' : 'clip-video';

      card.innerHTML = `
        <video class="${videoClass}" controls preload="metadata"
               src="/api/preview/${this.jobId}/${videoFile}"></video>
        <div class="clip-info">
          <div class="clip-title">${this.escapeHtml(clip.title)}</div>
          <div class="clip-scores">
            <span class="score-badge score-hook">🎣 Hook ${clip.hook_score}/10</span>
            <span class="score-badge score-flow">🌊 Flow ${clip.flow_score}/10</span>
            <span class="score-badge score-viral">🔥 Viral ${clip.virality_score}/10</span>
          </div>
          <div class="clip-caption" onclick="App.copyCaption(this)" title="Click to copy">
            <span class="copy-hint">📋 Click to copy</span>
            ${this.escapeHtml(clip.social_caption || clip.title)}
          </div>
          ${clip.hashtags?.length ? `<div class="clip-hashtags">${clip.hashtags.join(' ')}</div>` : ''}
          <div class="clip-actions">
            ${clip.has_captioned ? `<a class="btn btn-sm btn-download" href="/api/download/${this.jobId}/${clip.files.captioned}">⬇ Captioned</a>` : ''}
            ${clip.has_raw ? `<a class="btn btn-sm btn-outline" href="/api/download/${this.jobId}/${clip.files.raw}">⬇ Raw</a>` : ''}
            ${clip.has_srt ? `<a class="btn btn-sm btn-outline" href="/api/download/${this.jobId}/${clip.files.srt}">⬇ SRT</a>` : ''}
          </div>
        </div>
      `;
      grid.appendChild(card);
    });
  },

  copyCaption(el) {
    const text = el.textContent.replace('📋 Click to copy', '').trim();
    navigator.clipboard.writeText(text).then(() => {
      const orig = el.style.borderColor;
      el.style.borderColor = 'var(--accent-green)';
      setTimeout(() => { el.style.borderColor = orig; }, 1000);
    });
  },

  showState(state) {
    this.state = state;
    document.getElementById('stateInput')?.classList.toggle('hidden', state !== 'input');
    document.getElementById('stateProcessing')?.classList.toggle('hidden', state !== 'processing');
    document.getElementById('stateResults')?.classList.toggle('hidden', state !== 'results');
    if (state === 'input') {
      document.getElementById('errorBanner')?.classList.remove('show');
      if (this.eventSource) this.eventSource.close();
      if (this.logInterval) clearInterval(this.logInterval);
    }
  },

  escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
  },
};

document.addEventListener('DOMContentLoaded', () => App.init());
