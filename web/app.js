/**
 * nanoGPT Studio - Frontend Interactive Controller
 * Real-time Training Monitoring, Canvas Loss Curve, Model Dialogue & Sizing
 */

document.addEventListener('DOMContentLoaded', () => {
  // Warn if running directly from file:// protocol
  if (window.location.protocol === 'file:') {
    const banner = document.createElement('div');
    banner.style.cssText = 'background: linear-gradient(90deg, #b91c1c, #dc2626); color: #fff; padding: 12px 20px; font-size: 13px; font-weight: 500; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.4); position: sticky; top: 0; z-index: 99999; line-height: 1.5;';
    banner.innerHTML = '⚠️ <strong>提示</strong>：您当前正在通过本地文件 (<code>file://</code>) 访问此页面。浏览器的跨域沙箱会阻止数据请求！请在项目根目录运行 <strong>start.bat</strong> 启动服务，并在浏览器中打开 👉 <a href="http://127.0.0.1:8000" style="color: #fef08a; text-decoration: underline; font-weight: 700;">http://127.0.0.1:8000</a> 即可正常使用。';
    document.body.prepend(banner);
  }

  // Always resolve to the backend server whether running on http or file:// protocol
  const API_BASE = (window.location.origin && window.location.origin.startsWith('http')) ? '' : 'http://127.0.0.1:8000';

  // ==========================================
  // 1. Tab Navigation
  // ==========================================
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.tab;
      tabBtns.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');

      if (targetId === 'train-tab') {
        renderChart();
      } else if (targetId === 'arch-tab') {
        fetchCheckpoints();
      }
    });
  });

  // ==========================================
  // 1.5 Dynamic Dataset State & Switcher
  // ==========================================
  let currentDataset = 'shakespeare_char';
  let currentDatasetName = 'Shakespeare 英文 (65字符)';
  let currentOutDir = 'checkpoints/llm_shakespeare_char';
  let currentVocabSize = 65;
  let cachedDatasetsList = [];

  const headerDatasetBadge = document.getElementById('header-dataset-badge');
  const vocabInfoBadge = document.getElementById('vocab-info-badge');
  const promptPresetsTitle = document.getElementById('prompt-presets-title');
  const chinesePresets = document.getElementById('chinese-prompt-presets');
  const englishPresets = document.getElementById('english-prompt-presets');
  const promptInput = document.getElementById('prompt-input');
  const initialLossHint = document.getElementById('kpi-train-loss-hint');
  const llmDatasetPillsContainer = document.getElementById('llm-dataset-pills');

  function applyDatasetSelection(ds) {
    if (!ds) return;
    currentDataset = ds.id;
    currentDatasetName = ds.name || ds.id;
    currentOutDir = `checkpoints/llm_${ds.id}`;
    currentVocabSize = parseInt(ds.vocab_size) || 1000;

    // Update active style on pills
    if (llmDatasetPillsContainer) {
      llmDatasetPillsContainer.querySelectorAll('.dataset-pill-btn').forEach(btn => {
        if (btn.dataset.dataset === currentDataset) {
          btn.classList.add('active');
        } else {
          btn.classList.remove('active');
        }
      });
    }

    // Update badges
    if (headerDatasetBadge) {
      headerDatasetBadge.innerHTML = `<span>📚 数据集: ${escapeHtml(currentDatasetName)} (${currentVocabSize}字/符)</span>`;
    }
    if (vocabInfoBadge) {
      vocabInfoBadge.textContent = `词表: ${currentVocabSize.toLocaleString()} 字符`;
    }

    // Check language heuristic (Chinese vs English)
    const isChinese = (ds && ds.lang === 'zh') || (
      !currentDataset.includes('shakespeare') &&
      !currentDataset.includes('openwebtext') &&
      (/[\u4e00-\u9fa5]/.test(currentDataset) || /[\u4e00-\u9fa5]/.test(currentDatasetName.replace(/英文|英语|shakespeare|char|bpe/gi, '')) || currentDataset.includes('daode') || currentDataset.includes('yuye'))
    );
    if (isChinese) {
      if (promptPresetsTitle) promptPresetsTitle.textContent = `📜 ${escapeHtml(currentDatasetName)} 中文引导词`;
      if (chinesePresets) chinesePresets.style.display = 'block';
      if (englishPresets) englishPresets.style.display = 'none';
      if (promptInput) {
        promptInput.placeholder = '在此输入你的提示词（例如：道生一，或 雨夜之中，或留空换行）...';
        if (!promptInput.value || promptInput.value.startsWith('ROMEO:')) {
          promptInput.value = '道生一，一生二，二生三，三生万物。';
        }
      }
    } else {
      if (promptPresetsTitle) promptPresetsTitle.textContent = `🎭 ${escapeHtml(currentDatasetName)} 引导词`;
      if (chinesePresets) chinesePresets.style.display = 'none';
      if (englishPresets) englishPresets.style.display = 'block';
      if (promptInput) {
        promptInput.placeholder = '在此输入你的提示词（例如：ROMEO: 或 Once upon a time... 或留空换行）...';
        if (!promptInput.value || promptInput.value.includes('道生一')) {
          promptInput.value = 'ROMEO:\nO, she doth teach the torches to burn bright!';
        }
      }
    }

    if (initialLossHint) {
      const approxLoss = Math.log(Math.max(currentVocabSize, 2)).toFixed(2);
      initialLossHint.textContent = `初始基准: ~${approxLoss}`;
    }

    updateModelParamPreview();
  }

  function syncDatasetSelectors(items) {
    cachedDatasetsList = items || [];
    const textDatasets = cachedDatasetsList.filter(d => d.type !== 'vision');
    const visionDatasets = cachedDatasetsList.filter(d => d.type === 'vision');

    // 1. Render LLM Pills
    if (llmDatasetPillsContainer) {
      llmDatasetPillsContainer.innerHTML = '';
      if (textDatasets.length === 0) {
        llmDatasetPillsContainer.innerHTML = `
          <div style="font-size: 12px; color: #f87171; padding: 6px;">
            ⚠️ 本地无可用文本数据集，请在【数据集资产库】中新增创建
          </div>
        `;
      } else {
        textDatasets.forEach(ds => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = `dataset-pill dataset-pill-btn ${ds.id === currentDataset ? 'active' : ''}`;
          btn.dataset.dataset = ds.id;
          btn.dataset.name = ds.name;
          btn.dataset.vocab = ds.vocab_size;
          btn.dataset.out = `checkpoints/llm_${ds.id}`;
          
          const flag = /[\u4e00-\u9fa5]/.test(ds.name) ? 'CN' : 'EN';
          btn.innerHTML = `
            <span class="pill-badge" style="font-size: 10px; font-weight: 700; background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 5px; border-radius: 4px; margin-right: 6px;">${flag}</span>
            <span class="pill-title">${escapeHtml(ds.name)}</span>
            <span class="pill-sub" style="font-size: 11px; color: #94a3b8; margin-left: 4px;">(${ds.vocab_size ? ds.vocab_size + '字' : '已就绪'})</span>
          `;
          btn.addEventListener('click', () => applyDatasetSelection(ds));
          llmDatasetPillsContainer.appendChild(btn);
        });
      }
    }

    // 2. Ensure current selection is valid
    const currentStillExists = textDatasets.find(d => d.id === currentDataset);
    if (!currentStillExists && textDatasets.length > 0) {
      applyDatasetSelection(textDatasets[0]);
    } else if (currentStillExists) {
      applyDatasetSelection(currentStillExists);
    }

    // 3. Sync Vision Dataset Select Dropdown
    const visionSelect = document.getElementById('vision_dataset');
    if (visionSelect && visionDatasets.length > 0) {
      const prevVal = visionSelect.value;
      visionSelect.innerHTML = '';
      visionDatasets.forEach(vds => {
        const opt = document.createElement('option');
        opt.value = vds.id;
        opt.textContent = `${vds.name} (${vds.classes || 10}类 - ${vds.size_mb}MB)`;
        visionSelect.appendChild(opt);
      });
      if (prevVal && visionDatasets.some(v => v.id === prevVal)) {
        visionSelect.value = prevVal;
      }
    }
  }

  // ==========================================
  // 2. Presets & Parameter Calculation (6 Tiers)
  // ==========================================
  const presets = {
    baby: {
      n_layer: 4, n_head: 4, n_embd: 128, block_size: 64,
      batch_size: 12, max_iters: 1000, learning_rate: 0.001,
      dropout: 0.0, eval_interval: 100, log_interval: 20
    },
    tiny: {
      n_layer: 6, n_head: 8, n_embd: 256, block_size: 128,
      batch_size: 16, max_iters: 2000, learning_rate: 0.0008,
      dropout: 0.1, eval_interval: 100, log_interval: 20
    },
    mini: {
      n_layer: 8, n_head: 6, n_embd: 384, block_size: 256,
      batch_size: 16, max_iters: 3000, learning_rate: 0.0005,
      dropout: 0.1, eval_interval: 100, log_interval: 20
    },
    small: {
      n_layer: 10, n_head: 8, n_embd: 512, block_size: 256,
      batch_size: 12, max_iters: 4000, learning_rate: 0.0003,
      dropout: 0.15, eval_interval: 100, log_interval: 20
    },
    medium: {
      n_layer: 12, n_head: 10, n_embd: 640, block_size: 512,
      batch_size: 8, max_iters: 5000, learning_rate: 0.00025,
      dropout: 0.15, eval_interval: 100, log_interval: 20
    },
    gpt2: {
      n_layer: 12, n_head: 12, n_embd: 768, block_size: 1024,
      batch_size: 4, max_iters: 6000, learning_rate: 0.0002,
      dropout: 0.1, eval_interval: 200, log_interval: 20
    }
  };

  const presetCards = document.querySelectorAll('.preset-card');
  presetCards.forEach(card => {
    card.addEventListener('click', () => {
      presetCards.forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      applyPreset(presets[card.dataset.preset]);
    });
  });

  function applyPreset(p) {
    if (!p) return;
    for (const [key, val] of Object.entries(p)) {
      const input = document.getElementById(key);
      if (input) input.value = val;
    }
    updateModelParamPreview();
  }

  function updateModelParamPreview() {
    const n_layer = parseInt(document.getElementById('n_layer').value) || 4;
    const n_head = parseInt(document.getElementById('n_head').value) || 4;
    const n_embd = parseInt(document.getElementById('n_embd').value) || 128;
    const block_size = parseInt(document.getElementById('block_size').value) || 64;
    const vocab_size = currentVocabSize;

    // Validation check: n_embd % n_head == 0
    const startBtn = document.getElementById('start-btn');
    const warningEl = document.getElementById('param-error-hint') || (() => {
      const el = document.createElement('div');
      el.id = 'param-error-hint';
      el.style.cssText = 'color: #f87171; font-size: 12px; margin-top: 8px; font-weight: 500;';
      document.querySelector('.calc-preview-card').after(el);
      return el;
    })();

    if (n_embd % n_head !== 0) {
      warningEl.textContent = `⚠️ 错误：维度 (${n_embd}) 无法被头数 (${n_head}) 整除！多头注意力要求整除，建议将头数改为 ${[1,2,4,6,8,12,16].filter(h => n_embd % h === 0).join(', ')}`;
      warningEl.style.display = 'block';
      startBtn.disabled = true;
      startBtn.title = '参数不合法，无法启动';
    } else {
      warningEl.style.display = 'none';
      if (!isTrainingActive) {
        startBtn.disabled = false;
        startBtn.title = '';
      }
    }

    const token_emb = vocab_size * n_embd;
    const pos_emb = block_size * n_embd;
    const attn = (n_embd * 3 * n_embd) + (n_embd * n_embd);
    const mlp = (n_embd * 4 * n_embd) + (4 * n_embd * n_embd);
    const ln = 4 * n_embd;
    const total_block = n_layer * (attn + mlp + ln);
    const total = token_emb + pos_emb + total_block + (2 * n_embd);

    const mCount = (total / 1e6).toFixed(2);
    const memMb = ((total * 4 * 3) / (1024 * 1024)).toFixed(1);

    document.getElementById('model-param-count').textContent = `${mCount} M`;
    document.getElementById('model-mem-est').textContent = `~${memMb} MB`;
  }
  let isTrainingActive = false;

  ['n_layer', 'n_head', 'n_embd', 'block_size'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateModelParamPreview);
  });
  updateModelParamPreview();

  // Reset button
  document.getElementById('reset-btn').addEventListener('click', () => {
    applyPreset(presets.baby);
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    const babyCard = document.querySelector('.preset-card[data-preset="baby"]');
    if (babyCard) babyCard.classList.add('active');
  });

  // ==========================================
  // 3. Training Start / Stop Handling
  // ==========================================
  const startBtn = document.getElementById('start-btn');
  const stopBtn = document.getElementById('stop-btn');
  const statusPill = document.getElementById('status-pill');
  const statusText = document.getElementById('status-text');

  startBtn.addEventListener('click', async () => {
    const params = {
      n_layer: parseInt(document.getElementById('n_layer').value),
      n_head: parseInt(document.getElementById('n_head').value),
      n_embd: parseInt(document.getElementById('n_embd').value),
      block_size: parseInt(document.getElementById('block_size').value),
      batch_size: parseInt(document.getElementById('batch_size').value),
      max_iters: parseInt(document.getElementById('max_iters').value),
      learning_rate: parseFloat(document.getElementById('learning_rate').value),
      dropout: parseFloat(document.getElementById('dropout').value),
      eval_interval: parseInt(document.getElementById('eval_interval').value),
      log_interval: parseInt(document.getElementById('log_interval').value),
      device: document.getElementById('llm_device')?.value || 'auto',
      dataset: currentDataset,
      out_dir: currentOutDir
    };

    try {
      startBtn.disabled = true;
      const res = await fetch(`${API_BASE}/api/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
      });
      const data = await res.json();
      if (!data.success) {
        alert('启动失败: ' + (data.error || '未知错误'));
        startBtn.disabled = false;
      }
    } catch (err) {
      alert('网络请求失败: ' + err.message);
      startBtn.disabled = false;
    }
  });

  stopBtn.addEventListener('click', async () => {
    stopBtn.disabled = true;
    try {
      await fetch(`${API_BASE}/api/train/stop`, { method: 'POST' });
    } catch (err) {
      console.error(err);
    }
  });

  // ==========================================
  // 4. Real-time Canvas Loss Chart
  // ==========================================
  const canvas = document.getElementById('loss-canvas');
  const ctx = canvas.getContext('2d');
  let lossHistory = [];

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    renderChart();
  }
  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 50);

  function renderChart() {
    const rect = canvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    ctx.clearRect(0, 0, width, height);

    if (lossHistory.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '13px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('等待训练数据中... 启动训练后此处将实时绘制 Loss 收敛折线', width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 30, bottom: 30, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const iters = lossHistory.map(d => d.iter);
    const maxIter = Math.max(...iters, 10);
    const minIter = 0;

    let allLosses = [];
    lossHistory.forEach(d => {
      if (d.train_loss != null) allLosses.push(d.train_loss);
      if (d.val_loss != null) allLosses.push(d.val_loss);
    });

    const maxLoss = Math.min(5.0, Math.max(4.5, ...allLosses));
    const minLoss = Math.max(0.5, Math.min(...allLosses) - 0.2);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    const yTicks = 4;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';

    for (let i = 0; i <= yTicks; i++) {
      const val = minLoss + (maxLoss - minLoss) * (i / yTicks);
      const y = padding.top + chartH - (i / yTicks) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(val.toFixed(2), padding.left - 8, y + 3);
    }

    ctx.textAlign = 'center';
    const xTicks = 5;
    for (let i = 0; i <= xTicks; i++) {
      const iterVal = Math.round(minIter + (maxIter - minIter) * (i / xTicks));
      const x = padding.left + (i / xTicks) * chartW;
      ctx.fillText(iterVal, x, height - padding.bottom + 16);
    }

    const getX = (iter) => padding.left + ((iter - minIter) / (maxIter - minIter || 1)) * chartW;
    const getY = (loss) => padding.top + chartH - ((loss - minLoss) / (maxLoss - minLoss || 1)) * chartH;

    // Draw Train Loss Curve (Emerald)
    const trainPoints = lossHistory.filter(d => d.train_loss != null);
    if (trainPoints.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      trainPoints.forEach((p, idx) => {
        const x = getX(p.iter);
        const y = getY(p.train_loss);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.lineTo(getX(trainPoints[trainPoints.length - 1].iter), padding.top + chartH);
      ctx.lineTo(getX(trainPoints[0].iter), padding.top + chartH);
      ctx.closePath();
      const grad = ctx.createLinearGradient(0, padding.top, 0, height);
      grad.addColorStop(0, 'rgba(16, 185, 129, 0.2)');
      grad.addColorStop(1, 'rgba(16, 185, 129, 0.0)');
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // Draw Val Loss Curve & Dots (Amber)
    const valPoints = lossHistory.filter(d => d.val_loss != null);
    if (valPoints.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2.5;
      valPoints.forEach((p, idx) => {
        const x = getX(p.iter);
        const y = getY(p.val_loss);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      valPoints.forEach(p => {
        const x = getX(p.iter);
        const y = getY(p.val_loss);
        ctx.fillStyle = '#f59e0b';
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }
  }

  // ==========================================
  // 5. Polling Loop & State Sync
  // ==========================================
  const logConsole = document.getElementById('log-console');
  const autoScrollCheck = document.getElementById('auto-scroll-check');
  let lastLogCount = 0;

  async function pollState() {
    try {
      const res = await fetch(`${API_BASE}/api/state`);
      if (!res.ok) return;
      const state = await res.json();

      if (state.is_training) {
        isTrainingActive = true;
        statusPill.className = 'status-indicator training';
        statusText.textContent = `训练中 (Iter ${state.current_iter}/${state.max_iters})`;
        startBtn.disabled = true;
        stopBtn.disabled = false;
      } else {
        isTrainingActive = false;
        if (state.status === 'finished') {
          statusPill.className = 'status-indicator';
          statusText.textContent = '训练已完成 (FINISHED)';
        } else if (state.status === 'stopped') {
          statusPill.className = 'status-indicator stopped';
          statusText.textContent = '已手动停止 (STOPPED)';
        } else {
          statusPill.className = 'status-indicator';
          statusText.textContent = '空闲中 (IDLE)';
        }
        startBtn.disabled = false;
        stopBtn.disabled = true;
      }

      const maxI = state.max_iters || 1000;
      const curI = state.current_iter || 0;
      document.getElementById('kpi-iter').textContent = `${curI} / ${maxI}`;
      const pct = Math.min(100, Math.round((curI / maxI) * 100));
      document.getElementById('progress-bar').style.width = `${pct}%`;

      if (state.train_loss != null) {
        document.getElementById('kpi-train-loss').textContent = state.train_loss.toFixed(4);
      } else if (state.latest_loss != null) {
        document.getElementById('kpi-train-loss').textContent = state.latest_loss.toFixed(4);
      }

      if (state.val_loss != null) {
        document.getElementById('kpi-val-loss').textContent = state.val_loss.toFixed(4);
      }

      if (state.latest_time_ms != null) {
        document.getElementById('kpi-speed').textContent = `${Math.round(state.latest_time_ms)} ms`;
      }
      document.getElementById('kpi-time-elapsed').textContent = `已耗时: ${state.elapsed_seconds || 0}s`;

      if (state.loss_history && state.loss_history.length !== lossHistory.length) {
        lossHistory = state.loss_history;
        renderChart();
      }

      if (state.logs && state.logs.length > lastLogCount) {
        const newLogs = state.logs.slice(lastLogCount);
        newLogs.forEach(line => {
          const div = document.createElement('div');
          div.className = 'log-line';
          if (line.includes('train loss') || line.includes('val loss')) {
            div.style.color = '#38bdf8';
            div.style.fontWeight = '600';
          } else if (line.includes('saving checkpoint')) {
            div.style.color = '#34d399';
          } else if (line.includes('error') || line.includes('异常')) {
            div.style.color = '#f87171';
          }
          div.textContent = line;
          logConsole.appendChild(div);
        });
        lastLogCount = state.logs.length;
        if (autoScrollCheck.checked) {
          logConsole.scrollTop = logConsole.scrollHeight;
        }
      }

      const ckptInfoEl = document.getElementById('chat-ckpt-info');
      let activeCkpt = null;
      if (state.checkpoints) {
        activeCkpt = state.checkpoints[currentDataset];
        if (!activeCkpt && currentDataset && currentDataset.includes('shakespeare')) {
          activeCkpt = state.checkpoints['shakespeare_char'] || state.checkpoints['shakespeare'];
        }
      }
      if (!activeCkpt && (!currentDataset || currentDataset === 'shakespeare_char')) {
        activeCkpt = state.checkpoint;
      }

      if (ckptInfoEl) {
        if (activeCkpt && activeCkpt.exists) {
          const dsLabel = currentDatasetName || currentDataset;
          ckptInfoEl.textContent = `已就绪 ${dsLabel} 权重 (${activeCkpt.size_mb} MB, 更新于 ${activeCkpt.updated_at})`;
          ckptInfoEl.style.color = '#34d399';
        } else {
          ckptInfoEl.textContent = `当前选定模型暂无保存权重，请在【训练监控】面板启动训练保存权重后再进行测试`;
          ckptInfoEl.style.color = '#f87171';
        }
      }

    } catch (err) {
      console.warn('Poll state error:', err);
    } finally {
      setTimeout(pollState, 1000);
    }
  }
  pollState();

  // Log Clear and Copy
  document.getElementById('clear-log-btn').addEventListener('click', () => {
    logConsole.innerHTML = '';
    lastLogCount = 0;
  });

  document.getElementById('copy-log-btn').addEventListener('click', () => {
    const text = Array.from(logConsole.querySelectorAll('.log-line')).map(el => el.textContent).join('\n');
    navigator.clipboard.writeText(text).then(() => {
      alert('日志已复制到剪贴板！');
    });
  });

  // ==========================================
  // 6. Chat / Model Sampling Playground
  // ==========================================
  const tempSlider = document.getElementById('temperature');
  const tempVal = document.getElementById('temp-val');
  tempSlider.addEventListener('input', () => tempVal.textContent = tempSlider.value);

  const topkSlider = document.getElementById('top_k');
  const topkVal = document.getElementById('topk-val');
  topkSlider.addEventListener('input', () => topkVal.textContent = topkSlider.value);

  const tokensSlider = document.getElementById('max_new_tokens');
  const tokensVal = document.getElementById('tokens-val');
  tokensSlider.addEventListener('input', () => tokensVal.textContent = tokensSlider.value);

  document.querySelectorAll('.preset-prompt-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      promptInput.value = btn.dataset.prompt;
      promptInput.focus();
    });
  });

  const generateBtn = document.getElementById('generate-btn');
  const chatHistory = document.getElementById('chat-history');

  async function handleGenerate() {
    const prompt = promptInput.value.trim() || "\n";
    const temperature = parseFloat(tempSlider.value);
    const top_k = parseInt(topkSlider.value);
    const max_new_tokens = parseInt(tokensSlider.value);

    appendUserMessage(prompt);
    const loadingCard = appendLoadingMessage();

    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="btn-icon">⏳</span> 正在自回归采样...';

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt,
          temperature: temperature,
          top_k: top_k,
          max_new_tokens: max_new_tokens,
          device: document.getElementById('llm_device')?.value || 'auto',
          dataset: currentDataset,
          out_dir: currentOutDir
        })
      });

      const data = await res.json();
      loadingCard.remove();

      if (data.success) {
        appendModelMessage(data.generated_text, data.elapsed_seconds, data.tokens_requested, data.warning);
      } else {
        appendErrorMessage(data.error || '生成失败');
      }
    } catch (err) {
      loadingCard.remove();
      appendErrorMessage('网络请求异常: ' + err.message + '（请检查服务器是否正在运行: http://127.0.0.1:8000）');
    } finally {
      generateBtn.disabled = false;
      generateBtn.innerHTML = '<span class="btn-icon">⚡</span> 立即生成 (Generate)';
      chatHistory.scrollTop = chatHistory.scrollHeight;
    }
  }

  generateBtn.addEventListener('click', handleGenerate);
  promptInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  });

  function appendUserMessage(text) {
    const card = document.createElement('div');
    card.className = 'message-card';
    card.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-content">
        <div class="msg-author">提示输入 (Prompt)</div>
        <div class="msg-text">${escapeHtml(text)}</div>
      </div>
    `;
    chatHistory.appendChild(card);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function appendLoadingMessage() {
    const card = document.createElement('div');
    card.className = 'message-card';
    card.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="msg-author">nanoGPT 正在预测中...</div>
        <div class="msg-text" style="color: #94a3b8;">Transformer 正在逐 Token 预测下一个字符并自回归循环，请稍候片刻...</div>
      </div>
    `;
    chatHistory.appendChild(card);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return card;
  }

  function appendModelMessage(text, elapsed, tokens, warning) {
    const card = document.createElement('div');
    card.className = 'message-card';
    const warnHtml = warning ? `<div style="font-size: 11px; color: #fbbf24; margin-top: 4px;">${escapeHtml(warning)}</div>` : '';
    const authorName = currentDataset === 'yuye_tanjing' ? 'nanoGPT (🇨🇳《雨夜谈经》中文模型)' : 'nanoGPT (🇬🇧 Shakespeare Char)';
    card.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-content">
        <div class="msg-author">${authorName}</div>
        <div class="msg-text model-generated-text">${escapeHtml(text)}</div>
        ${warnHtml}
        <div class="msg-meta">
          <span>⏱️ 耗时: ${elapsed}s</span>
          <span>🔤 生成字符数: ~${tokens}</span>
        </div>
      </div>
    `;
    chatHistory.appendChild(card);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function appendErrorMessage(error) {
    const card = document.createElement('div');
    card.className = 'message-card';
    card.style.borderColor = 'rgba(244, 63, 94, 0.4)';
    card.innerHTML = `
      <div class="msg-avatar">⚠️</div>
      <div class="msg-content">
        <div class="msg-author" style="color: #fb7185;">错误提示</div>
        <div class="msg-text" style="color: #fda4af;">${escapeHtml(error)}</div>
      </div>
    `;
    chatHistory.appendChild(card);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function escapeHtml(str) {
    return (str || '').replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  // ==========================================
  // 7. Checkpoints Table in Architecture Tab
  // ==========================================
  async function fetchCheckpoints() {
    const tbody = document.getElementById('checkpoint-table-body');
    if (!tbody) return;
    try {
      const res = await fetch(`${API_BASE}/api/checkpoints`);
      const items = await res.json();
      tbody.innerHTML = '';

      if (!Array.isArray(items) || items.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: #94a3b8; padding: 18px;">暂无本地权重检查点</td></tr>';
        return;
      }

      items.forEach(item => {
        const tr = document.createElement('tr');
        const exists = item.exists;
        const statusBadge = exists
          ? `<span class="badge badge-device" style="color: #34d399;">🟢 已保存 (可用)</span>`
          : '<span class="badge" style="color: #94a3b8;">⚪ 未就绪</span>';

        const deleteBtn = exists
          ? `<button type="button" class="term-btn delete-ckpt-btn" data-id="${item.id}" style="color: #f87171; border-color: rgba(248, 113, 113, 0.3); padding: 3px 8px; font-size: 11px;">🗑️ 清理权重</button>`
          : '';

        const dataOriginHint = item.has_dataset
          ? `<span style="display: block; font-size: 11px; color: #64748b; margin-top: 2px;">🔗 关联数据源: ${escapeHtml(item.linked_ds)}</span>`
          : `<span style="display: block; font-size: 11px; color: #f59e0b; margin-top: 2px;">⚠️ 数据源状态: ${escapeHtml(item.linked_ds)}</span>`;

        tr.innerHTML = `
          <td><code>${escapeHtml(item.dir)}</code></td>
          <td>
            <strong>${escapeHtml(item.name)}</strong>
            ${dataOriginHint}
          </td>
          <td>${exists ? item.size_mb + ' MB' : '-'}</td>
          <td>${exists ? item.updated_at : '-'}</td>
          <td>
            <div style="display: flex; align-items: center; gap: 8px;">
              ${statusBadge}
              ${deleteBtn}
            </div>
          </td>
        `;
        tbody.appendChild(tr);
      });

      // Bind delete handlers
      tbody.querySelectorAll('.delete-ckpt-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const cid = btn.dataset.id;
          if (confirm(`确定要清理模型检查点 [${cid}] 吗？此操作将彻底删除该目录下的 ckpt.pt 模型权重以释放硬盘空间。`)) {
            try {
              const dRes = await fetch(`${API_BASE}/api/checkpoints/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: cid })
              });
              const dData = await dRes.json();
              if (dData.success) {
                fetchCheckpoints();
              } else {
                alert('清理失败: ' + dData.error);
              }
            } catch (err) {
              alert('清理请求失败: ' + err.message);
            }
          }
        });
      });
    } catch (err) {
      console.error('Fetch checkpoints error:', err);
    }
  }

  // ==========================================
  // 8. Multi-Modal Task Hub (LLM / Vision / Diffusion / Datasets)
  // ==========================================
  const taskPills = document.querySelectorAll('.task-pill');
  const taskViews = document.querySelectorAll('.task-view');

  taskPills.forEach(pill => {
    pill.addEventListener('click', () => {
      taskPills.forEach(p => p.classList.remove('active'));
      taskViews.forEach(v => v.classList.remove('active'));

      pill.classList.add('active');
      const mode = pill.dataset.task;
      const targetView = document.getElementById(`${mode}-task-view`);
      if (targetView) targetView.classList.add('active');

      if (mode === 'vision') {
        resizeVisionCanvas();
        pollVisionState();
      } else if (mode === 'datasets') {
        fetchDatasets();
      } else if (mode === 'llm') {
        renderChart();
      }
    });
  });

  // Hardware Status Badge Update & Detail Modal
  let cachedHwData = null;
  const hwModal = document.getElementById('hardware-modal');
  const hwModalBody = document.getElementById('hw-modal-body');
  const closeHwModalBtn = document.getElementById('close-hw-modal-btn');
  const devicePill = document.getElementById('device-pill');

  if (devicePill) {
    devicePill.style.cursor = 'pointer';
    devicePill.title = '点击查看当前算力硬件探测与 GPU 加速指引';
    devicePill.addEventListener('click', () => {
      if (hwModal && cachedHwData) {
        renderHardwareModal(cachedHwData);
        hwModal.style.display = 'flex';
      }
    });
  }

  if (closeHwModalBtn && hwModal) {
    closeHwModalBtn.addEventListener('click', () => {
      hwModal.style.display = 'none';
    });
  }
  if (hwModal) {
    hwModal.addEventListener('click', (e) => {
      if (e.target === hwModal) hwModal.style.display = 'none';
    });
  }

  function renderHardwareModal(data) {
    if (!hwModalBody) return;
    const isCuda = data.cuda_available;
    const hasPhysGpu = data.physical_gpu_detected;
    const gpuName = data.physical_gpu_name || data.device_name;

    hwModalBody.innerHTML = `
      <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 14px; margin-bottom: 14px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="color: #94a3b8;">💻 本地处理器 (CPU):</span>
          <span style="color: #f8fafc; font-weight: 600;">${data.cpu_count} 核心高并发</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="color: #94a3b8;">🎮 物理显卡硬件 (GPU):</span>
          <span style="color: ${hasPhysGpu ? '#38bdf8' : '#94a3b8'}; font-weight: 600;">
            ${hasPhysGpu ? escapeHtml(gpuName) + ' (' + (data.physical_gpu_vram_mb || 4096) + 'MB 显存)' : '未检测到独立显卡'}
          </span>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
          <span style="color: #94a3b8;">🐍 Python 与 PyTorch 构建:</span>
          <span style="color: #cbd5e1; font-family: monospace;">PyTorch ${escapeHtml(data.torch_version || '2.14.1')}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
          <span style="color: #94a3b8;">⚡ CUDA 加速状态:</span>
          <span style="font-weight: 600; color: ${isCuda ? '#34d399' : '#f59e0b'};">
            ${isCuda ? '🟢 活跃中 (已就绪)' : '⚪ 未激活 (当前为 CPU 模式)'}
          </span>
        </div>
      </div>

      ${!isCuda && hasPhysGpu ? `
        <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
          <div style="font-weight: 600; color: #fbbf24; margin-bottom: 4px;">💡 为什么有显卡但显示为 CPU？</div>
          <div style="font-size: 12px; color: #e2e8f0; line-height: 1.6;">
            系统已成功探测到本地配备有 <strong>${escapeHtml(gpuName)}</strong> 独立显卡！<br>
            当前之所以使用 CPU 模式，是因为当前系统环境安装的是 <strong>纯 CPU 版本的 PyTorch</strong>（缺少 NVIDIA CUDA 驱动运行库支持）。
          </div>
        </div>

        <div style="margin-top: 10px;">
          <div style="font-weight: 600; color: #f8fafc; margin-bottom: 6px;">🚀 如何开启此显卡的 GPU 加速训练？</div>
          <div style="font-size: 12px; color: #94a3b8; margin-bottom: 8px;">
            系统底层代码已原生支持 GPU。只需在终端安装支持 CUDA 的 PyTorch 版本，nanoStudio 会自动识别显卡并激活绿色 GPU 状态：
          </div>
          <div style="background: #000; border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; padding: 10px 12px; font-family: monospace; font-size: 12px; color: #38bdf8; user-select: all;">
            pip install torch --index-url https://download.pytorch.org/whl/cu121
          </div>
        </div>
      ` : ''}

      ${isCuda ? `
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px;">
          <span style="color: #34d399; font-weight: 600;">🎉 显卡加速已完美就绪！</span>
          <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">
            在左侧超参数中选择【自动探测】或【NVIDIA GPU】，启动训练时将自动调用 ${escapeHtml(gpuName)} 独显执行高速矩阵张量计算。
          </div>
        </div>
      ` : ''}
    `;
  }

  async function updateHardwareBadge() {
    try {
      const res = await fetch(`${API_BASE}/api/hardware`);
      const data = await res.json();
      cachedHwData = data;
      const devicePill = document.getElementById('device-pill');
      const deviceHint = document.getElementById('device-hint-text');

      if (devicePill) {
        if (data.cuda_available) {
          devicePill.innerHTML = `<span>🚀 GPU: ${escapeHtml(data.device_name)} (${data.vram_total_gb}GB 显存)</span>`;
          devicePill.style.borderColor = '#10b981';
          if (deviceHint) deviceHint.textContent = `🟢 已就绪：${data.device_name} 独立显卡加速`;
        } else if (data.physical_gpu_detected) {
          devicePill.innerHTML = `<span>💻 运算平台: CPU <small style="opacity: 0.8; font-weight: normal;">(识别到 ${escapeHtml(data.physical_gpu_name)})</small></span>`;
          devicePill.style.borderColor = 'rgba(245, 158, 11, 0.4)';
          if (deviceHint) deviceHint.textContent = `⚠️ 检测到本地独立显卡 ${data.physical_gpu_name}，当前 PyTorch 为 CPU 版（点击顶部徽章可查看启用指引）`;
        } else {
          devicePill.innerHTML = `<span>💻 运算平台: CPU (${data.cpu_count} 核心并发优化)</span>`;
          if (deviceHint) deviceHint.textContent = `当前使用本地 ${data.cpu_count} 核心 CPU 满血多线程并发加速`;
        }
      }
    } catch (e) {
      console.warn('Hardware check error:', e);
    }
  }
  updateHardwareBadge();

  // ==========================================
  // 9. Vision Module Sub-tabs & Backbone Selector
  // ==========================================
  const visionTabBtns = document.querySelectorAll('.vision-tabs .tab-btn');
  visionTabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      visionTabBtns.forEach(b => b.classList.remove('active'));
      document.querySelectorAll('#vision-task-view .tab-pane').forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.dataset.tab;
      const pane = document.getElementById(targetId);
      if (pane) pane.classList.add('active');

      if (targetId === 'vision-train-tab') {
        resizeVisionCanvas();
      }
    });
  });

  document.querySelectorAll('input[name="vision_model"]').forEach(radio => {
    radio.addEventListener('change', () => {
      document.querySelectorAll('.model-radio-card').forEach(c => c.classList.remove('active'));
      radio.closest('.model-radio-card').classList.add('active');
    });
  });

  // ==========================================
  // 10. Vision Training & Dual-Metric Canvas Chart
  // ==========================================
  const visionStartBtn = document.getElementById('vision-start-btn');
  const visionStopBtn = document.getElementById('vision-stop-btn');
  const visionCanvas = document.getElementById('vision-chart-canvas');
  const visionCtx = visionCanvas.getContext('2d');
  let visionLossHistory = [];
  let isVisionTrainingActive = false;
  let lastVisionLogCount = 0;
  const visionLogConsole = document.getElementById('vision-log-console');

  function resizeVisionCanvas() {
    if (!visionCanvas) return;
    const rect = visionCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    visionCanvas.width = rect.width * dpr;
    visionCanvas.height = rect.height * dpr;
    visionCtx.scale(dpr, dpr);
    renderVisionChart();
  }
  window.addEventListener('resize', resizeVisionCanvas);

  function renderVisionChart() {
    if (!visionCanvas) return;
    const rect = visionCanvas.getBoundingClientRect();
    const width = rect.width;
    const height = rect.height;
    visionCtx.clearRect(0, 0, width, height);

    if (visionLossHistory.length === 0) {
      visionCtx.fillStyle = '#64748b';
      visionCtx.font = '13px Inter, sans-serif';
      visionCtx.textAlign = 'center';
      visionCtx.fillText('等待视觉训练数据中... 启动训练后实时绘制 Loss 与 Top-1 准确率 (Accuracy %)', width / 2, height / 2);
      return;
    }

    const padding = { top: 20, right: 45, bottom: 30, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const steps = visionLossHistory.map(d => d.step);
    const maxStep = Math.max(...steps, 10);
    const minStep = 0;

    let losses = visionLossHistory.map(d => d.train_loss).concat(visionLossHistory.map(d => d.val_loss)).filter(v => v != null);
    const maxLoss = Math.min(3.0, Math.max(2.5, ...losses));
    const minLoss = Math.max(0.0, Math.min(...losses) - 0.1);

    // Left Y Axis (Loss)
    visionCtx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    visionCtx.fillStyle = '#94a3b8';
    visionCtx.font = '10px Inter, sans-serif';
    visionCtx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const val = minLoss + (maxLoss - minLoss) * (i / 4);
      const y = padding.top + chartH - (i / 4) * chartH;
      visionCtx.beginPath();
      visionCtx.moveTo(padding.left, y);
      visionCtx.lineTo(width - padding.right, y);
      visionCtx.stroke();
      visionCtx.fillText(val.toFixed(2), padding.left - 6, y + 3);
    }

    // Right Y Axis (Accuracy %)
    visionCtx.fillStyle = '#06b6d4';
    visionCtx.textAlign = 'left';
    for (let i = 0; i <= 4; i++) {
      const accVal = i * 25; // 0, 25, 50, 75, 100%
      const y = padding.top + chartH - (i / 4) * chartH;
      visionCtx.fillText(accVal + '%', width - padding.right + 6, y + 3);
    }

    // X Axis (Steps)
    visionCtx.fillStyle = '#64748b';
    visionCtx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
      const sVal = Math.round(minStep + (maxStep - minStep) * (i / 4));
      const x = padding.left + (i / 4) * chartW;
      visionCtx.fillText(sVal, x, height - padding.bottom + 16);
    }

    const getX = (step) => padding.left + ((step - minStep) / (maxStep - minStep || 1)) * chartW;
    const getLossY = (l) => padding.top + chartH - ((l - minLoss) / (maxLoss - minLoss || 1)) * chartH;
    const getAccY = (acc) => padding.top + chartH - (acc / 100.0) * chartH;

    // Draw Train Loss Curve (Emerald)
    const trainPts = visionLossHistory.filter(d => d.train_loss != null);
    if (trainPts.length > 1) {
      visionCtx.beginPath();
      visionCtx.strokeStyle = '#10b981';
      visionCtx.lineWidth = 2;
      trainPts.forEach((p, idx) => {
        const x = getX(p.step);
        const y = getLossY(p.train_loss);
        if (idx === 0) visionCtx.moveTo(x, y);
        else visionCtx.lineTo(x, y);
      });
      visionCtx.stroke();
    }

    // Draw Val Loss Curve (Amber)
    const valPts = visionLossHistory.filter(d => d.val_loss != null);
    if (valPts.length > 1) {
      visionCtx.beginPath();
      visionCtx.strokeStyle = '#f59e0b';
      visionCtx.lineWidth = 2;
      valPts.forEach((p, idx) => {
        const x = getX(p.step);
        const y = getLossY(p.val_loss);
        if (idx === 0) visionCtx.moveTo(x, y);
        else visionCtx.lineTo(x, y);
      });
      visionCtx.stroke();
    }

    // Draw Accuracy Curve (Cyan)
    const accPts = visionLossHistory.filter(d => d.val_acc != null);
    if (accPts.length > 1) {
      visionCtx.beginPath();
      visionCtx.strokeStyle = '#06b6d4';
      visionCtx.lineWidth = 2.5;
      visionCtx.setLineDash([4, 4]);
      accPts.forEach((p, idx) => {
        const x = getX(p.step);
        const y = getAccY(p.val_acc);
        if (idx === 0) visionCtx.moveTo(x, y);
        else visionCtx.lineTo(x, y);
      });
      visionCtx.stroke();
      visionCtx.setLineDash([]);
    }
  }

  visionStartBtn.addEventListener('click', async () => {
    const model = document.querySelector('input[name="vision_model"]:checked')?.value || 'simple_cnn';
    const dataset = document.getElementById('vision_dataset').value || 'cifar10';
    const batchSize = parseInt(document.getElementById('vision_batch_size').value) || 32;
    const epochs = parseInt(document.getElementById('vision_epochs').value) || 5;
    const lr = parseFloat(document.getElementById('vision_lr').value) || 0.001;
    const device = document.getElementById('vision_device').value || 'auto';

    visionStartBtn.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/api/vision/train/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, dataset, batch_size: batchSize, epochs, learning_rate: lr, device })
      });
      const data = await res.json();
      if (!data.success) {
        alert('视觉训练启动失败: ' + data.error);
        visionStartBtn.disabled = false;
      }
    } catch (e) {
      alert('请求异常: ' + e.message);
      visionStartBtn.disabled = false;
    }
  });

  visionStopBtn.addEventListener('click', async () => {
    visionStopBtn.disabled = true;
    try {
      await fetch(`${API_BASE}/api/vision/train/stop`, { method: 'POST' });
    } catch (e) {
      console.warn(e);
    }
  });

  document.getElementById('vision-clear-log-btn').addEventListener('click', () => {
    visionLogConsole.innerHTML = '';
    lastVisionLogCount = 0;
  });

  async function pollVisionState() {
    try {
      const res = await fetch(`${API_BASE}/api/vision/state`);
      const state = await res.json();

      isVisionTrainingActive = state.is_training;
      visionStartBtn.disabled = state.is_training;
      visionStopBtn.disabled = !state.is_training;

      // Update KPIs
      const maxEp = state.max_epochs || 5;
      document.getElementById('vision-kpi-progress').textContent = `${state.epoch} / ${maxEp} 轮`;
      const pFill = document.getElementById('vision-progress-bar');
      if (pFill) pFill.style.width = `${Math.min(100, (state.epoch / maxEp) * 100)}%`;

      document.getElementById('vision-kpi-train-loss').textContent = state.train_loss != null ? state.train_loss.toFixed(4) : (state.latest_loss != null ? state.latest_loss.toFixed(4) : '-');
      document.getElementById('vision-kpi-val-loss').textContent = state.val_loss != null ? state.val_loss.toFixed(4) : '-';
      document.getElementById('vision-kpi-acc').textContent = state.val_acc != null ? `${state.val_acc.toFixed(2)} %` : '- %';

      // Update Chart
      if (state.loss_history && state.loss_history.length !== visionLossHistory.length) {
        visionLossHistory = state.loss_history;
        renderVisionChart();
      }

      // Append Logs
      if (state.logs && state.logs.length > lastVisionLogCount) {
        const newLines = state.logs.slice(lastVisionLogCount);
        newLines.forEach(line => {
          const div = document.createElement('div');
          div.className = 'log-line';
          if (line.includes('val acc')) {
            div.style.color = '#38bdf8';
            div.style.fontWeight = '600';
          } else if (line.includes('saving checkpoint')) {
            div.style.color = '#34d399';
          }
          div.textContent = line;
          visionLogConsole.appendChild(div);
        });
        lastVisionLogCount = state.logs.length;
        visionLogConsole.scrollTop = visionLogConsole.scrollHeight;
      }
    } catch (e) {
      console.warn('Poll vision state error:', e);
    } finally {
      const activeTaskPill = document.querySelector('.task-pill.active');
      if (activeTaskPill && activeTaskPill.dataset.task === 'vision') {
        setTimeout(pollVisionState, 1000);
      }
    }
  }

  // ==========================================
  // 11. Vision Playground (Interactive Prediction)
  // ==========================================
  const dropZone = document.getElementById('vision-drop-zone');
  const fileInput = document.getElementById('vision-file-input');
  const previewCanvas = document.getElementById('vision-preview-canvas');
  const previewCtx = previewCanvas.getContext('2d');
  const previewStatus = document.getElementById('preview-status');
  const predictBtn = document.getElementById('vision-predict-btn');
  let currentImageDataUrl = null;

  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      loadUserImage(e.target.files[0]);
    }
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadUserImage(e.dataTransfer.files[0]);
    }
  });

  function loadUserImage(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      currentImageDataUrl = evt.target.result;
      const img = new Image();
      img.onload = () => {
        previewCtx.clearRect(0, 0, 128, 128);
        previewCtx.drawImage(img, 0, 0, 128, 128);
        previewStatus.textContent = file.name;
        previewStatus.style.background = 'rgba(16, 185, 129, 0.2)';
        previewStatus.style.color = '#34d399';
      };
      img.src = currentImageDataUrl;
    };
    reader.readAsDataURL(file);
  }

  // 10 Sample Buttons Generation
  const classEmojis = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐶', '🐸', '🐴', '🚢', '🚚'];
  const sampleBtns = document.querySelectorAll('.sample-btn');

  sampleBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const cls = parseInt(btn.dataset.class);
      generateSyntheticSampleImage(cls);
    });
  });

  function generateSyntheticSampleImage(cls) {
    // Generate 32x32 synthetic pattern matching benchmark
    const offCanvas = document.createElement('canvas');
    offCanvas.width = 32;
    offCanvas.height = 32;
    const offCtx = offCanvas.getContext('2d');

    offCtx.fillStyle = '#1e293b';
    offCtx.fillRect(0, 0, 32, 32);

    if (cls === 0) { // Red square
      offCtx.fillStyle = '#ef4444';
      offCtx.fillRect(8, 8, 16, 16);
    } else if (cls === 1) { // Green circle
      offCtx.fillStyle = '#10b981';
      offCtx.beginPath();
      offCtx.arc(16, 16, 9, 0, Math.PI * 2);
      offCtx.fill();
    } else if (cls === 2) { // Blue cross
      offCtx.fillStyle = '#3b82f6';
      offCtx.fillRect(13, 6, 6, 20);
      offCtx.fillRect(6, 13, 20, 6);
    } else if (cls === 3) { // Yellow diagonal
      offCtx.strokeStyle = '#eab308';
      offCtx.lineWidth = 4;
      offCtx.beginPath();
      offCtx.moveTo(0, 0);
      offCtx.lineTo(32, 32);
      offCtx.stroke();
    } else if (cls === 4) { // Cyan bars
      offCtx.fillStyle = '#06b6d4';
      offCtx.fillRect(0, 6, 32, 5);
      offCtx.fillRect(0, 20, 32, 5);
    } else if (cls === 5) { // Magenta dots
      offCtx.fillStyle = '#d946ef';
      [8, 16, 24].forEach(r => {
        [8, 16, 24].forEach(c => {
          offCtx.fillRect(r-2, c-2, 5, 5);
        });
      });
    } else if (cls === 6) { // Orange diamond
      offCtx.fillStyle = '#f97316';
      offCtx.beginPath();
      offCtx.moveTo(16, 6);
      offCtx.lineTo(26, 16);
      offCtx.lineTo(16, 26);
      offCtx.lineTo(6, 16);
      offCtx.closePath();
      offCtx.fill();
    } else if (cls === 7) { // Purple gradient
      const grad = offCtx.createLinearGradient(0, 0, 0, 32);
      grad.addColorStop(0, '#581c87');
      grad.addColorStop(1, '#a855f7');
      offCtx.fillStyle = grad;
      offCtx.fillRect(4, 4, 24, 24);
    } else if (cls === 8) { // White triangle
      offCtx.fillStyle = '#ffffff';
      offCtx.beginPath();
      offCtx.moveTo(16, 8);
      offCtx.lineTo(26, 24);
      offCtx.lineTo(6, 24);
      offCtx.closePath();
      offCtx.fill();
    } else if (cls === 9) { // Concentric rings
      offCtx.strokeStyle = '#facc15';
      offCtx.lineWidth = 3;
      offCtx.beginPath();
      offCtx.arc(16, 16, 8, 0, Math.PI * 2);
      offCtx.stroke();
      offCtx.beginPath();
      offCtx.arc(16, 16, 13, 0, Math.PI * 2);
      offCtx.stroke();
    }

    currentImageDataUrl = offCanvas.toDataURL('image/png');

    // Draw zoomed to preview canvas
    previewCtx.clearRect(0, 0, 128, 128);
    previewCtx.imageSmoothingEnabled = false;
    previewCtx.drawImage(offCanvas, 0, 0, 128, 128);

    previewStatus.textContent = `样本 #${cls} (${classEmojis[cls]})`;
    previewStatus.style.background = 'rgba(99, 102, 241, 0.2)';
    previewStatus.style.color = '#818cf8';

    // Auto trigger prediction for smooth experience
    handleVisionPredict();
  }

  predictBtn.addEventListener('click', handleVisionPredict);

  async function handleVisionPredict() {
    if (!currentImageDataUrl) {
      alert('请先拖拽上传一张图片，或点击左侧的样张测试按钮！');
      return;
    }

    predictBtn.disabled = true;
    predictBtn.innerHTML = '<span class="btn-icon">⏳</span> 正在前向推理...';

    try {
      const res = await fetch(`${API_BASE}/api/vision/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: currentImageDataUrl, top_k: 5 })
      });
      const data = await res.json();

      if (data.success) {
        const top = data.top_prediction;
        const iconEl = document.getElementById('top-pred-icon');
        const classEl = document.getElementById('top-pred-class');
        const probEl = document.getElementById('top-pred-prob');
        const metaEl = document.getElementById('vision-infer-meta');

        iconEl.textContent = classEmojis[top.class_id] || '🏷️';
        classEl.textContent = top.class_name;
        probEl.textContent = `置信度: ${top.probability}%`;
        metaEl.textContent = `⚡ 推理耗时: ${data.inference_time_ms} ms | 权重: checkpoints/vision/ckpt.pt`;

        // Render Top-5 Bars
        const barsContainer = document.getElementById('vision-prob-bars');
        barsContainer.innerHTML = '';
        data.top_k.forEach((item, idx) => {
          const div = document.createElement('div');
          div.className = 'prob-bar-item';
          const emoji = classEmojis[item.class_id] || '';
          div.innerHTML = `
            <div class="prob-bar-info">
              <span class="prob-bar-name">${emoji} ${escapeHtml(item.class_name)}</span>
              <span class="prob-bar-pct">${item.probability}%</span>
            </div>
            <div class="prob-bar-track">
              <div class="prob-bar-fill" style="width: ${item.probability}%; ${idx === 0 ? 'background: linear-gradient(90deg, #10b981, #06b6d4);' : ''}"></div>
            </div>
          `;
          barsContainer.appendChild(div);
        });
      } else {
        alert('推理失败: ' + data.error);
      }
    } catch (e) {
      alert('网络请求失败: ' + e.message);
    } finally {
      predictBtn.disabled = false;
      predictBtn.innerHTML = '<span class="btn-icon">⚡</span> 立即前向推理预测 (Predict)';
    }
  }

  // ==========================================
  // 12. Datasets Asset Manager (CRUD & Dynamic Sync)
  // ==========================================
  const refreshDsBtn = document.getElementById('refresh-datasets-btn');
  if (refreshDsBtn) refreshDsBtn.addEventListener('click', () => fetchDatasets());

  const openCreateDsBtn = document.getElementById('open-create-dataset-btn');
  const closeCreateDsBtn = document.getElementById('close-create-dataset-btn');
  const createDsPanel = document.getElementById('create-dataset-panel');
  const createDsForm = document.getElementById('create-dataset-form');
  const newDsIdInput = document.getElementById('new_ds_id');
  const newDsNameInput = document.getElementById('new_ds_name');
  const newDsFileInput = document.getElementById('new_ds_file');
  const newDsContentInput = document.getElementById('new_ds_content');
  const newDsStatsLabel = document.getElementById('new_ds_stats');
  const submitCreateDsBtn = document.getElementById('submit-create-ds-btn');

  if (openCreateDsBtn && createDsPanel) {
    openCreateDsBtn.addEventListener('click', () => {
      createDsPanel.style.display = 'block';
      if (newDsIdInput) newDsIdInput.focus();
    });
  }

  if (closeCreateDsBtn && createDsPanel) {
    closeCreateDsBtn.addEventListener('click', () => {
      createDsPanel.style.display = 'none';
    });
  }

  if (newDsContentInput && newDsStatsLabel) {
    newDsContentInput.addEventListener('input', () => {
      const len = newDsContentInput.value.length;
      newDsStatsLabel.textContent = `字数统计: ${len.toLocaleString()} 字符`;
    });
  }

  if (newDsFileInput && newDsContentInput) {
    newDsFileInput.addEventListener('change', (e) => {
      const file = e.target.files && e.target.files[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (evt) => {
        const text = evt.target.result;
        newDsContentInput.value = text;
        if (newDsStatsLabel) newDsStatsLabel.textContent = `字数统计: ${text.length.toLocaleString()} 字符`;

        // Auto suggest ID and Name if empty
        const baseName = file.name.replace(/\.[^/.]+$/, '');
        if (newDsNameInput && !newDsNameInput.value.trim()) {
          newDsNameInput.value = baseName;
        }
        if (newDsIdInput && !newDsIdInput.value.trim()) {
          // Convert to lowercase alphanumeric/underscore
          const safeId = baseName.toLowerCase().replace(/[^a-z0-9_]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '') || 'custom_corpus';
          newDsIdInput.value = safeId;
        }
      };
      reader.onerror = () => {
        alert('读取本地文件失败，请确认文件编码为 UTF-8！');
      };
      reader.readAsText(file, 'utf-8');
    });
  }

  if (createDsForm) {
    createDsForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const id = newDsIdInput.value.trim();
      const name = newDsNameInput.value.trim();
      const content = newDsContentInput.value;

      if (!id) {
        alert('请输入数据集英文标识 (ID)！');
        newDsIdInput.focus();
        return;
      }
      if (!name) {
        alert('请输入数据集显示名称！');
        newDsNameInput.focus();
        return;
      }
      if (!content || content.length < 50) {
        alert('文本内容过短（至少需要 50 字以上），请粘贴更充足的文本材料以生成词表！');
        newDsContentInput.focus();
        return;
      }

      if (submitCreateDsBtn) {
        submitCreateDsBtn.disabled = true;
        submitCreateDsBtn.innerHTML = '<span>⏳ 正在提取字符词表并编译二进制...</span>';
      }

      try {
        const res = await fetch(`${API_BASE}/api/datasets/create`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id, name, content })
        });
        const data = await res.json();
        if (data.success) {
          alert(`🎉 数据集 [${name}] 创建并打包成功！\n\n• 字符总数: ${data.total_chars.toLocaleString()} 字\n• 独立词表大小: ${data.vocab_size.toLocaleString()} 字符\n• 训练集切分: ${data.train_tokens.toLocaleString()} tokens\n• 验证集切分: ${data.val_tokens.toLocaleString()} tokens\n\n已自动加入可选数据集列表并即时生效！`);
          
          createDsForm.reset();
          if (newDsStatsLabel) newDsStatsLabel.textContent = '字数统计: 0 字';
          if (createDsPanel) createDsPanel.style.display = 'none';

          // Refresh datasets and switch to new dataset
          const items = await fetchDatasets();
          const newlyCreated = items && items.find(d => d.id === id);
          if (newlyCreated) {
            applyDatasetSelection(newlyCreated);
          }
        } else {
          alert('创建数据集失败: ' + (data.error || '未知错误'));
        }
      } catch (err) {
        alert('创建请求异常: ' + err.message);
      } finally {
        if (submitCreateDsBtn) {
          submitCreateDsBtn.disabled = false;
          submitCreateDsBtn.innerHTML = '<span>⚡ 立即打包并生成数据集</span>';
        }
      }
    });
  }

  async function fetchDatasets() {
    const container = document.getElementById('datasets-grid-container');
    if (!container) return [];
    container.innerHTML = '<div class="loading-datasets-msg">正在刷新本地数据集资产...</div>';

    try {
      const res = await fetch(`${API_BASE}/api/datasets`);
      const items = await res.json();
      container.innerHTML = '';

      // Sync selector UI across LLM & Vision
      syncDatasetSelectors(items);

      if (items.length === 0) {
        container.innerHTML = '<div style="color: #94a3b8; padding: 20px;">暂无数据集</div>';
        return items;
      }

      items.forEach(ds => {
        const card = document.createElement('div');
        card.className = 'dataset-card';
        const isVision = ds.type === 'vision';
        const badgeClass = isVision ? 'ds-badge-vision' : 'ds-badge-text';
        const badgeText = isVision ? '🖼️ 视觉图像' : '📝 文本语言';
        const detailText = isVision ? `类别数: ${ds.classes} 类` : `词表: ${ds.vocab_size ? ds.vocab_size.toLocaleString() : '—'} 字符`;

        card.innerHTML = `
          <div class="ds-card-header">
            <span class="ds-name">${escapeHtml(ds.name)}</span>
            <span class="ds-badge ${badgeClass}">${badgeText}</span>
          </div>
          <div class="ds-meta-row">
            <span>${detailText}</span>
            <span>大小: ${ds.size_mb} MB</span>
          </div>
          <div style="font-size: 11px; color: #64748b; word-break: break-all;">
            路径: <code>${escapeHtml(ds.id)}</code>
          </div>
          <div style="margin-top: auto; display: flex; justify-content: flex-end;">
            <button type="button" class="term-btn delete-ds-btn" data-id="${ds.id}" data-type="${ds.type}" style="color: #f87171; border-color: rgba(248, 113, 113, 0.3);">
              🗑️ 删除资产
            </button>
          </div>
        `;
        container.appendChild(card);
      });

      // Bind delete handlers
      document.querySelectorAll('.delete-ds-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
          const dsId = btn.dataset.id;
          const dsType = btn.dataset.type;
          if (confirm(`确定要删除数据集 [${dsId}] 吗？此操作将清理本地对应数据文件。`)) {
            try {
              const dRes = await fetch(`${API_BASE}/api/datasets/delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ id: dsId, type: dsType })
              });
              const dData = await dRes.json();
              if (dData.success) {
                await fetchDatasets();
              } else {
                alert('删除失败: ' + dData.error);
              }
            } catch (err) {
              alert('删除请求失败: ' + err.message);
            }
          }
        });
      });

      return items;
    } catch (e) {
      const isFileProto = window.location.protocol === 'file:';
      const hint = isFileProto
        ? '您当前正在通过 <code>file://</code> 直接打开网页，被浏览器跨域安全策略拦截。请双击运行 <strong>start.bat</strong> 并访问 <code>http://127.0.0.1:8000</code>！'
        : '无法连接到本地后端服务。请检查是否已在项目根目录运行 <strong>start.bat</strong> 或 <code>python server.py</code>！';
      container.innerHTML = `
        <div style="color: #f87171; padding: 20px; background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; margin: 10px 0;">
          <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px;">❌ 加载数据集列表失败 (${escapeHtml(e.message)})</div>
          <div style="font-size: 13px; color: #cbd5e1; line-height: 1.6;">${hint}</div>
        </div>
      `;
      return [];
    }
  }

  // ==========================================
  // Hardware Detection & GPU Acceleration Modal
  // ==========================================
  const devicePill = document.getElementById('device-pill');
  const hwModal = document.getElementById('hardware-modal');
  const closeHwModalBtn = document.getElementById('close-hw-modal-btn');
  const hwModalBody = document.getElementById('hw-modal-body');
  const deviceHintText = document.getElementById('device-hint-text');

  async function updateHardwareBadge() {
    try {
      const res = await fetch(`${API_BASE}/api/hardware`);
      if (!res.ok) return;
      const hw = await res.json();
      
      if (devicePill) {
        if (hw.cuda_available) {
          devicePill.innerHTML = `<span>🚀 GPU 加速: ${escapeHtml(hw.device_name)} (${hw.vram_total_gb}GB)</span>`;
          devicePill.style.borderColor = 'rgba(52, 211, 153, 0.4)';
          devicePill.style.color = '#34d399';
          devicePill.style.cursor = 'pointer';
        } else if (hw.physical_gpu_detected) {
          devicePill.innerHTML = `<span>⚡ 探测到物理显卡: ${escapeHtml(hw.physical_gpu_name)} (点击查看)</span>`;
          devicePill.style.cursor = 'pointer';
          devicePill.style.borderColor = 'rgba(251, 191, 36, 0.5)';
          devicePill.style.color = '#fbbf24';
        } else {
          devicePill.innerHTML = `<span>💻 运算平台: ${escapeHtml(hw.device_name)}</span>`;
          devicePill.style.cursor = 'pointer';
        }
      }

      if (deviceHintText) {
        if (hw.cuda_available) {
          deviceHintText.textContent = `✅ 已启用 CUDA 硬件加速 (${hw.device_name})，训练效率极高。`;
          deviceHintText.style.color = '#34d399';
        } else if (hw.physical_gpu_detected) {
          deviceHintText.textContent = `ℹ️ 已识别物理显卡 ${hw.physical_gpu_name} (4GB)。当前 Python 环境未启用 CUDA 库，点击上方状态栏可查看 GPU 加速开启步骤。`;
          deviceHintText.style.color = '#fbbf24';
        } else {
          deviceHintText.textContent = `ℹ️ 当前使用 CPU 多线程高效运算。`;
          deviceHintText.style.color = '#94a3b8';
        }
      }

      if (hwModalBody) {
        let statusHtml = '';
        if (hw.cuda_available) {
          statusHtml = `
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
              <div style="font-weight: 600; color: #34d399; font-size: 14px;">✅ GPU / CUDA 加速已就绪！</div>
              <div style="margin-top: 4px; color: #e2e8f0;">当前显卡：${escapeHtml(hw.device_name)} · 显存：${hw.vram_total_gb} GB</div>
            </div>
          `;
        } else if (hw.physical_gpu_detected) {
          statusHtml = `
            <div style="background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
              <div style="font-weight: 600; color: #fbbf24; font-size: 14px;">⚡ 已成功探测到本地物理独立显卡：</div>
              <div style="margin-top: 4px; color: #f8fafc; font-weight: 600;">${escapeHtml(hw.physical_gpu_name)} (${Math.round(hw.physical_gpu_vram_mb/1024)}GB 显存)</div>
              <div style="margin-top: 6px; font-size: 12px; color: #cbd5e1; line-height: 1.5;">
                由于当前 Python 环境中安装的 PyTorch 版本为 <code>${escapeHtml(hw.torch_version)}</code>（CPU 编译版本），尚未包含 NVIDIA CUDA 运行时驱动库，因此 PyTorch 会在 CPU 上以多线程模式平稳运行。
              </div>
            </div>

            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 8px; padding: 14px; margin-bottom: 14px;">
              <div style="font-weight: 600; color: #818cf8; margin-bottom: 6px;">💡 如何开启 GPU 显卡加速？</div>
              <div style="font-size: 12px; color: #cbd5e1; line-height: 1.6;">
                1. 确保系统已安装 NVIDIA 显卡官方驱动程序。<br>
                2. 打开终端控制台，在对应 Python 环境中安装带 CUDA 支持的 PyTorch：<br>
                <code style="display: block; background: #000; padding: 8px 10px; border-radius: 4px; margin: 8px 0; color: #38bdf8; font-family: monospace; font-size: 11px;">
                  pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
                </code>
                3. 重启 <code>start.bat</code>，控制台将自动启用 GPU 显卡加速！
              </div>
            </div>
          `;
        } else {
          statusHtml = `
            <div style="background: rgba(148, 163, 184, 0.1); border: 1px solid rgba(148, 163, 184, 0.3); border-radius: 8px; padding: 12px; margin-bottom: 14px;">
              <div style="font-weight: 600; color: #94a3b8; font-size: 14px;">💻 当前运算平台：CPU 多线程模式</div>
              <div style="margin-top: 4px; color: #e2e8f0;">CPU 逻辑核心数：${hw.cpu_count || '多核'} 线程优化</div>
            </div>
          `;
        }

        hwModalBody.innerHTML = `
          ${statusHtml}
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
            <div style="background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px;">
              <span style="color: #64748b;">CPU 核心数:</span> <strong style="color: #f8fafc;">${hw.cpu_count || 12} 线程</strong>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 8px 12px; border-radius: 6px;">
              <span style="color: #64748b;">PyTorch 版本:</span> <strong style="color: #f8fafc;">${escapeHtml(hw.torch_version || '2.14.1')}</strong>
            </div>
          </div>
        `;
      }
    } catch (e) {
      console.warn('Failed to fetch hardware info:', e);
    }
  }

  if (devicePill) {
    devicePill.addEventListener('click', () => {
      if (hwModal) hwModal.style.display = 'flex';
    });
  }
  if (closeHwModalBtn && hwModal) {
    closeHwModalBtn.addEventListener('click', () => {
      hwModal.style.display = 'none';
    });
  }
  if (hwModal) {
    hwModal.addEventListener('click', (e) => {
      if (e.target === hwModal) hwModal.style.display = 'none';
    });
  }

  // Initial load
  fetchDatasets();
  updateHardwareBadge();

});

