/* ============================================================
   实时监控台脚本（原生 JS，无外部依赖）
   ============================================================ */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const SEV_CLASS = { info: 'info', warning: 'warning', critical: 'critical' };
  const STATUS_CLASS = [
    [/运行中/, 'ok run'], [/已停止|待机/, ''], [/播放结束/, 'info'],
    [/错误|异常/, 'crit'], [/.*/, 'warn']
  ];
  let catalog = {};
  try { catalog = JSON.parse(document.body.dataset.eventCatalog || '{}'); } catch (e) { catalog = {}; }
  // 素材路径 -> 是否为 real_video/ 下的 AI 生成演示视频（非实拍），由 /api/sources 填充
  const DEMO_SRC = {};

  function esc(s) {
    return String(s === null || s === undefined ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function badgeClass(text) {
    for (const [re, cls] of STATUS_CLASS) if (re.test(text)) return cls;
    return '';
  }
  function showMsg(text, kind) {
    const el = $('msg');
    el.className = 'notice' + (kind ? ' ' + kind : '');
    el.textContent = text;
    el.style.display = text ? 'block' : 'none';
  }

  /* ---------------------------------------------------- 素材列表 */
  // AI 生成演示素材（非实拍）声明横幅。
  // flag 省略时以当前下拉选中项为准（未启动时也能提前声明）；
  // 运行时由后端 ai_demo 标记裁决，避免「选了别的素材但横幅还在」。
  function updateDemoNotice(flag) {
    const el = $('demoNotice');
    if (!el) return;
    const on = (flag === undefined) ? !!DEMO_SRC[$('selSource').value] : !!flag;
    el.style.display = on ? 'block' : 'none';
  }

  async function loadSources(keep) {
    const sel = $('selSource');
    try {
      const r = await fetch('/api/sources');
      const d = await r.json();
      // 优先用后端给出的 current_name（纯文件名）匹配选中项；旧字段 current 是绝对路径
      const curName = d.current_name || '';
      const cur = keep ? sel.value : (curName || d.current);
      const groups = {};
      d.sources.forEach((s) => {
        (groups[s.group] = groups[s.group] || []).push(s);
        DEMO_SRC[s.path] = !!s.ai_demo;
      });
      let html = '';
      Object.keys(groups).forEach((g) => {
        html += '<optgroup label="' + esc(g) + '">';
        groups[g].forEach((s) => {
          html += '<option value="' + esc(s.path) + '"' +
            (s.path === cur || s.name === cur ? ' selected' : '') + '>' +
            esc(s.name) + (s.ai_demo ? '（AI 生成·非实拍）' : '') +
            '（' + s.size_mb + ' MB）</option>';
        });
        html += '</optgroup>';
      });
      if (!html) html = '<option value="">未发现素材，请将视频放入 data/simulated/ 或 data/videos/</option>';
      sel.innerHTML = html;
      updateDemoNotice();
      if (!d.sources.length) showMsg('未发现可用视频素材。请把 mp4/avi 放入 data/simulated/ 或 data/videos/ 后点击“刷新素材”。', 'warn');
    } catch (e) {
      showMsg('素材列表加载失败：' + e.message, 'crit');
    }
  }

  /* ---------------------------------------------------- 控制 */
  async function control(payload, okMsg) {
    ['btnStart', 'btnStop', 'btnRestart'].forEach((i) => { $(i).disabled = true; });
    try {
      const r = await fetch('/api/control', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const d = await r.json();
      if (!d.ok) throw new Error(d.error || '操作失败');
      showMsg(okMsg, '');
      setTimeout(() => showMsg('', ''), 2600);
      refreshStatus();
    } catch (e) {
      showMsg('操作失败：' + e.message, 'crit');
    } finally {
      ['btnStart', 'btnStop', 'btnRestart'].forEach((i) => { $(i).disabled = false; });
    }
  }

  function currentForm() {
    return {
      source: $('selSource').value,
      backend: $('selBackend').value,
      detect_fps: parseFloat($('inpFps').value) || 15,
      display_fps: parseFloat($('inpDisplayFps').value) || 30,
      imgsz: parseInt($('selImgsz').value, 10) || 480,
      loop: $('selLoop').value === '1'
    };
  }

  $('btnStart').onclick = () => control(Object.assign({ action: 'start' }, currentForm()), '已按所选素材启动处理');
  $('btnRestart').onclick = () => control(Object.assign({ action: 'restart' }, currentForm()), '已重新开始');
  $('btnStop').onclick = () => control({ action: 'stop' }, '已停止处理');
  $('btnRefreshSources').onclick = () => loadSources(true);

  /* ---------------------------------------------------- 报警横幅（管理员首要关注） */
  // 两级信息：① 持续态 = 未关柜门标记（出现即常驻，柜门关闭后自动消失）；
  //          ② 事件态 = 60s 内最近一条异常事件（刷新即更新）。
  // 「知道了」只隐藏当前条；出现新告警（新柜号集合 / 新事件 id）会重新弹出。
  let ackKey = null;      // 已确认的告警键
  let lastAlertKey = null;

  function showAlert(sev, title, sub, key) {
    const el = $('alertBar');
    if (key === ackKey) { el.hidden = true; return; }   // 该条已被确认
    el.hidden = false;
    el.dataset.sev = sev === 'critical' ? 'critical' : 'warning';
    $('alertTitle').textContent = title;
    $('alertSub').textContent = sub;
    if (key !== lastAlertKey) {          // 新告警：闪烁两次提示
      el.classList.remove('flash');
      void el.offsetWidth;               // 重启动画
      el.classList.add('flash');
      lastAlertKey = key;
    }
  }

  function updateAlertBar(s, items) {
    const flagged = s.flagged_door_labels || [];
    if (flagged.length) {
      showAlert('warning',
        '柜门未关：' + flagged.join('、'),
        '请现场核实；柜门关闭后标记自动解除',
        'flagged:' + flagged.join(','));
      return;
    }
    const ab = (items || []).find((e) => e.severity && e.severity !== 'info' && e.is_abnormal);
    if (ab) {
      const t = Date.parse(ab.ts || '');
      const fresh = isFinite(t) && (Date.now() - t) < 60000;
      if (fresh) {
        showAlert(ab.severity === 'critical' ? 'critical' : 'warning',
          ab.event_name + '：' + (ab.detail || ''),
          '发生于 ' + (ab.ts || '').replace('T', ' ') + '（近 60 秒内）',
          'evt:' + (ab.id !== undefined ? ab.id : ab.ts + ab.detail));
        return;
      }
    }
    $('alertBar').hidden = true;
  }

  $('alertAck').onclick = () => {
    ackKey = lastAlertKey;
    $('alertBar').hidden = true;
  };

  /* ---------------------------------------------------- 高级设置折叠 */
  $('btnAdv').onclick = () => {
    const row = $('advRow');
    row.hidden = !row.hidden;
    $('btnAdv').textContent = row.hidden ? '高级设置 ▾' : '高级设置 ▴';
    $('btnAdv').setAttribute('aria-expanded', String(!row.hidden));
  };

  /* ---------------------------------------------------- 状态渲染 */
  let lastFrame = -1, lastFrameTs = Date.now(), sameCount = 0;
  let lastItems = [];

  function renderStatus(s) {
    const stateTxt = s.status || (s.running ? '运行中' : '已停止');
    const b1 = $('badgeBackend');
    b1.className = 'badge ' + (s.backend === 'yolov8n' ? 'info' : (s.backend === 'hog' ? 'warn' : ''));
    b1.innerHTML = '<span class="dot"></span><span>检测后端 ' + esc(s.backend || '—') + '</span>';
    b1.title = s.backend_name || '';

    const b2 = $('badgeState');
    b2.className = 'badge ' + badgeClass(stateTxt);
    b2.innerHTML = '<span class="dot"></span><span>' + esc(stateTxt) + '</span>';

    const b3 = $('badgeSource');
    b3.className = 'badge';
    b3.innerHTML = '<span class="dot"></span><span>' + esc(s.source || '未选择素材') + '</span>';

    $('badgeLive').hidden = !s.running;

    $('liveCorner').textContent = (s.resolution || '—') + '  ' + (s.fps || 0).toFixed(1) + ' fps';
    // 运行中以工作线程的标记为准；未运行时跟随下拉选中项
    updateDemoNotice(s.running ? s.ai_demo : undefined);

    const total = s.total_frames || 0;
    const loopNo = total ? Math.floor((s.frame_index || 0) / total) + 1 : 0;
    const frameSub = !total ? '' : (s.loop ? '第 ' + loopNo + ' 轮' : '/ ' + total);
    const metrics = [
      ['已处理帧', (s.frame_index || 0), frameSub],
      ['显示帧率', (s.fps || 0).toFixed(2), 'fps'],
      ['检测帧率', (s.detect_fps || 0).toFixed(2), 'fps'],
      ['单帧推理', Math.round(s.infer_ms || 0), 'ms'],
      ['活动目标', s.active_tracks || 0, '人'],
      ['事件总数', s.event_count || 0, ''],
      ['异常事件', s.abnormal_count || 0, ''],
      ['正常取餐', s.normal_retrievals || 0, '次'],
      ['运行时长', Math.round(s.uptime_s || 0), 's'],
    ];
    $('metrics').innerHTML = metrics.map((m) =>
      '<div class="metric"><div class="k">' + m[0] + '</div><div class="v">' +
      m[1] + (m[2] ? ' <small>' + m[2] + '</small>' : '') + '</div></div>').join('');

    const rules = s.rules || {};
    const flagged = s.flagged_door_labels || [];
    const kv = [
      ['★ 未关柜门柜号', flagged.length ? flagged.join('、') + '（柜门关闭后自动解除）' : '无'],
      ['设备编号', s.device_id || '—'],
      ['设备名称', s.device_name || '—'],
      ['素材类型', s.source_type || '—'],
      ['源视频帧率', s.src_fps ? s.src_fps.toFixed(1) + ' fps' : '未知'],
      ['显示 / 检测帧率', (s.fps || 0).toFixed(1) + ' / ' + (s.detect_fps || 0).toFixed(1) + ' fps'],
      ['检测间隔 / 已跳过', (s.detect_interval || 1) + ' 帧（目标 ' + (s.detect_interval_target || 1)
        + '）/ 已跳过 ' + (s.detect_skipped || 0) + ' 次'],
      ['播放进度', s.total_frames ? ((s.progress || 0) * 100).toFixed(1) + '%' : '实时/图片序列'],
      ['循环播放', s.loop ? '开启' : '关闭'],
      ['检测后端', s.backend_name || '—'],
      ['柜门未关阈值', rules.door_open_alarm_s !== undefined ? rules.door_open_alarm_s + ' s（关门确认 ' + rules.door_close_confirm_s + ' s）' : '—'],
      ['一人取多份阈值', rules.multi_package_count !== undefined ? rules.multi_package_count + ' 份' : '—'],
      ['徘徊判定', rules.loiter_s !== undefined ? rules.loiter_s + ' s / ' + rules.loiter_entries + ' 次' : '—'],
      ['事件冷却', rules.cooldown_s !== undefined ? rules.cooldown_s + ' s' : '—'],
      ['数据库', s.db_path || '—'],
    ];
    $('kvStatus').innerHTML = kv.map((r) =>
      '<div class="k">' + esc(r[0]) + '</div><div class="v">' + esc(r[1]) + '</div>').join('');

    $('cooldownSuppressed').textContent = s.suppressed_by_cooldown || 0;
    renderDist($('dist'), catalog, s.events_by_type || {});
    updateAlertBar(s, lastItems);

    if (s.error) showMsg('运行错误：' + s.error, 'crit');
    else if (!s.running && s.last_error) showMsg('启动失败：' + s.last_error, 'crit');

    // 帧率真实性提示：仅当连续多次轮询帧号都不变、且确实过了足够时间才告警，
    // 避免因浏览器时钟跳变或单次轮询恰好落在同一帧上而误报。
    const same = (s.frame_index === lastFrame);
    if (same) {
      sameCount += 1;
    } else {
      sameCount = 0;
      lastFrame = s.frame_index;
      lastFrameTs = Date.now();
    }
    if (sameCount >= 6 && (Date.now() - lastFrameTs) > 8000 && s.running) {
      showMsg('处理线程连续 ' + sameCount + ' 次轮询无新帧输出（约 ' +
        Math.round((Date.now() - lastFrameTs) / 1000) + ' 秒），请检查素材是否可读或降低推理帧率上限。', 'warn');
    }
  }

  function renderDist(host, cat, counts) {
    const keys = Object.keys(cat).length ? Object.keys(cat) : Object.keys(counts);
    const max = Math.max(1, ...keys.map((k) => counts[k] || 0));
    host.innerHTML = keys.map((k) => {
      const n = counts[k] || 0;
      const sev = (cat[k] && cat[k].severity) || 'info';
      const name = (cat[k] && cat[k].name) || k;
      return '<div class="dist-row" title="' + esc(k) + '">' +
        '<span>' + esc(name) + '</span>' +
        '<span class="track"><span class="fill sev-' + sev + '" style="width:' + (n / max * 100) + '%"></span></span>' +
        '<span class="num">' + n + '</span></div>';
    }).join('') || '<div class="muted" style="font-size:12.5px">尚无数据</div>';
  }

  /* ---------------------------------------------------- 事件流 */
  function snapUrl(rel) {
    // 后端把事件截图以 /snapshots/<文件名> 提供（与后台页同一约定）
    return '/snapshots/' + encodeURIComponent(String(rel).split('/').pop());
  }

  function renderEvents(items) {
    lastItems = items || [];
    const host = $('events');
    if (!items.length) { host.innerHTML = '<div class="empty">暂无事件</div>'; return; }
    host.innerHTML = items.map((e) => {
      const conf = (e.confidence !== undefined && e.confidence !== null)
        ? (Math.round(e.confidence * 1000) / 1000) : '—';
      const thumb = e.snapshot
        ? '<img class="thumb" src="' + snapUrl(e.snapshot) + '" data-lightbox="' +
          snapUrl(e.snapshot) + '" alt="事件截图" loading="lazy">'
        : '';
      return '<div class="event" data-sev="' + esc(e.severity) + '">' +
        '<span class="bar"></span>' + thumb + '<div class="main">' +
        '<div class="top"><span class="name">' + esc(e.event_name) + '</span>' +
        '<span class="tag ' + (SEV_CLASS[e.severity] || '') + '">' + esc(e.severity) + '</span>' +
        '<span class="time">' + esc(String(e.ts || '').replace('T', ' ')) + '</span></div>' +
        '<div class="detail">' + esc(e.detail) + '</div>' +
        '<div class="meta">conf=' + conf +
        '  track=' + (e.track_id === null || e.track_id === undefined ? 'scene' : e.track_id) +
        '</div></div></div>';
    }).join('');
  }

  async function refreshStatus() {
    try {
      const r = await fetch('/api/status');
      renderStatus(await r.json());
    } catch (e) { /* 服务未就绪时静默 */ }
  }
  async function refreshEvents() {
    try {
      const r = await fetch('/api/events?size=20&page=1');
      const d = await r.json();
      renderEvents(d.items || []);
    } catch (e) { /* ignore */ }
  }
  // 注：updateAlertBar 在 renderStatus 中调用，用最近一次事件快照；
  // tick() 每轮同时刷新两者，时序上status略先、事件随后（≤1.5s 差异，可接受）。

  /* ---------------------------------------------------- 灯箱 */
  document.addEventListener('click', (ev) => {
    const t = ev.target;
    if (t.tagName === 'IMG' && t.dataset.lightbox) {
      $('lbImg').src = t.dataset.lightbox; $('lightbox').classList.add('on');
    }
  });
  $('lbClose').onclick = () => $('lightbox').classList.remove('on');
  $('lightbox').onclick = (e) => { if (e.target.id === 'lightbox') $('lightbox').classList.remove('on'); };

  /* ---------------------------------------------------- 启动 */
  // 状态与事件在同一轮刷新中同时拉取，避免「规则命中分布」与「事件流」因轮询周期不同
  // 而出现瞬时不一致（例如分布显示 6 条、事件流已显示 8 条）。
  async function tick() {
    await Promise.all([refreshStatus(), refreshEvents()]);
  }

  loadSources(false).then(() => {
    tick();
    setInterval(tick, 1500);
  });

  // 未启动时也要随下拉选择即时更新「AI 生成演示视频（非实拍）」声明
  $('selSource').addEventListener('change', () => updateDemoNotice());
})();
