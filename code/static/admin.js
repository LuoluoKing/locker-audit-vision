/* ============================================================
   云端管理后台脚本（原生 JS + Canvas，无外部依赖）
   ============================================================ */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const pageState = { page: 1, size: 20, total: 0, catalog: {} };

  const esc = (s) => String(s === null || s === undefined ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function csvSafeLocal(dt) {          // datetime-local -> ISO 字符串
    return dt ? dt.replace('T', ' ') + ':00' : '';
  }

  /* ---------------------------------------------------- 筛选条件 */
  function filters() {
    return {
      type: $('fType').value,
      start: csvSafeLocal($('fStart').value),
      end: csvSafeLocal($('fEnd').value),
      q: $('fQ').value.trim(),
      abnormal: $('fAbnormal').value
    };
  }
  function qs(extra) {
    const f = filters();
    const p = new URLSearchParams();
    if (f.type) p.set('type', f.type);
    if (f.start) p.set('start', f.start);
    if (f.end) p.set('end', f.end);
    if (f.q) p.set('q', f.q);
    if (f.abnormal === '1') p.set('abnormal', '1');
    Object.keys(extra || {}).forEach((k) => p.set(k, extra[k]));
    return p.toString();
  }

  /* ---------------------------------------------------- KPI */
  function renderKpis(s) {
    const items = [
      ['事件总数', s.total, '条', '数据库累计记录'],
      ['异常事件', s.abnormal, '条', '徘徊观察 / 柜门未关 / 一人取多份 / 取走未关柜门外卖'],
      ['正常事件', s.normal, '条', '区域进入等追溯记录'],
      ['平均置信度', (s.avg_conf * 100).toFixed(1), '%', '全部事件检测置信度均值'],
      ['覆盖素材', s.sources, '个', '已处理过的视频源数量'],
      ['时间范围', s.first_ts ? s.first_ts.slice(5, 10) : '—', '', s.last_ts ? '至 ' + s.last_ts.slice(0, 16).replace('T', ' ') : '暂无记录']
    ];
    $('kpis').innerHTML = items.map((i) =>
      '<div class="kpi"><div class="k">' + i[0] + '</div><div class="v">' + i[1] +
      (i[2] ? '<small>' + i[2] + '</small>' : '') + '</div><div class="sub">' + esc(i[3]) + '</div></div>').join('');
  }

  /* ---------------------------------------------------- 分布条 */
  function renderDist(byType) {
    const cat = pageState.catalog;
    const keys = Object.keys(cat).length ? Object.keys(cat) : byType.map((r) => r.event_type);
    const map = {};
    byType.forEach((r) => { map[r.event_type] = r.cnt; });
    const max = Math.max(1, ...keys.map((k) => map[k] || 0));
    $('dist').innerHTML = keys.map((k) => {
      const n = map[k] || 0, sev = (cat[k] && cat[k].severity) || 'info', name = (cat[k] && cat[k].name) || k;
      return '<div class="dist-row" title="' + esc(k) + '">' +
        '<span>' + esc(name) + '</span>' +
        '<span class="track"><span class="fill sev-' + sev + '" style="width:' + (n / max * 100) + '%"></span></span>' +
        '<span class="num">' + n + '</span></div>';
    }).join('');
  }

  /* ---------------------------------------------------- 趋势图（Canvas） */
  // 本地日期格式化：不能用 toISOString()——它返回 UTC 日期，
  // 在东八区凌晨会让整个 14 天分桶错位一天，导致所有数据匹配不上、图表静默显示为空。
  function ymd(d) {
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, '0');
    const dd = String(d.getDate()).padStart(2, '0');
    return y + '-' + m + '-' + dd;
  }

  function drawTrend(rows) {
    const cvs = $('trend');
    const dpr = window.devicePixelRatio || 1;
    // clientWidth 在首次布局未完成时可能为 0，退化会导致图表错乱，故设兜底宽度
    const W = cvs.clientWidth || (cvs.parentElement && cvs.parentElement.clientWidth) || 560;
    const H = 188;
    cvs.width = W * dpr; cvs.height = H * dpr;
    cvs.style.height = H + 'px';
    const g = cvs.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    const pad = { l: 38, r: 12, t: 12, b: 26 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;

    // 补齐最近 14 天（本地时区）
    const days = [];
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    for (let i = 13; i >= 0; i--) {
      const d = new Date(today.getTime() - i * 86400000);
      days.push({ day: ymd(d), total: 0, abnormal: 0 });
    }
    const map = {}; rows.forEach((r) => { map[r.day] = r; });
    days.forEach((d) => { if (map[d.day]) { d.total = map[d.day].total || 0; d.abnormal = map[d.day].abnormal || 0; } });

    const maxV = Math.max(1, ...days.map((d) => d.total));
    const y = (v) => pad.t + ih - (v / maxV) * ih;

    // 网格
    g.strokeStyle = '#edf1f6'; g.lineWidth = 1;
    g.fillStyle = '#8a97a8'; g.font = '11px "Segoe UI","Microsoft YaHei",sans-serif';
    for (let i = 0; i <= 4; i++) {
      const v = Math.round(maxV * i / 4);
      const yy = Math.round(y(v)) + 0.5;
      g.beginPath(); g.moveTo(pad.l, yy); g.lineTo(pad.l + iw, yy); g.stroke();
      g.textAlign = 'right'; g.textBaseline = 'middle';
      g.fillText(String(v), pad.l - 7, yy);
    }

    const step = iw / days.length;
    const bw = Math.max(6, step * 0.52);

    days.forEach((d, i) => {
      const cx = pad.l + step * i + step / 2;
      g.fillStyle = '#cfe0f0';
      const hT = (d.total / maxV) * ih;
      g.fillRect(cx - bw / 2, pad.t + ih - hT, bw, hT);
      if (d.abnormal > 0) {
        g.fillStyle = '#c0392b';
        const hA = (d.abnormal / maxV) * ih;
        g.fillRect(cx - bw / 2, pad.t + ih - hA, bw, hA);
      }
      // 从最后一天倒推每隔一天标注，保证「今天」始终有刻度标签
      if ((days.length - 1 - i) % 2 === 0 || days.length <= 8) {
        g.fillStyle = '#8a97a8'; g.textAlign = 'center'; g.textBaseline = 'top';
        g.fillText(d.day.slice(5), cx, pad.t + ih + 6);
      }
    });

    // 基线
    g.strokeStyle = '#dfe5ee';
    g.beginPath(); g.moveTo(pad.l, pad.t + ih + 0.5); g.lineTo(pad.l + iw, pad.t + ih + 0.5); g.stroke();
  }

  /* ---------------------------------------------------- 表格 */
  function renderEvents(items) {
    const tb = $('tblEvents').querySelector('tbody');
    if (!items.length) { tb.innerHTML = '<tr><td colspan="8" class="muted">没有符合条件的事件记录</td></tr>'; return; }
    tb.innerHTML = items.map((e) => {
      const snap = e.snapshot
        ? '<img class="thumb" data-lightbox="/snapshots/' + encodeURIComponent(e.snapshot.split('/').pop()) + '" src="/snapshots/' + encodeURIComponent(e.snapshot.split('/').pop()) + '" alt="截图">'
        : '<span class="muted">—</span>';
      return '<tr>' +
        '<td class="mono">' + esc((e.ts || '').replace('T', ' ')) + '</td>' +
        '<td>' + esc(e.event_name) + '<div class="muted mono" style="font-size:11px">' + esc(e.event_type) + '</div></td>' +
        '<td><span class="tag ' + esc(e.severity) + '">' + esc(e.severity) + '</span></td>' +
        '<td>' + esc(e.detail) + '</td>' +
        '<td class="mono">' + (e.confidence !== null ? Number(e.confidence).toFixed(3) : '—') + '</td>' +
        '<td class="mono">' + (e.track_id === null ? 'scene' : e.track_id) + '</td>' +
        '<td class="mono" style="font-size:11.5px">' + esc(e.source || '') + '</td>' +
        '<td>' + snap + '</td></tr>';
    }).join('');
  }

  function renderDevices(devices) {
    const tb = $('tblDevices').querySelector('tbody');
    if (!devices.length) { tb.innerHTML = '<tr><td colspan="10" class="muted">暂无设备上报记录</td></tr>'; return; }
    tb.innerHTML = devices.map((d) => {
      const live = (Date.now() - new Date(d.updated_at.replace('T', ' ')).getTime()) < 15000;
      return '<tr>' +
        '<td class="mono">' + esc(d.device_id) + '</td>' +
        '<td>' + esc(d.device_name) + '</td>' +
        '<td class="mono" style="font-size:11.5px">' + esc(d.source) + '</td>' +
        '<td class="mono">' + esc(d.backend) + '</td>' +
        '<td><span class="badge ' + (live ? 'ok' : 'warn') + '"><span class="dot"></span>' +
        esc(d.status) + (live ? '（在线）' : '（离线）') + '</span></td>' +
        '<td class="mono">' + Number(d.fps || 0).toFixed(2) + '</td>' +
        '<td class="mono">' + Math.round(d.infer_ms || 0) + ' ms</td>' +
        '<td class="mono">' + (d.frame_index || 0) + '</td>' +
        '<td class="mono">' + (d.event_count || 0) + '</td>' +
        '<td class="mono" style="font-size:11.5px">' + esc((d.updated_at || '').replace('T', ' ')) + '</td></tr>';
    }).join('');
  }

  function renderSessions(rows) {
    const tb = $('tblSessions').querySelector('tbody');
    if (!rows.length) { tb.innerHTML = '<tr><td colspan="11" class="muted">暂无运行记录</td></tr>'; return; }
    tb.innerHTML = rows.map((s) => '<tr>' +
      '<td class="mono">' + s.id + '</td>' +
      '<td class="mono" style="font-size:11.5px">' + esc(s.source) + '</td>' +
      '<td class="mono">' + esc(s.backend) + '</td>' +
      '<td class="mono">' + esc(s.resolution) + '</td>' +
      '<td class="mono" style="font-size:11.5px">' + esc((s.started_at || '').replace('T', ' ')) + '</td>' +
      '<td class="mono" style="font-size:11.5px">' + esc((s.ended_at || '运行中').replace('T', ' ')) + '</td>' +
      '<td class="mono">' + s.processed_frames + (s.total_frames ? ' / ' + s.total_frames : '') + '</td>' +
      '<td class="mono">' + Number(s.elapsed_s || 0).toFixed(1) + '</td>' +
      '<td class="mono">' + Number(s.avg_fps || 0).toFixed(2) + '</td>' +
      '<td class="mono">' + Number(s.avg_infer_ms || 0).toFixed(1) + '</td>' +
      '<td class="mono">' + (s.event_count || 0) + '</td></tr>').join('');
  }

  /* ---------------------------------------------------- 数据拉取 */
  async function refreshStats() {
    const r = await fetch('/api/stats?' + qs({ days: 14 }));
    const d = await r.json();
    pageState.catalog = d.catalog || {};
    renderKpis(d.summary);
    renderDist(d.by_type || []);
    drawTrend(d.by_day || []);

    const badge = $('badgeState');
    const running = d.live && d.live.running;
    badge.className = 'badge ' + (running ? 'ok run' : '');
    badge.innerHTML = '<span class="dot"></span><span>边缘节点 ' + (running ? '运行中' : '已停止') + '</span>';
    badge.title = (d.live && d.live.source) || '';
  }

  async function refreshEvents() {
    const r = await fetch('/api/events?' + qs({ page: pageState.page, size: pageState.size }));
    const d = await r.json();
    pageState.total = d.total;
    renderEvents(d.items || []);
    const pages = Math.max(1, Math.ceil(d.total / pageState.size));
    $('pageInfo').textContent = '共 ' + d.total + ' 条 · 第 ' + pageState.page + ' / ' + pages + ' 页';
    $('btnPrev').disabled = pageState.page <= 1;
    $('btnNext').disabled = pageState.page >= pages;
  }

  async function refreshDevices() {
    try { renderDevices((await (await fetch('/api/devices')).json()).devices || []); } catch (e) { }
  }
  async function refreshSessions() {
    try { renderSessions((await (await fetch('/api/sessions?limit=15')).json()).sessions || []); } catch (e) { }
  }
  async function refreshAll() {
    try { await refreshStats(); await refreshEvents(); refreshDevices(); refreshSessions(); } catch (e) { }
  }

  /* ---------------------------------------------------- 事件绑定 */
  $('btnQuery').onclick = () => { pageState.page = 1; refreshAll(); };
  $('btnReset').onclick = () => {
    $('fType').value = ''; $('fStart').value = ''; $('fEnd').value = '';
    $('fQ').value = ''; $('fAbnormal').value = '0';
    pageState.page = 1; refreshAll();
  };
  $('btnPrev').onclick = () => { if (pageState.page > 1) { pageState.page--; refreshEvents(); } };
  $('btnNext').onclick = () => { pageState.page++; refreshEvents(); };
  $('pageSize').onchange = (e) => { pageState.size = parseInt(e.target.value, 10); pageState.page = 1; refreshEvents(); };
  $('fQ').addEventListener('keydown', (e) => { if (e.key === 'Enter') { pageState.page = 1; refreshAll(); } });
  $('btnExport').onclick = () => { window.location.href = '/api/export.csv?' + qs(); };

  document.addEventListener('click', (ev) => {
    const t = ev.target;
    if (t.tagName === 'IMG' && t.dataset.lightbox) {
      $('lbImg').src = t.dataset.lightbox; $('lightbox').classList.add('on');
    }
  });
  $('lbClose').onclick = () => $('lightbox').classList.remove('on');
  $('lightbox').onclick = (e) => { if (e.target.id === 'lightbox') $('lightbox').classList.remove('on'); };
  window.addEventListener('resize', () => { refreshStats().catch(() => { }); });

  /* ---------------------------------------------------- 初始化 */
  (async function init() {
    try {
      const d = await (await fetch('/api/stats?days=14')).json();
      pageState.catalog = d.catalog || {};
      const sel = $('fType');
      sel.innerHTML = '<option value="">全部类型</option>' + Object.keys(pageState.catalog).map((k) =>
        '<option value="' + esc(k) + '">' + esc(pageState.catalog[k].name) + '（' + esc(k) + '）</option>').join('');
    } catch (e) { }
    await refreshAll();
    setInterval(refreshStats, 5000);
    setInterval(refreshEvents, 8000);
    setInterval(refreshDevices, 5000);
    setInterval(refreshSessions, 15000);
  })();
})();
