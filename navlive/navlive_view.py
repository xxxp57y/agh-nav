#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_view.py —— 实时导航**工作台**（自带控制面板，不依赖 DSH 的插件安装流程）。

参考实例（院递工作台）的功能拆开看就是这几块，这里按同一套路做，但换成导航场景：

  ① 任务安排   选场景 / 选那条自然语言任务 / 载入
  ② 派送指令   用人话说去哪 → 解析成真实坐标 → 真的让机器人走
  ③ 地名册     场景里"叫得出名字的地方"，点一下直接派过去
  ④ 演示控制   暂停 / 恢复 / 重开本轮
  ⑤ 机器人状态 目标 / 剩余距离 / 已走步数 / 碰撞 / plan_age / 回合判定
  ⑥ 画面·地图·事件  第一人称渲染 + 整层楼平面图与轨迹 + 决策事件流

**交互闭环**：所有按钮都打真实的 HTTP 请求到 navlive 服务，不是演示动画。
纯标准库（http.server），不引入任何依赖。

用法（服务器上）::

    python3 navlive_view.py --port 43122 --navlive http://127.0.0.1:43121
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
import urllib.parse
from typing import Any

DEFAULT_NAVLIVE = os.environ.get("NAVLIVE_URL", "http://127.0.0.1:43121")

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Agnes · 实时导航工作台</title>
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --panel2: #1c2330; --line: #30363d; --text: #e6edf3;
    --muted: #8b949e; --accent: #e85a14; --ok: #22c55e; --warn: #f59e0b; --info: #58a6ff;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 13px/1.5 -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  header { display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
           padding: 9px 14px; border-bottom: 1px solid var(--line); background: var(--panel);
           position: sticky; top: 0; z-index: 10; }
  h1 { font-size: 14px; margin: 0; font-weight: 600; }
  .badge { padding: 1px 8px; border-radius: 999px; border: 1px solid var(--line);
           color: var(--muted); font-size: 11px; white-space: nowrap; }
  .badge.ok { color: var(--ok); border-color: var(--ok); }
  .badge.warn { color: var(--warn); border-color: var(--warn); }
  .badge.info { color: var(--info); border-color: var(--info); }
  main { display: grid; grid-template-columns: 310px minmax(0,1fr); gap: 12px; padding: 12px; align-items: start; }
  @media (max-width: 1100px) { main { grid-template-columns: minmax(0,1fr); } }
  .right { display: grid; grid-template-columns: minmax(0,1fr) minmax(0,1.05fr); gap: 12px; align-items: start; }
  @media (max-width: 920px) { .right { grid-template-columns: minmax(0,1fr); } }
  section { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
            padding: 11px; min-width: 0; }
  section + section { margin-top: 12px; }
  h2 { margin: 0 0 8px; font-size: 12px; font-weight: 600; color: var(--muted);
       display: flex; align-items: center; gap: 6px; }
  img { width: 100%; display: block; border-radius: 8px; background: #000; }
  .mapwrap { position: relative; border-radius: 8px; overflow: hidden; background: #11161d; }
  .mapwrap svg { position: absolute; inset: 0; width: 100%; height: 100%; }
  label { display: block; color: var(--muted); font-size: 11px; margin: 7px 0 3px; }
  select, textarea { width: 100%; background: var(--panel2); color: var(--text);
           border: 1px solid var(--line); border-radius: 6px; padding: 6px 8px; font-size: 12px;
           font-family: inherit; }
  textarea { resize: vertical; min-height: 52px; }
  .row { display: flex; gap: 6px; flex-wrap: wrap; }
  .row > * { flex: 1 1 auto; }
  button { background: var(--panel2); color: var(--text); border: 1px solid var(--line);
           border-radius: 6px; padding: 6px 10px; font-size: 12px; cursor: pointer;
           font-family: inherit; white-space: nowrap; }
  button:hover:not(:disabled) { background: #253044; }
  button:disabled { opacity: .45; cursor: default; }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button.primary:hover:not(:disabled) { background: #2b7fff; }
  button.warn { border-color: #b45309; color: #f59e0b; }
  button.danger { border-color: #b91c1c; color: #f87171; }
  .kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; }
  .kv .k { color: var(--muted); font-size: 11px; white-space: nowrap; }
  .kv .v { font-variant-numeric: tabular-nums; text-align: right; font-size: 12px;
           overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  ul.events, ul.landmarks, ul.tasks { list-style: none; margin: 0; padding: 0; overflow-y: auto; }
  ul.events { max-height: 190px; }
  ul.landmarks { max-height: 150px; }
  ul.tasks { max-height: 230px; }
  ul.events li { display: flex; gap: 7px; font-size: 11px; padding: 3px 6px;
                 border-radius: 5px; background: rgba(139,148,158,.08); margin-bottom: 3px; }
  .ek { color: var(--muted); flex: 0 0 auto; }
  .ed { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .arrived .ek, .waypoint_reached .ek { color: var(--ok); }
  .planner_error .ek, .stalled .ek, .episode_over .ek { color: var(--warn); }
  .plan_changed .ek { color: var(--info); }
  ul.landmarks li { display: flex; gap: 7px; align-items: baseline; font-size: 11px;
                    padding: 3px 7px; border-radius: 5px; background: rgba(44,110,200,.12);
                    margin-bottom: 3px; cursor: pointer; }
  ul.landmarks li:hover { background: rgba(44,110,200,.28); }
  .lm-name { color: #7ab7ff; font-weight: 600; flex: 0 0 auto; }
  .lm-en { color: var(--muted); flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis;
           white-space: nowrap; }
  .lm-pos { font-variant-numeric: tabular-nums; color: var(--muted); flex: 0 0 auto; }
  ul.tasks li { display: flex; gap: 7px; align-items: baseline; font-size: 11px; padding: 4px 6px;
                border-radius: 5px; margin-bottom: 3px; background: rgba(139,148,158,.08); cursor: pointer; }
  ul.tasks li:hover { background: rgba(31,111,235,.22); }
  ul.tasks li.active { background: rgba(31,111,235,.3); border: 1px solid #1f6feb; }
  .t-id { color: var(--muted); flex: 0 0 auto; font-family: ui-monospace, monospace; }
  .t-instr { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .hint { color: var(--muted); font-size: 11px; line-height: 1.45; }
  .toast { position: fixed; right: 14px; bottom: 14px; z-index: 40; max-width: 400px;
           background: #1c2330; border: 1px solid var(--line); border-radius: 8px;
           padding: 9px 11px; font-size: 12px; box-shadow: 0 6px 20px rgba(0,0,0,.45); }
  .toast.ok { border-color: var(--ok); } .toast.err { border-color: #b91c1c; }
  .sep { height: 1px; background: var(--line); margin: 9px 0; }
  code { background: rgba(139,148,158,.16); padding: 1px 4px; border-radius: 4px; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Agnes · 实时导航工作台</h1>
  <span class="badge" id="svc">连接中…</span>
  <span class="badge" id="scene">—</span>
  <span class="badge" id="steps">—</span>
  <span class="badge" id="state">—</span>
</header>

<main>
  <div class="left">
    <section>
      <h2>① 任务安排</h2>
      <label for="sceneSel">场景（已跑好的 Matterport3D 场景）</label>
      <select id="sceneSel"></select>
      <label>这个场景里的自然语言任务（点一条选中）</label>
      <ul class="tasks" id="taskList"></ul>
      <div class="sep"></div>
      <button class="primary" id="btnOpen" style="width:100%">载入场景 / 任务</button>
      <div class="hint" style="margin-top:6px">换场景要重载 .glb（8–20 秒）；同场景换任务约 3 秒。</div>
    </section>

    <section>
      <h2>② 派送指令</h2>
      <label for="instr">用人话说要它去哪（会解析成真实坐标）</label>
      <textarea id="instr" placeholder="例：带客人去客厅&#10;例：先去厨房，再回卧室"></textarea>
      <div class="row" style="margin-top:6px">
        <button class="primary" id="btnDispatch">派送</button>
        <button id="btnClear">清空</button>
      </div>
      <div class="hint" style="margin-top:6px">
        名字取自下面的地名册；解不出会明确说"没有这个地点"，不会瞎走。
        写「先A再B」会用 <code>、</code><code>，</code><code>然后</code><code>再</code><code>→</code> 拆成两个航点。
        Ctrl+Enter 也能派送。
      </div>
    </section>

    <section>
      <h2>③ 地名册 <span class="badge" id="lmcount">—</span></h2>
      <div class="hint">点一个名字直接派它过去。</div>
      <ul class="landmarks" id="landmarks"></ul>
    </section>

    <section>
      <h2>④ 演示控制</h2>
      <div class="row">
        <button class="warn" id="btnPause">暂停</button>
        <button id="btnResume">恢复</button>
      </div>
      <div class="row" style="margin-top:6px">
        <button class="danger" id="btnRestart">重开本轮</button>
      </div>
      <div class="hint" style="margin-top:6px">
        走到成功点后 Habitat 会结束本回合，此时再派远处地标不会执行 ——
        按「重开本轮」把它放回起点再派活。
      </div>
    </section>

    <section>
      <h2>⑤ 机器人状态</h2>
      <div class="kv" id="telemetry"></div>
    </section>
  </div>

  <div class="right">
    <div>
      <section>
        <h2>实时仿真画面（第一人称）<span class="badge" id="fps">—</span></h2>
        <img id="frame" alt="第一人称画面" />
      </section>
      <section>
        <h2>决策 / 事件流</h2>
        <ul class="events" id="events"></ul>
      </section>
    </div>
    <div>
      <section>
        <h2>楼层图与实际轨迹</h2>
        <div class="mapwrap">
          <img id="map" alt="楼层图" />
          <svg id="overlay" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        <div class="hint" style="margin-top:6px">
          蓝方块 = 地名册上的地点；橙线 = 实际走过；橙圆 = 机器人；绿点 = 本回合原始终点。
        </div>
      </section>
      <section>
        <h2>最近一次操作（核对用）</h2>
        <div class="hint" id="lastResult">还没有操作。</div>
      </section>
    </div>
  </div>
</main>

<div class="toast" id="toast" style="display:none"></div>

<script>
const NAVLIVE = '__NAVLIVE_BASE__';
const $ = (id) => document.getElementById(id);
let session = null, scenes = [], pickedScene = null, pickedEpisode = null, lastSeq = 0;

const KIND = {
  session_opened: '打开任务', plan_changed: '改派目标', waypoint_reached: '到达航点',
  arrived: '到达终点', episode_over: '回合结束', paused: '暂停', resumed: '恢复',
  planner_error: '规划失败', stalled: '卡住', note: '备注', session_closed: '关闭任务',
};
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(d);
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function toast(msg, kind) {
  const el = $('toast');
  el.textContent = msg;
  el.className = 'toast' + (kind ? ' ' + kind : '');
  el.style.display = 'block';
  clearTimeout(el._t);
  el._t = setTimeout(() => { el.style.display = 'none'; }, 7000);
}

async function api(path, opts) {
  const res = await fetch(NAVLIVE + path, Object.assign({ cache: 'no-store' }, opts || {}));
  const text = await res.text();
  let body;
  try { body = JSON.parse(text); } catch (e) { body = { raw: text }; }
  if (!res.ok) throw new Error(body && (body.detail || body.error) ? (body.detail || body.error) : ('HTTP ' + res.status));
  return body;
}

function setLast(html) { $('lastResult').innerHTML = html; }

async function loadScenes() {
  const data = await api('/scenes');
  scenes = data.scenes || [];
  $('sceneSel').innerHTML = scenes.map((s) =>
    `<option value="${esc(s.scene_id)}">${esc(s.scene_id)}（${s.episode_count} 个任务）</option>`).join('');
  pickedScene = scenes.length ? scenes[0].scene_id : null;
  $('sceneSel').onchange = (ev) => { pickedScene = ev.target.value; pickedEpisode = null; renderTasks(); };
  renderTasks();
}

function renderTasks() {
  const scene = scenes.find((s) => s.scene_id === pickedScene);
  const list = $('taskList');
  if (!scene) { list.innerHTML = ''; return; }
  list.innerHTML = scene.episodes.map((e) => {
    const active = (pickedEpisode === String(e.episode_id)) ? ' class="active"' : '';
    return `<li${active} data-ep="${esc(e.episode_id)}"><span class="t-id">#${esc(e.episode_id)}</span>` +
           `<span class="t-instr">${esc(String(e.instruction || '').trim())}</span></li>`;
  }).join('');
  [...list.querySelectorAll('li')].forEach((li) => {
    li.onclick = () => { pickedEpisode = li.dataset.ep; renderTasks(); };
  });
}

async function openScene() {
  if (!pickedScene) { toast('先选一个场景', 'err'); return; }
  toast('正在载入（换场景需 8–20 秒）…');
  try {
    const data = await api('/open', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scene: pickedScene, episode: pickedEpisode || null }),
    });
    const s = data.state || data;
    session = s.session_id;
    lastSeq = 0; $('events').innerHTML = '';
    toast('已载入 ' + s.scene_id + ' · 任务 #' + s.episode_id, 'ok');
    setLast('<code>POST /open</code> → 场景 ' + esc(s.scene_id) + '，任务 #' + esc(s.episode_id) +
            '<br>该任务原始指令：' + esc(String(s.instruction || '').trim()));
    await tick();
  } catch (e) { toast('载入失败：' + e.message, 'err'); }
}

const SPLIT = /\s*(?:、|，|,|然后|接着|之后|再|->|→|；|;)\s*/;
const STRIP = /^(请|帮我|麻烦)?(先去|去|到|带客人去|带我去|走到|前往|送到|先送|导航到|移动到)/;

async function dispatch(text) {
  const raw = String(text !== undefined ? text : $('instr').value).trim();
  if (!raw) { toast('先说要去哪', 'err'); return; }
  const names = raw.split(SPLIT).map((x) => x.trim()).map((x) => x.replace(STRIP, '').trim()).filter(Boolean);
  if (!names.length) { toast('没解析出地名', 'err'); return; }
  try {
    const data = await api('/action', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session, kind: 'goto_landmark', names, note: raw }),
    });
    const r = data.result || {};
    const resolved = (r.resolved || []).map((x) => `${x.label}(${fmt(x.position[0])},${fmt(x.position[2])})`);
    toast('已派送：' + resolved.join(' → '), 'ok');
    setLast('<code>POST /action goto_landmark</code><br>names = ' + esc(JSON.stringify(names)) +
            '<br>解析 → ' + esc(resolved.join(' → ')));
  } catch (e) {
    toast('派送失败：' + e.message, 'err');
    setLast('<span style="color:#f87171">派送失败：' + esc(e.message) + '</span>');
  }
}

async function simple(kind, extra) {
  try {
    const data = await api('/action', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(Object.assign({ session, kind }, extra || {})),
    });
    toast(kind + ' 完成', 'ok');
    setLast('<code>POST /action ' + esc(kind) + '</code>');
    return data;
  } catch (e) { toast(kind + ' 失败：' + e.message, 'err'); }
}

function worldToMap(a, x, z) {
  if (!a || !a.anchor_map || !a.anchor_world) return null;
  const s = Number(a.meters_per_pixel || 0.01);
  return [a.anchor_map[0] + (x - a.anchor_world[0]) / s,
          a.anchor_map[1] + (z - a.anchor_world[1]) / s];
}

function renderTelemetry(s) {
  const m = s.metrics || {}, pos = (s.pose || {}).position || [];
  const rows = [
    ['任务指令', String(s.instruction || '').trim().slice(0, 46) || '—'],
    ['当前目标', s.target_label || '—'],
    ['位置 (x, z)', `${fmt(pos[0])}, ${fmt(pos[2])}`],
    ['剩余直线距离', `${fmt(m.distance_to_goal)} m`],
    ['已走步数', s.step_count ?? '—'],
    ['碰撞次数', m.collisions ?? '—'],
    ['plan_age', m.plan_age ?? '—'],
    ['回合判定', m.success === 1 ? '成功' : (s.episode_finished ? '已结束' : '进行中')],
    ['计划队列', (s.plan || []).map((p) => p.label).join(' → ') || '—'],
    ['备注', s.notes || '—'],
  ];
  $('telemetry').innerHTML = rows.map(([k, v]) =>
    `<div class="k">${esc(k)}</div><div class="v" title="${esc(v)}">${esc(v)}</div>`).join('');
}

function renderMap(s) {
  const a = s.map || {}, size = a.map_size || [0, 0];
  $('map').src = `${NAVLIVE}/map.png?session=${s.session_id}&t=${Math.round((s.map_ts || 0) * 5) / 5}`;
  if (!size[0]) return;
  const svg = $('overlay');
  svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`);
  const pts = (s.trail || []).map((p) => worldToMap(a, p[0], p[2])).filter(Boolean)
    .map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const robot = (s.pose && s.pose.position) ? worldToMap(a, s.pose.position[0], s.pose.position[2]) : null;
  const goal = a.goal_map;
  const marks = (s.landmarks || []).map((lm) => {
    const rc = worldToMap(a, lm.position[0], lm.position[2]);
    if (!rc) return '';
    return `<g><rect x="${(rc[0]-5).toFixed(1)}" y="${(rc[1]-5).toFixed(1)}" width="10" height="10"` +
           ` fill="#2c6ec8" stroke="#fff" stroke-width="1.5" />` +
           `<text x="${(rc[0]+9).toFixed(1)}" y="${(rc[1]+5).toFixed(1)}" fill="#7ab7ff"` +
           ` font-size="21" font-weight="600">${esc(lm.name)}</text></g>`;
  }).join('');
  svg.innerHTML = [
    marks,
    pts ? `<polyline points="${pts}" fill="none" stroke="#e85a14" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" />` : '',
    goal ? `<circle cx="${goal[0]}" cy="${goal[1]}" r="9" fill="#12a150" stroke="#fff" stroke-width="2" />` : '',
    robot ? `<circle cx="${robot[0]}" cy="${robot[1]}" r="11" fill="#f0780a" stroke="#7a3b00" stroke-width="3" />` : '',
  ].join('');
}

function renderEvents(s) {
  for (const e of (s.events_tail || [])) {
    if (e.seq <= lastSeq) continue;
    lastSeq = e.seq;
    const li = document.createElement('li');
    li.className = e.kind;
    const detail = (e.payload && (e.payload.label || e.payload.reason || e.payload.note)) || '';
    li.innerHTML = `<span class="ek">${esc(KIND[e.kind] || e.kind)}</span><span class="ed">${esc(detail)}</span>`;
    $('events').prepend(li);
  }
  while ($('events').children.length > 40) $('events').lastChild.remove();
}

function renderLandmarks(s) {
  const list = s.landmarks || [];
  $('lmcount').textContent = list.length ? `${list.length} 个` : '空';
  $('landmarks').innerHTML = list.map((lm) =>
    `<li data-name="${esc(lm.name)}"><span class="lm-name">${esc(lm.name)}</span>` +
    `<span class="lm-en">${esc(lm.name_en || '')}</span>` +
    `<span class="lm-pos">(${fmt(lm.position[0])}, ${fmt(lm.position[2])})</span></li>`).join('');
  [...$('landmarks').querySelectorAll('li')].forEach((li) => { li.onclick = () => dispatch(li.dataset.name); });
}

async function tick() {
  try {
    const s = await api('/state' + (session ? '?session=' + encodeURIComponent(session) : ''));
    session = s.session_id || session;
    $('svc').textContent = 'navlive 在线'; $('svc').className = 'badge ok';
    $('scene').textContent = `${s.scene_id} · 任务 #${s.episode_id}`;
    $('steps').textContent = `第 ${s.step_count} 步`;
    const m = s.metrics || {};
    const st = m.success === 1 ? '成功' : (s.episode_finished ? '已结束（需重开本轮）' : (s.paused ? '已暂停' : '进行中'));
    $('state').textContent = st;
    $('state').className = 'badge ' + (st === '进行中' ? 'ok' : (st === '成功' ? 'info' : 'warn'));
    $('btnRestart').disabled = !s.episode_finished;
    const age = Date.now() / 1000 - (s.frame_ts || 0);
    $('fps').textContent = s.frame_ts ? `画面更新于 ${age < 2 ? '刚刚' : age.toFixed(0) + 's 前'}` : '—';
    $('frame').src = `${NAVLIVE}/frame.jpg?session=${session}&t=${Math.round((s.frame_ts || 0) * 5) / 5}`;
    renderTelemetry(s); renderMap(s); renderEvents(s); renderLandmarks(s);
  } catch (e) {
    $('svc').textContent = 'navlive 未连接'; $('svc').className = 'badge warn';
    setLast('<span style="color:#f87171">连不上 navlive：' + esc(e.message) +
            '</span><br>排查：服务器 <code>tmux ls</code> 看 navlive 会话；本机到 43121 的端口转发。');
  }
}

$('btnOpen').onclick = openScene;
$('btnDispatch').onclick = () => dispatch();
$('btnClear').onclick = () => { $('instr').value = ''; };
$('btnPause').onclick = () => simple('pause', { reason: '人工暂停' });
$('btnResume').onclick = () => simple('resume');
$('btnRestart').onclick = async () => {
  try {
    const data = await api('/action', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ session, kind: 'restart' }),
    });
    session = (data.state || {}).session_id || session;
    lastSeq = 0; $('events').innerHTML = '';
    toast('已重开本轮', 'ok');
    setLast('<code>POST /action restart</code> → 回合已重置回起点');
  } catch (e) { toast('重开失败：' + e.message, 'err'); }
};
$('instr').addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) { ev.preventDefault(); dispatch(); }
});

(async function boot() {
  try { await loadScenes(); } catch (e) { toast('拿不到场景清单：' + e.message, 'err'); }
  await tick();
  setInterval(tick, 400);
})();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "navlive-view/2"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("[view] %s\n" % (fmt % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            page = PAGE.replace("__NAVLIVE_BASE__", self.server.navlive_url)  # type: ignore[attr-defined]
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=43122)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--navlive", default=DEFAULT_NAVLIVE)
    args = parser.parse_args()

    httpd = Server((args.host, args.port), Handler)
    httpd.navlive_url = args.navlive.rstrip("/")  # type: ignore[attr-defined]
    sys.stderr.write(
        "[view] 实时导航工作台 http://%s:%d/ → %s\n"
        % (args.host, args.port, httpd.navlive_url)  # type: ignore[attr-defined]
    )
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
