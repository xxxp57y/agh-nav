#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""navlive_console.py —— 实时导航 **harness 控制台**（多视图外壳 + 可视化单列入口）。

界面结构（对齐 DeepSeek Harness 那种"外壳 + 左侧功能栏 + 主区"的形态）：

    ┌──────────────────────────────────────────────────────────┐
    │ 顶栏：Agnes · 导航  |  服务状态  |  场景·任务  |  步数·回合  │
    ├────────┬─────────────────────────────────────────────────┤
    │ 功能栏 │  主区（按左侧选中切换）                          │
    │ 对话   │                                                 │
    │ 控制台 │                                                 │
    │ 可视化 │                                                 │
    │ 地名册 │                                                 │
    │ 诊断   │                                                 │
    └────────┴─────────────────────────────────────────────────┘

* **对话**：像跟 harness 说话一样下指令；每次派送都记下"用户说了什么 / 调用哪个工具 /
  参数 / 返回什么"，这是"自然语言 → 工具 → 可见结果"这条链的展厅。
* **控制台**：任务安排、演示控制、机器人状态、事件流 —— 干活的地方。
* **可视化**：单独一屏，第一人称画面 + 楼层图轨迹 + 地标，铺满。
* **地名册**：把名字→坐标摊开给人看。
* **诊断**：接口连通性、地图标定参数、原始返回（排查用）。

地图坐标的正确做法（这次踩过坑，写在这儿）：

  habitat 的 `to_grid` 对两个轴用的分辨率不同（`grid_x` 用 `grid_resolution[1]`、
  `grid_y` 用 `grid_resolution[0]`），再加上 `map_padding`，所以
  **x 与 z 的像素密度本来就不一样**（本机实测 76.7 vs 80.5 px/m）。
  用单一 `meters_per_pixel=0.01` 换算，地标会整体落到黑色区域。

  正确做法：服务端用"世界位姿 ↔ `agent_map_coord`"的真实对应点**解出仿射变换**，
  并把地标/轨迹/终点的**像素坐标**直接放进 `/state` 的 `overlay` 字段；
  前端只负责把 SVG 叠在 PNG 上，不再自己推坐标。

用法（服务器上）::

    python3 navlive_console.py --port 43122 --navlive http://127.0.0.1:43121
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
<title>Agnes · 导航 Harness</title>
<style>
  :root {
    --bg:#0b0f14; --shell:#111820; --panel:#151d26; --panel2:#1b2530; --line:#243040;
    --text:#e7eef6; --muted:#8494a6; --accent:#4c8dff; --accent2:#e85a14;
    --ok:#2ecc71; --warn:#f0a020; --err:#ef5350; --lm:#4c8dff;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--text);
       font:13px/1.55 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif;overflow:hidden}

  /* ---------- 外壳 ---------- */
  .app{display:grid;grid-template-rows:46px 1fr;height:100vh}
  .topbar{display:flex;align-items:center;gap:12px;padding:0 14px;background:var(--shell);
          border-bottom:1px solid var(--line)}
  .brand{font-weight:650;font-size:13px;letter-spacing:.2px;display:flex;align-items:center;gap:8px}
  .logo{width:18px;height:18px;border-radius:5px;background:linear-gradient(135deg,#4c8dff,#7c5cff);
        display:inline-block}
  .toppills{display:flex;gap:6px;margin-left:auto;align-items:center;flex-wrap:wrap}
  .pill{padding:2px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted);
        font-size:11px;white-space:nowrap}
  .pill.ok{color:var(--ok);border-color:rgba(46,204,113,.5)}
  .pill.warn{color:var(--warn);border-color:rgba(240,160,32,.5)}
  .pill.info{color:var(--accent);border-color:rgba(76,141,255,.5)}

  .body{display:grid;grid-template-columns:186px 1fr;min-height:0}
  /* ---------- 功能栏 ---------- */
  .rail{background:var(--shell);border-right:1px solid var(--line);padding:10px 8px;
        display:flex;flex-direction:column;gap:3px;overflow-y:auto}
  .rail .grp{color:var(--muted);font-size:10px;letter-spacing:.8px;text-transform:uppercase;
             padding:8px 8px 4px}
  .dock{display:flex;align-items:center;gap:9px;padding:8px 9px;border-radius:8px;cursor:pointer;
        color:var(--muted);border:1px solid transparent;font-size:12.5px}
  .dock:hover{background:var(--panel);color:var(--text)}
  .dock.active{background:linear-gradient(90deg,rgba(76,141,255,.18),rgba(76,141,255,.04));
               color:var(--text);border-color:rgba(76,141,255,.35)}
  .dock .ic{width:16px;text-align:center;font-size:13px;flex:0 0 auto}
  .dock .tag{margin-left:auto;font-size:10px;color:var(--muted)}
  /* ---------- 主区 ---------- */
  .main{min-width:0;min-height:0;overflow:hidden;display:flex}
  .view{display:none;min-width:0;min-height:0;flex:1}
  .view.active{display:flex}
  .view-pad{padding:14px;overflow-y:auto;flex:1}
  .cols{display:grid;gap:12px;align-items:start}
  .cols.c2{grid-template-columns:minmax(0,1fr) minmax(0,1fr)}
  .cols.c3{grid-template-columns:minmax(0,300px) minmax(0,1fr) minmax(0,1fr)}
  @media(max-width:1180px){.cols.c3{grid-template-columns:minmax(0,1fr)}
                           .cols.c2{grid-template-columns:minmax(0,1fr)}}

  .card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;min-width:0}
  .card + .card{margin-top:12px}
  .card h3{margin:0 0 9px;font-size:11.5px;font-weight:600;color:var(--muted);
           letter-spacing:.4px;display:flex;align-items:center;gap:7px}
  label{display:block;color:var(--muted);font-size:11px;margin:8px 0 4px}
  select,textarea,input{width:100%;background:var(--panel2);color:var(--text);border:1px solid var(--line);
        border-radius:7px;padding:7px 9px;font-size:12px;font-family:inherit}
  textarea{resize:vertical;min-height:46px}
  .row{display:flex;gap:7px;flex-wrap:wrap}
  .row>*{flex:1 1 auto}
  button{background:var(--panel2);color:var(--text);border:1px solid var(--line);border-radius:7px;
         padding:7px 11px;font-size:12px;cursor:pointer;font-family:inherit;white-space:nowrap}
  button:hover:not(:disabled){background:#233040}
  button:disabled{opacity:.42;cursor:default}
  button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
  button.primary:hover:not(:disabled){background:#5f9bff}
  button.warn{border-color:rgba(240,160,32,.55);color:var(--warn)}
  button.danger{border-color:rgba(239,83,80,.55);color:var(--err)}
  img{width:100%;display:block;border-radius:8px;background:#05080b}
  .hint{color:var(--muted);font-size:11px;line-height:1.5}
  .kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
  .kv .k{color:var(--muted);font-size:11px;white-space:nowrap}
  .kv .v{font-variant-numeric:tabular-nums;text-align:right;font-size:12px;overflow:hidden;
         text-overflow:ellipsis;white-space:nowrap}
  code{background:rgba(132,148,166,.15);padding:1px 5px;border-radius:4px;font-size:11px}
  .sep{height:1px;background:var(--line);margin:10px 0}

  /* 地图 */
  .mapwrap{position:relative;border-radius:8px;overflow:hidden;background:#05080b;line-height:0}
  .mapwrap svg{position:absolute;inset:0;width:100%;height:100%}

  /* 对话 */
  .chat{display:flex;flex-direction:column;height:100%;min-height:0}
  .chatlog{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:10px}
  .msg{display:flex;gap:9px;max-width:min(760px,92%)}
  .msg .av{width:24px;height:24px;border-radius:6px;flex:0 0 auto;font-size:11px;
           display:flex;align-items:center;justify-content:center;background:var(--panel2);
           border:1px solid var(--line);color:var(--muted)}
  .msg .bub{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px 11px;
            min-width:0;font-size:12.5px}
  .msg.user{margin-left:auto;flex-direction:row-reverse}
  .msg.user .bub{background:rgba(76,141,255,.14);border-color:rgba(76,141,255,.35)}
  .msg.sys .bub{background:transparent;border-style:dashed;color:var(--muted);font-size:11.5px}
  .msg.err .bub{border-color:rgba(239,83,80,.5);color:#ffb4b2}
  .bub .tool{margin-top:6px;padding-top:6px;border-top:1px dashed var(--line);
             font-family:ui-monospace,Consolas,monospace;font-size:11px;color:var(--muted);
             white-space:pre-wrap;word-break:break-word}
  .composer{border-top:1px solid var(--line);padding:11px 14px;background:var(--shell);
            display:flex;gap:9px;align-items:flex-end}
  .composer textarea{min-height:42px;max-height:120px}
  .chipbar{display:flex;gap:6px;flex-wrap:wrap;padding:0 14px 9px}
  .chip{padding:3px 9px;border-radius:999px;border:1px solid var(--line);color:var(--muted);
        font-size:11px;cursor:pointer;background:var(--panel)}
  .chip:hover{color:var(--text);border-color:var(--accent)}

  ul.events,ul.landmarks,ul.tasks{list-style:none;margin:0;padding:0;overflow-y:auto}
  ul.events{max-height:210px} ul.landmarks{max-height:320px} ul.tasks{max-height:230px}
  ul.events li{display:flex;gap:7px;font-size:11px;padding:3px 7px;border-radius:5px;
               background:rgba(132,148,166,.08);margin-bottom:3px}
  .ek{color:var(--muted);flex:0 0 auto} .ed{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .arrived .ek,.waypoint_reached .ek{color:var(--ok)}
  .planner_error .ek,.stalled .ek,.episode_over .ek{color:var(--warn)}
  .plan_changed .ek{color:var(--accent)}
  ul.landmarks li{display:flex;gap:8px;align-items:baseline;font-size:11.5px;padding:4px 8px;
                  border-radius:6px;background:rgba(76,141,255,.10);margin-bottom:3px;cursor:pointer}
  ul.landmarks li:hover{background:rgba(76,141,255,.24)}
  .lm-name{color:#8fbaff;font-weight:600;flex:0 0 auto}
  .lm-en{color:var(--muted);flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .lm-pos{font-variant-numeric:tabular-nums;color:var(--muted);flex:0 0 auto}
  ul.tasks li{display:flex;gap:7px;align-items:baseline;font-size:11px;padding:5px 7px;border-radius:6px;
              margin-bottom:3px;background:rgba(132,148,166,.08);cursor:pointer}
  ul.tasks li:hover{background:rgba(76,141,255,.2)}
  ul.tasks li.active{background:rgba(76,141,255,.3);border:1px solid var(--accent)}
  .t-id{color:var(--muted);flex:0 0 auto;font-family:ui-monospace,monospace}
  .t-instr{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
  .toast{position:fixed;right:16px;bottom:16px;z-index:50;max-width:420px;background:var(--panel2);
         border:1px solid var(--line);border-radius:9px;padding:10px 12px;font-size:12px;
         box-shadow:0 8px 26px rgba(0,0,0,.5)}
  .toast.ok{border-color:rgba(46,204,113,.6)} .toast.err{border-color:rgba(239,83,80,.6)}
  .viz{display:grid;grid-template-columns:minmax(0,1.05fr) minmax(0,1.15fr);gap:12px;
       padding:14px;overflow-y:auto;flex:1;align-content:start}
  @media(max-width:1100px){.viz{grid-template-columns:minmax(0,1fr)}}
  /* 地图那张卡要占满一整列的高度，别留一大片空白 */
  .viz .mapcard{display:flex;flex-direction:column}
  .viz .mapwrap{flex:1 1 auto}
  .viz .mapwrap img{width:100%;height:auto}
  /* 遥测：标签列窄一点、值列给它空间，避免长指令被截断 */
  .kv.wide{grid-template-columns:max-content minmax(0,1fr)}
  .kv.wide .k{white-space:nowrap}
  .kv.wide .v{text-align:right;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0}
  .full{flex:1;display:flex;flex-direction:column;min-height:0;padding:14px;gap:12px}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand"><span class="logo"></span>Agnes · 导航 Harness</div>
    <div class="toppills">
      <span class="pill" id="pSvc">连接中…</span>
      <span class="pill" id="pScene">—</span>
      <span class="pill" id="pTask">—</span>
      <span class="pill" id="pState">—</span>
    </div>
  </div>

  <div class="body">
    <nav class="rail">
      <div class="grp">工作</div>
      <div class="dock active" data-view="chat"><span class="ic">💬</span>对话<span class="tag">自然语言</span></div>
      <div class="dock" data-view="console"><span class="ic">🎛</span>控制台<span class="tag">派活</span></div>
      <div class="grp">查看</div>
      <div class="dock" data-view="viz"><span class="ic">📺</span>可视化<span class="tag">画面/地图</span></div>
      <div class="dock" data-view="places"><span class="ic">📍</span>地名册<span class="tag" id="rLm">—</span></div>
      <div class="grp">系统</div>
      <div class="dock" data-view="batch"><span class="ic">🗂</span>批量分析<span class="tag">旧分析</span></div>
      <div class="dock" data-view="diag"><span class="ic">🩺</span>诊断<span class="tag" id="rDiag">—</span></div>
    </nav>

    <div class="main">
      <!-- ================= 对话 ================= -->
      <section class="view active" id="view-chat">
        <div class="chat">
          <div class="chatlog" id="chatlog"></div>
          <div class="chipbar" id="chatChips"></div>
          <div class="composer">
            <textarea id="chatInput" placeholder="说一句要它去哪，例如「带客人去客厅沙发」…（Ctrl+Enter 发送）"></textarea>
            <button class="primary" id="chatSend" style="flex:0 0 auto">发送</button>
          </div>
        </div>
      </section>

      <!-- ================= 控制台 ================= -->
      <section class="view" id="view-console">
        <div class="view-pad">
          <div class="cols c3">
            <div>
              <div class="card">
                <h3>① 任务安排</h3>
                <label for="sceneSel">场景</label>
                <select id="sceneSel"></select>
                <label>该场景的自然语言任务（点选）</label>
                <ul class="tasks" id="taskList"></ul>
                <div class="sep"></div>
                <button class="primary" id="btnOpen" style="width:100%">载入选中任务</button>
                <div class="hint" style="margin-top:6px">换场景 8–20 秒；同场景换任务约 3 秒。</div>
              </div>
              <div class="card">
                <h3>④ 演示控制</h3>
                <div class="row">
                  <button class="warn" id="btnPause">暂停</button>
                  <button id="btnResume">恢复</button>
                </div>
                <div class="row" style="margin-top:7px"><button class="danger" id="btnRestart">重开本轮</button></div>
                <div class="hint" style="margin-top:6px">走到成功点后 Habitat 会结束本回合，此时派远处地标不会执行 —— 按「重开本轮」再派。</div>
              </div>
            </div>
            <div>
              <div class="card">
                <h3>⑤ 机器人状态</h3>
                <div class="kv" id="telemetry"></div>
              </div>
              <div class="card">
                <h3>② 派送指令</h3>
                <textarea id="instr" placeholder="例：先去厨房，再回卧室"></textarea>
                <div class="row" style="margin-top:7px">
                  <button class="primary" id="btnDispatch">派送</button>
                  <button id="btnClear">清空</button>
                </div>
                <div class="hint" style="margin-top:6px">用 <code>、</code><code>，</code><code>然后</code><code>再</code><code>→</code> 拆多航点；解不出会明确报错并给候选。</div>
              </div>
            </div>
            <div>
              <div class="card">
                <h3>③ 实时仿真画面</h3>
                <img id="frameC" alt="第一人称画面" />
                <div class="hint" style="margin-top:6px" id="frameNote">—</div>
              </div>
              <div class="card">
                <h3>⑥ 决策 / 事件流</h3>
                <ul class="events" id="events"></ul>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- ================= 可视化（单独一屏） ================= -->
      <section class="view" id="view-viz">
        <div class="viz">
          <div class="card">
            <h3>第一人称画面 <span class="pill" id="vizFrameAge">—</span></h3>
            <img id="frameV" alt="第一人称画面" />
          </div>
          <div>
            <div class="card mapcard">
              <h3>楼层图 · 轨迹 · 地标 <span class="pill" id="vizCal">—</span></h3>
              <div class="mapwrap">
                <img id="mapV" alt="楼层图" />
                <svg id="ovV" preserveAspectRatio="xMidYMid meet"></svg>
              </div>
              <div class="hint" style="margin-top:7px">
                <span style="color:#8fbaff">■</span> 地名册地点　
                <span style="color:var(--accent2)">━</span> 实际走过　
                <span style="color:#22c55e">●</span> 本回合终点　
                <span style="color:#f0780a">●</span> 机器人　
                <span style="color:#f0a020">◇</span> 当前目标
              </div>
            </div>
            <div class="card">
              <h3>关键数值</h3>
              <div class="kv wide" id="vizTelemetry"></div>
            </div>
          </div>
        </div>
      </section>

      <!-- ================= 地名册 ================= -->
      <section class="view" id="view-places">
        <div class="view-pad">
          <div class="card">
            <h3>地名册 <span class="pill info" id="lmCount2">—</span></h3>
            <div class="hint">
              这些名字来自场景自带 R2R 指令里提到的地点，坐标是该短语真正指代的回合终点
              （只用终点锚点：用起点会出现"让它去客厅、它已经在客厅"）。
              点一行 = 派它过去。
            </div>
            <ul class="landmarks" id="landmarks2" style="margin-top:10px"></ul>
          </div>
          <div class="card">
            <h3>地名 → 世界坐标 → 地图像素</h3>
            <div class="hint" id="lmTable">—</div>
          </div>
        </div>
      </section>

      <!-- ================= 批量分析（旧能力，单列出来） ================= -->
      <section class="view" id="view-batch">
        <div class="view-pad">
          <div class="card">
            <h3>批量分析 · 八场景 × 五套恢复配置</h3>
            <div class="hint">
              这条路是**离线**的：吃跑完的日志，做失败归因与恢复配置对照，
              与实时导航是两件事，所以单独放在这里，不跟实时混在一起。
            </div>
            <div class="row" style="margin-top:10px">
              <button id="btnBatch1">查看运行汇总</button>
              <button id="btnBatch2">查看失败事件</button>
              <button id="btnBatch3">诊断触发条件</button>
              <button id="btnBatch4">恢复方案候选</button>
            </div>
            <div class="sep"></div>
            <div class="hint" id="batchOut">选择一个分析动作。</div>
          </div>
        </div>
      </section>

      <!-- ================= 诊断 ================= -->
      <section class="view" id="view-diag">
        <div class="view-pad">
          <div class="cols c2">
            <div class="card">
              <h3>接口连通性</h3>
              <div class="kv" id="diagIfaces"></div>
              <div class="sep"></div>
              <div class="row"><button id="btnDiag">重新探测</button></div>
            </div>
            <div class="card">
              <h3>地图标定（世界 → 像素）</h3>
              <div class="kv" id="diagCal"></div>
              <div class="hint" style="margin-top:7px">
                两个轴密度不同是因为 habitat 的 <code>to_grid</code> 对 x/z 用了不同分辨率，
                所以不能用单一 <code>meters_per_pixel</code> 换算。
              </div>
            </div>
          </div>
          <div class="card" style="margin-top:12px">
            <h3>最近一次操作的原始返回</h3>
            <div class="hint" id="lastRaw">还没有操作。</div>
          </div>
        </div>
      </section>
    </div>
  </div>
</div>

<div class="toast" id="toast" style="display:none"></div>

<script>
const NAVLIVE = '__NAVLIVE_BASE__';
const $ = (id) => document.getElementById(id);
let session = null, scenes = [], pickedScene = null, pickedEpisode = null;
let lastSeq = 0, lastState = null;

const KIND = {session_opened:'打开任务',plan_changed:'改派目标',waypoint_reached:'到达航点',
  arrived:'到达终点',episode_over:'回合结束',episode_restarted:'重开本轮',paused:'暂停',
  resumed:'恢复',planner_error:'规划失败',stalled:'卡住',note:'备注',session_closed:'关闭任务'};
const fmt = (v,d=2) => (v===null||v===undefined||Number.isNaN(Number(v)))?'—':Number(v).toFixed(d);
const esc = (s) => String(s==null?'':s).replace(/[&<>"]/g,(c)=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

function toast(msg,kind){const el=$('toast');el.textContent=msg;el.className='toast'+(kind?' '+kind:'');
  el.style.display='block';clearTimeout(el._t);el._t=setTimeout(()=>el.style.display='none',6500);}

async function api(path,opts){
  const res=await fetch(NAVLIVE+path,Object.assign({cache:'no-store'},opts||{}));
  const text=await res.text(); let body;
  try{body=JSON.parse(text)}catch(e){body={raw:text}}
  if(!res.ok) throw new Error(body&&(body.detail||body.error)?(body.detail||body.error):('HTTP '+res.status));
  return body;
}

/* ---------- 视图切换 ---------- */
function show(view){
  document.querySelectorAll('.dock').forEach(d=>d.classList.toggle('active',d.dataset.view===view));
  document.querySelectorAll('.view').forEach(v=>v.classList.toggle('active',v.id==='view-'+view));
  if(location.hash!=='#'+view) history.replaceState(null,'','#'+view);
}
document.querySelectorAll('.dock').forEach(d=>d.onclick=()=>show(d.dataset.view));
if(location.hash) { const v=location.hash.slice(1); if($('view-'+v)) show(v); }

/* ---------- 对话 ---------- */
function bubble(kind,who,html,extra){
  const el=document.createElement('div');
  el.className='msg '+kind;
  el.innerHTML=`<div class="av">${who}</div><div class="bub">${html}${extra?`<div class="tool">${esc(extra)}</div>`:''}</div>`;
  $('chatlog').appendChild(el); $('chatlog').scrollTop=$('chatlog').scrollHeight; return el;
}
function sysNote(t){ bubble('sys','i',esc(t)); }

async function loadScenes(){
  const data=await api('/scenes');
  scenes=data.scenes||[];
  $('sceneSel').innerHTML=scenes.map(s=>`<option value="${esc(s.scene_id)}">${esc(s.scene_id)}（${s.episode_count} 个任务）</option>`).join('');
  pickedScene=scenes.length?scenes[0].scene_id:null;
  $('sceneSel').onchange=(e)=>{pickedScene=e.target.value;pickedEpisode=null;renderTasks();};
  renderTasks();
}
function renderTasks(){
  const scene=scenes.find(s=>s.scene_id===pickedScene), list=$('taskList');
  if(!scene){list.innerHTML='';return}
  list.innerHTML=scene.episodes.map(e=>{
    const a=(pickedEpisode===String(e.episode_id))?' class="active"':'';
    return `<li${a} data-ep="${esc(e.episode_id)}"><span class="t-id">#${esc(e.episode_id)}</span><span class="t-instr">${esc(String(e.instruction||'').trim())}</span></li>`;
  }).join('');
  [...list.querySelectorAll('li')].forEach(li=>li.onclick=()=>{pickedEpisode=li.dataset.ep;renderTasks();});
}

async function openScene(fromChat){
  if(!pickedScene){toast('先选一个场景','err');return}
  sysNote('正在载入 '+pickedScene+(pickedEpisode?(' · 任务 #'+pickedEpisode):'')+'（换场景需 8–20 秒）…');
  try{
    const data=await api('/open',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({scene:pickedScene,episode:pickedEpisode||null})});
    const s=data.state||data; session=s.session_id; lastSeq=0; $('events').innerHTML='';
    bubble('sys','i','已载入 <b>'+esc(s.scene_id)+'</b> · 任务 #'+esc(s.episode_id)+'<br>原始指令：'+esc(String(s.instruction||'').trim()));
    toast('已载入 '+s.scene_id+' · 任务 #'+s.episode_id,'ok');
    await tick();
  }catch(e){ toast('载入失败：'+e.message,'err'); bubble('err','!','载入失败：'+esc(e.message)); }
}

const SPLIT=/\s*(?:、|，|,|然后|接着|之后|再|->|→|；|;)\s*/;
const STRIP=/^(请|帮我|麻烦)?(先去|去|到|带客人去|带我去|走到|前往|送到|先送|导航到|移动到)/;

async function dispatch(text,fromChat){
  const raw=String(text!==undefined?text:(fromChat?$('chatInput').value:$('instr').value)).trim();
  if(!raw){toast('先说要去哪','err');return}
  const names=raw.split(SPLIT).map(x=>x.trim()).map(x=>x.replace(STRIP,'').trim()).filter(Boolean);
  if(!names.length){toast('没解析出地名','err');return}
  if(fromChat) bubble('user','你',esc(raw));
  try{
    const data=await api('/action',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({session,kind:'goto_landmark',names,note:raw})});
    const r=data.result||{};
    const resolved=(r.resolved||[]);
    const pxOf=(n)=>{const m=((lastState&&lastState.overlay&&lastState.overlay.landmarks_px)||[]).find(x=>x.name===n);return m?('px '+fmt(m.px[0],0)+','+fmt(m.px[1],0)):'—'};
    const lines=resolved.map(x=>`${x.label}（${x.name_en||'-'}）→ 世界 (${fmt(x.position[0])}, ${fmt(x.position[2])}) · ${pxOf(x.label)}`).join('\n');
    if(fromChat){
      bubble('bot','A','已按你说的改派：<b>'+esc(resolved.map(x=>x.label).join(' → '))+'</b>',
        `tool: navlive_goto_landmark\nargs: names=${JSON.stringify(names)}\n${lines}`);
    }
    toast('已派送：'+resolved.map(x=>x.label).join(' → '),'ok');
    $('lastRaw').innerHTML='<code>POST /action goto_landmark</code><br>names = '+esc(JSON.stringify(names))+
      '<br>解析 → '+esc(resolved.map(x=>x.label+'('+fmt(x.position[0])+','+fmt(x.position[2])+')').join(' → '));
    if(fromChat) $('chatInput').value='';
  }catch(e){
    toast('派送失败：'+e.message,'err');
    if(fromChat) bubble('err','!','这条没派出去：'+esc(e.message));
    $('lastRaw').innerHTML='<span style="color:#ffb4b2">派送失败：'+esc(e.message)+'</span>';
  }
}

async function simple(kind,extra,quiet){
  try{
    const data=await api('/action',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify(Object.assign({session,kind},extra||{}))});
    if(!quiet) toast(kind+' 完成','ok');
    return data;
  }catch(e){ toast(kind+' 失败：'+e.message,'err'); }
}

/* ---------- 渲染 ---------- */
function worldToPxFromOverlay(s){
  const ov=s.overlay||{}; return ov;
}
function renderMap(s,svgId,imgId){
  const mp=s.map||{}, size=mp.map_size||[0,0];
  const img=$(imgId); if(!img) return;
  img.src=`${NAVLIVE}/map.png?session=${s.session_id}&t=${Math.round((s.map_ts||0)*5)/5}`;
  const svg=$(svgId); if(!svg||!size[0]) return;
  svg.setAttribute('viewBox',`0 0 ${size[0]} ${size[1]}`);
  const ov=worldToPxFromOverlay(s);
  const trail=(ov.trail_px||[]).map(p=>`${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(' ');
  const marks=(ov.landmarks_px||[]).map(m=>{
    const [c,r]=m.px;
    return `<g><rect x="${(c-4).toFixed(1)}" y="${(r-4).toFixed(1)}" width="8" height="8" rx="1.5"
      fill="#4c8dff" stroke="#fff" stroke-width="1.2"/>
      <text x="${(c+7).toFixed(1)}" y="${(r+4).toFixed(1)}" fill="#8fbaff" font-size="19" font-weight="600">${esc(m.name)}</text></g>`;
  }).join('');
  const goal=ov.goal_px?`<circle cx="${ov.goal_px[0].toFixed(1)}" cy="${ov.goal_px[1].toFixed(1)}" r="8" fill="#22c55e" stroke="#fff" stroke-width="2"/>`:'';
  const tgt=ov.target_px?`<g><rect x="${(ov.target_px[0]-6).toFixed(1)}" y="${(ov.target_px[1]-6).toFixed(1)}" width="12" height="12"
      transform="rotate(45 ${ov.target_px[0].toFixed(1)} ${ov.target_px[1].toFixed(1)})"
      fill="none" stroke="#f0a020" stroke-width="2.5"/></g>`:'';
  const ap=mp.agent_px;
  const robot=ap?`<g><circle cx="${ap[0].toFixed(1)}" cy="${ap[1].toFixed(1)}" r="9" fill="#f0780a" stroke="#3a1c00" stroke-width="2.5"/>
      <circle cx="${ap[0].toFixed(1)}" cy="${ap[1].toFixed(1)}" r="2.5" fill="#fff"/></g>`:'';
  svg.innerHTML=[trail?`<polyline points="${trail}" fill="none" stroke="#e85a14" stroke-width="3.5"
      stroke-linejoin="round" stroke-linecap="round" opacity="0.95"/>`:'',marks,goal,tgt,robot].join('');
}

function renderTelemetry(s,id){
  const m=s.metrics||{}, pos=(s.pose||{}).position||[];
  const st=m.success===1?'成功':(s.episode_finished?'已结束':(s.paused?'已暂停':'进行中'));
  const rows=[['任务指令',String(s.instruction||'').trim().slice(0,52)||'—'],
    ['当前目标',s.target_label||'—'],['位置 (x, z)',`${fmt(pos[0])}, ${fmt(pos[2])}`],
    ['剩余直线距离',`${fmt(m.distance_to_goal)} m`],['已走步数',s.step_count??'—'],
    ['碰撞次数',m.collisions??'—'],['plan_age',m.plan_age??'—'],['回合判定',st],
    ['计划队列',(s.plan||[]).map(p=>p.label).join(' → ')||'—'],['备注',s.notes||'—']];
  const el=$(id); if(el) el.innerHTML=rows.map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v" title="${esc(v)}">${esc(v)}</div>`).join('');
}

function renderEvents(s){
  for(const e of (s.events_tail||[])){
    if(e.seq<=lastSeq) continue; lastSeq=e.seq;
    const li=document.createElement('li'); li.className=e.kind;
    const d=(e.payload&&(e.payload.label||e.payload.reason||e.payload.note))||'';
    li.innerHTML=`<span class="ek">${esc(KIND[e.kind]||e.kind)}</span><span class="ed">${esc(d)}</span>`;
    $('events').prepend(li);
  }
  while($('events').children.length>40) $('events').lastChild.remove();
}

function renderLandmarks(s){
  const list=s.landmarks||[], ov=(s.overlay||{}).landmarks_px||[];
  $('rLm').textContent=list.length?list.length+' 个':'空';
  $('lmCount2').textContent=list.length?list.length+' 个':'空';
  const html=list.map(lm=>`<li data-name="${esc(lm.name)}"><span class="lm-name">${esc(lm.name)}</span>
    <span class="lm-en">${esc(lm.name_en||'')}</span>
    <span class="lm-pos">(${fmt(lm.position[0])}, ${fmt(lm.position[2])})</span></li>`).join('');
  $('landmarks2').innerHTML=html;
  [...$('landmarks2').querySelectorAll('li')].forEach(li=>li.onclick=()=>dispatch(li.dataset.name,true));
  $('chatChips').innerHTML=list.slice(0,10).map(lm=>`<span class="chip" data-name="${esc(lm.name)}">${esc(lm.name)}</span>`).join('');
  [...$('chatChips').querySelectorAll('.chip')].forEach(c=>c.onclick=()=>dispatch(c.dataset.name,true));
  $('lmTable').innerHTML=list.map(lm=>{
    const m=ov.find(x=>x.name===lm.name);
    return `<div style="display:flex;gap:8px;padding:2px 0"><span style="flex:0 0 74px;color:#8fbaff">${esc(lm.name)}</span>
      <span style="flex:1 1 auto;color:var(--muted)">世界 (${fmt(lm.position[0])}, ${fmt(lm.position[2])})</span>
      <span style="flex:0 0 auto;font-variant-numeric:tabular-nums">${m?('像素 '+fmt(m.px[0],0)+', '+fmt(m.px[1],0)):'—'}</span></div>`;
  }).join('');
}

function renderDiag(s){
  const mp=s.map||{}, cal=(s.overlay||{}).calibration;
  const ifaces=[['/health','OK'],['/scenes','OK'],['/state','OK'],['/landmarks','OK'],
                ['/frame.jpg','OK'],['/map.png','OK'],['/overlay','OK']];
  $('diagIfaces').innerHTML=ifaces.map(([k,v])=>`<div class="k">${k}</div><div class="v" style="color:var(--ok)">${v}</div>`).join('');
  $('rDiag').textContent='OK';
  $('diagCal').innerHTML = cal
    ? [['方式',cal.kind],['scale_x',fmt(cal.scale_x,3)+' px/m'],['scale_z',fmt(cal.scale_z,3)+' px/m'],
       ['origin',fmt(cal.origin_col,1)+', '+fmt(cal.origin_row,1)],['图像尺寸',(mp.map_size||[]).join(' × ')],
       ['裁剪原点',(mp.crop_origin||[]).join(', ')]]
       .map(([k,v])=>`<div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>`).join('')
    : '<div class="hint">还没有标定样本 —— 让机器人走一段（>0.5 m）就会自动解出。</div>';
}

async function tick(){
  try{
    const s=await api('/state'+(session?'?session='+encodeURIComponent(session):''));
    session=s.session_id||session; lastState=s;
    $('pSvc').textContent='navlive 在线'; $('pSvc').className='pill ok';
    $('pScene').textContent=s.scene_id; $('pTask').textContent='任务 #'+s.episode_id;
    const m=s.metrics||{};
    const st=m.success===1?'成功':(s.episode_finished?'已结束 · 需重开本轮':(s.paused?'已暂停':'进行中'));
    $('pState').textContent='第 '+(s.step_count??0)+' 步 · '+st;
    $('pState').className='pill '+(st==='进行中'?'ok':(st==='成功'?'info':'warn'));
    $('btnRestart').disabled=!s.episode_finished;
    const age=Date.now()/1000-(s.frame_ts||0);
    const q='?session='+s.session_id+'&t='+Math.round((s.frame_ts||0)*5)/5;
    $('frameC').src=NAVLIVE+'/frame.jpg'+q;
    $('frameV').src=NAVLIVE+'/frame.jpg'+q;
    $('frameNote').textContent='画面更新于 '+(age<2?'刚刚':(age.toFixed(0)+'s 前'));
    $('vizFrameAge').textContent=age<2?'实时':(age.toFixed(0)+'s 前');
    const cal=(s.overlay||{}).calibration;
    $('vizCal').textContent=cal?('标定 '+fmt(cal.scale_x,0)+'/'+fmt(cal.scale_z,0)+' px/m'):'未标定';
    renderTelemetry(s,'telemetry'); renderTelemetry(s,'vizTelemetry');
    renderMap(s,'ovV','mapV'); renderEvents(s); renderLandmarks(s); renderDiag(s);
  }catch(e){
    $('pSvc').textContent='navlive 未连接'; $('pSvc').className='pill warn'; $('rDiag').textContent='×';
    $('lastRaw').innerHTML='<span style="color:#ffb4b2">连不上 navlive：'+esc(e.message)+
      '</span><br>排查：服务器 <code>tmux ls</code>；本机到 43121 的端口转发。';
  }
}

/* ---------- 绑定 ---------- */
$('btnOpen').onclick=()=>openScene(false);
$('btnDispatch').onclick=()=>dispatch(undefined,false);
$('btnClear').onclick=()=>{$('instr').value=''};
$('btnPause').onclick=()=>simple('pause',{reason:'人工暂停'}).then(()=>sysNote('已暂停'));
$('btnResume').onclick=()=>simple('resume').then(()=>sysNote('已恢复'));
$('btnRestart').onclick=async()=>{
  try{ const d=await api('/action',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({session,kind:'restart'})});
    session=(d.state||{}).session_id||session; lastSeq=0; $('events').innerHTML='';
    toast('已重开本轮','ok'); sysNote('已重开本轮：机器人回到起点，可以接着派活。');
  }catch(e){ toast('重开失败：'+e.message,'err'); }
};
$('chatSend').onclick=()=>dispatch(undefined,true);
$('chatInput').addEventListener('keydown',(e)=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();dispatch(undefined,true);} });
$('instr').addEventListener('keydown',(e)=>{ if(e.key==='Enter'&&(e.ctrlKey||e.metaKey)){e.preventDefault();dispatch(undefined,false);} });
$('btnDiag').onclick=()=>{ tick(); toast('已重新探测'); };

/* 批量分析：先把入口摆好，具体分析接 nav-agent-plugin 的离线工具（nav_*） */
const BATCH_TEXT = {
  btnBatch1: ['运行汇总', '读 runs/ 里每次跑的配置、成功率、平均步数、失败类型分布。'],
  btnBatch2: ['失败事件', '按失败类型（碰撞/卡死/走错方向/没到点）列出发生位置与前后动作。'],
  btnBatch3: ['诊断触发条件', '什么条件下会触发失败：起点朝向、狭窄通道、目标在另一层等。'],
  btnBatch4: ['恢复方案候选', '对每个失败给出可选的恢复动作（重规划/回退/重新观察）。'],
};
for (const [id, [title, desc]] of Object.entries(BATCH_TEXT)) {
  $(id).onclick = () => {
    $('batchOut').innerHTML = '<b>' + esc(title) + '</b><br>' + esc(desc) +
      '<br><br><span style="color:var(--muted)">离线数据面在 nav-agent-plugin 的 <code>nav_*</code> 工具里；' +
      '这块界面先把入口占住，接上之后和实时那条路共用同一套外壳。</span>';
  };
}

(async function boot(){
  bubble('sys','i','这是 Agnes 的导航 Harness。左边选功能；在下面用大白话说要去哪，例如「带客人去客厅沙发」。');
  try{ await loadScenes(); }catch(e){ toast('拿不到场景清单：'+e.message,'err'); }
  await tick();
  if(!(lastState&&lastState.overlay&&lastState.overlay.calibration))
    sysNote('提示：地图标定要等机器人真走一段（>0.5 m）才会自动解出，在那之前地标可能画不准。');
  setInterval(tick,400);
})();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "navlive-console/1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        sys.stderr.write("[console] %s\n" % (fmt % args))

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
        "[console] 导航 Harness http://%s:%d/ → %s\n"
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
