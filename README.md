# agh-nav

把 Habitat 室内导航接入 **Agnes Harness（AGH）原生 MHS / Devices 工作台**：用中文对话派发导航任务，同时查看真实相机、地图、位置、碰撞和执行进度。

机器人是一个 MHS 设备 `habitat-01`。模型负责理解、派活和解释，设备与仿真服务负责执行和停止。

## 功能

- 原生 **Devices / Flow / Activity / World**，通过官方 `agnes_mhs` Python SDK 接入。
- 场景语义地标驱动的单站、多站导览；未知地点拒绝执行。
- 相机 JPEG 更新约 2 Hz；真实 navmesh 占据栅格与位置方向。
- 导航进度、到达和失败结果；Pause / Resume / Cancel / Stop。
- 专用仿真会话、底盘资源互斥、断线停止和控制心跳。
- 保留早期 Agent、离线评估工具和看板插件，便于了解项目演进。

```text
AGH 原生对话 + MHS 插件
          │ AgnesHub / 官方设备协议
     mhs/nav_device.py
          │ HTTP / 专用会话 / 控制心跳
 navlive/navlive_server.py
          │
 Habitat + InternNav + 本地场景数据
```

## 开始使用

需要 Linux 导航主机、可运行的 Habitat / InternNav 环境，以及含 `packages/mhs` 的 AGH 源码。MHS 适配器要求 Python ≥ 3.10，AGH 使用 Node.js ≥ 24.10。导航主机另需 GPU、tmux、curl 和 AGH 原生构建工具。

```sh
git clone https://github.com/xxxp57y/agh-nav.git
cd agh-nav
```

先阅读 [MHS 安装与启动](mhs/README.md)。默认演示路径为 `/srv/agh-demo`，请改成你有权限的独立目录。

本仓库包含项目源码和路线模块，**场景资产、数据集、模型权重、真实地标坐标、实验日志、缓存、账号密钥和会话历史需自行准备**。代码不自动下载这些数据，也不附带原服务器访问配置。场景数据受各自许可约束。

导航服务使用现有 InternNav 配置中的 `eval_cfg.env.env_settings["config_path"]`；通过 `NAVLIVE_CONFIG` 指向可工作的配置，通过 `MULTISCENE_SPLIT` 指定本地存在的数据划分。默认历史配置中的 `val_unseen_multiscene` 是自行生成的实验划分；新环境可使用已准备的 `val_unseen`。

## 不需要 GPU 的验证

将官方 AGH 源码放到相邻目录 `agnes-harness-mhs`，固定到本次适配的版本：

```sh
git clone https://github.com/AgnesAI-Labs/agnes-harness.git ../agnes-harness-mhs
git -C ../agnes-harness-mhs checkout 98bc11572ba162bcdf6ce2b29612a44508faf951
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-mhs.txt
export PYTHONPATH="$(pwd)/../agnes-harness-mhs/packages/mhs/python/src:$(pwd)/src"
python -m unittest discover -s mhs -p test_nav_device.py -v
python scripts/check-source.py
```

Windows 使用 `.venv\Scripts\Activate.ps1`，`PYTHONPATH` 路径以分号分隔。

这些测试使用合成状态，不消耗模型额度，不连接仿真服务。真实协议检查与模型导航另见 [MHS 说明](mhs/README.md)。

## 已记录的演示结果与边界

2026-10-13 的开发环境验收记录：本地行为测试 17/17；官方协议检查 43 项通过、0 项失败，另有 1 项提醒、2 项人工检查和 13 项跳过。运动场景未纳入该轮官方协议检查。

原生 Web 与 Agnes 3.0 Flash 的一次自然语言导航实测：39 步、0 次碰撞、距目标平面直线距离 0.244 m。**这是一条演示记录，不能推算整体导航成功率。** 相机约 2 Hz JPEG，尚未实现 20 Hz H.264。停止机制用于仿真演示，不代表真实硬件安全认证。

这些数字来自原开发环境；公共仓库的 CI 验证合成行为和源码，不重跑 GPU 实验。

## 目录与参考

| 目录 | 内容 |
| --- | --- |
| `mhs/` | 原生设备适配、行为测试、独立启动、协议检查、地图标签补丁 |
| `navlive/` | Habitat HTTP 服务、会话、地图、地标生成、旧控制台 |
| `src/route1/` | 失败事件契约、恢复通道、离线分析与实验模块 |
| `agent/` | 早期独立导航 Agent |
| `nav-agent-plugin/`、`nav-board-plugin/` | 早期 AGH 工具与看板插件 |
| `docs/` | 开发记录与早期使用说明；推荐以当前 MHS 说明为准 |

早期离线工具需要原评估日志与相应 InternNav 扩展，缓存未随仓库发布；它们的历史自检结果不表示新环境可以直接重现这些数据。完整评估复现需要按自己的环境准备配置、数据划分和日志。

[MHS 总览](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/mhs.md) · [设备开发指南](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/mhs-device.md) · [火星基地示例](https://github.com/AgnesAI-Labs/agnes-harness/blob/main/docs/guide/mars-world.md)

## 许可证

本项目原创代码使用 [MIT](LICENSE)。`mhs/world-labels.patch` 含 AGH 上游代码上下文，按 Apache-2.0 发布，附上游许可证与通知。外部框架、SDK 和数据集保留自己的许可，详见 [第三方说明](THIRD-PARTY-NOTICES.md)。
