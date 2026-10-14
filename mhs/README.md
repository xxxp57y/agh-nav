# Habitat → AGH 原生 Devices

使用官方 `@agnes/mhs` 插件与 Python `agnes_mhs` SDK，将导航服务注册为 `habitat-01`，使用 AGH 原生对话与设备面板。

| 数据 | 来源 |
| --- | --- |
| 相机 | navlive 的 `/frame.jpg`，约 2 Hz JPEG |
| 位置 | Habitat 真值，MHS x=Habitat x, y=-Habitat z |
| 地图 | pathfinder 生成的 5 cm 占据栅格；255 可走、0 不可走 |
| 地点 | 使用者在本地从场景语义标注生成的地标表 |
| 距离 | 当前目标平面直线距离，不是剩余路径长度 |
| 到达 | 真实 waypoint_reached 事件；计划清空不算到达 |
| 失败 | stalled、planner_error、超时等事件 |

适配器默认创建专用会话。底盘资源使用 reject 策略；导航支持 Pause / Resume / Cancel / Stop。MHS 断线时主动暂停；仿真端控制心跳过期后约 4 秒加一个 tick 自动暂停。仅专用会话启用心跳。

## 准备独立目录

路径可自由选择，下例使用 /srv/agh-demo；请使用自己有权限的独立目录。

```text
/srv/agh-demo/
  agh-nav/              # 本仓库
  agh-mhs-runtime/      # 官方 AGH 源码与构建产物
  .secrets/agnes-key.txt # 自己的模型密钥，不提交
  agh-mhs-home/         # 独立实例状态，自动创建
  agh-mhs-workspace/    # 独立 Agent 工作区，自动创建
  agh-mhs-logs/         # 日志，自动创建
```

导航依赖另行准备：可用的 Habitat / InternNav 环境、场景资产与数据划分，以及可工作的 eval 配置。`NAVLIVE_CONFIG` 指向一个 Python 配置，导出 `eval_cfg.env.env_settings["config_path"]`；该字段指向 Habitat YAML。标准 habitat_dual_system_cfg.py 可作为起点，具体数据路径以自己的环境为准。

```sh
cd /srv/agh-demo
git clone https://github.com/AgnesAI-Labs/agnes-harness.git agh-mhs-runtime
git -C agh-mhs-runtime checkout 98bc11572ba162bcdf6ce2b29612a44508faf951
cd agh-mhs-runtime
corepack pnpm install --frozen-lockfile
corepack pnpm --filter @agnes/cli build:local
git apply ../agh-nav/mhs/world-labels.patch
corepack pnpm --filter @agnes/mhs build:plugin

cd ../agh-nav
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mhs.txt
cp .env.example .env
# 编辑 .env，填写自己的目录、Habitat Python、配置、场景和密钥文件路径。
set -a
. ./.env
set +a
```

API 密钥放到 AGH_MHS_KEY_FILE 指定的本地文件，限制文件权限。初次安装通过官方配置服务保存凭据，不打印密钥。也可先通过官方 UI 配置模型。

AGH 官方 SDK 从 $AGH_MHS_RUNTIME/packages/mhs/python/src 加载。适配器的 .venv 无需安装 GPU 依赖；仿真服务用 AGH_HABITAT_PYTHON 指定的已有环境。

## 本地生成地标

公共仓库不携带真实场景坐标。先验证本地 Habitat 能载入所选场景，再生成表：

```sh
export PYTHONPATH="$AGH_MHS_NAVROOT/src:$ROUTE1_REPO"
"$AGH_HABITAT_PYTHON" navlive/mine_semantic_landmarks.py \
  --scenes "$AGH_MHS_SCENE" --out navlive/data/landmarks_semantic.json
# 可选：使用自己的模型翻译地标，可能产生 API 费用。
"$AGH_HABITAT_PYTHON" navlive/translate_landmarks.py \
  --in navlive/data/landmarks_semantic.json --cache navlive/data/zh_glossary.json \
  --key-file "$AGH_MHS_KEY_FILE"
```

生成文件保留在本地。缺少地标时无法执行按地名导航；不要把示例空表当真实坐标。

## 安装与启动

```sh
bash mhs/start-mhs.sh bootstrap
# 从官方 AGH 源码目录运行，确保能解析其 tsx 依赖。
cd "$AGH_MHS_RUNTIME"
node --import tsx "$AGH_MHS_NAVROOT/mhs/setup-native.mts"
cd "$AGH_MHS_NAVROOT"
bash mhs/start-mhs.sh start
bash mhs/start-mhs.sh status
```

安装脚本通过正式包流程安装、信任并启用官方 MHS 包；更新时可能短暂停用本独立实例中的插件。日常启动不需要重复安装。

Web 为 http://127.0.0.1:43130/，导航服务 43131，Hub 43132，聊天通信 43134；服务监听回环地址。需确保端口与专用 agh_mhs_* tmux 名称未被其他部署占用。`bash mhs/start-mhs.sh stop` 停止这套专用实例。

Windows 访问远端时，设置 AGH_MHS_SSH_HOST、AGH_MHS_SSH_PORT、AGH_MHS_SSH_KEY 后运行 mhs/open-mhs.ps1 或双击 打开MHS导航工作台.cmd。也可自行转发：

```sh
ssh -N -L 43130:127.0.0.1:43130 -L 43131:127.0.0.1:43131 \
  -L 43132:127.0.0.1:43132 -L 43134:127.0.0.1:43134 your-user@your-server.example
```

## 演示与验证

原生对话中先“查看机器人状态，列出可以去的地方”，从返回列表选完整名称，再“带我去【完整地标名】”。多个地点按顺序一次派发；改派前先停止当前调用。工作区 AGENTS.md 包含导览规范。

本地合成行为测试见根 README，无需 GPU 或模型账号。真实协议检查会连接专用仿真会话，并执行观察、停止等检查，应在演示空闲时运行：

```sh
bash mhs/check-mhs.sh
node mhs/verify-live.mjs
```

官方检查使用 core / streaming / maps / perception / derived，明确 --no-motion；不把它当导航成功率评估。独立检查端口为 43133。

## 记录与限制

原开发环境验收：行为测试 17/17；协议检查 43 通过、0 失败、1 提醒、2 人工检查、13 跳过；原生 Web 真实模型的一次导航为 39 步、0 碰撞、目标直线距离 0.244 m，该场景有 43 个中文地标。

公共发布不包含原场景衍生坐标和真实报告，新环境需要自己的数据。原验收记录不表示新环境已验收，单次交互不代表整体导航成功率。JPEG 约 2 Hz，未实现高帧率 H.264。停止和心跳用于仿真展示，不代表硬件安全认证。
