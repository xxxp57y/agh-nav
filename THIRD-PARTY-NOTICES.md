# 第三方来源与许可

- Agnes Harness / MHS：外部依赖，不随本仓库分发完整源码或构建产物。来源 https://github.com/AgnesAI-Labs/agnes-harness ，适配基准 `98bc11572ba162bcdf6ce2b29612a44508faf951`。
- `mhs/world-labels.patch` 修改 AGH 的 `packages/mhs/plugin/client/src/layout/world.tsx`，保留上游代码上下文；该补丁按 Apache License 2.0 发布。许可证在 `mhs/APACHE-2.0.txt`，上游通知在 `mhs/AGH-UPSTREAM-NOTICE.txt`。新增修改为中文地标文字的避让布局。
- Habitat、Habitat Baselines、InternNav 与相关 Python / JavaScript 库：由使用者独立安装，遵守各项目许可。本仓库中的 `src/route1` 为项目扩展，调用已有接口。
- Matterport3D / R2R / VLN-CE 的场景、语义标注、指令、数据划分和衍生地标表不随本仓库分发。请自行取得授权并在本地生成地标；本项目 MIT 许可证不授予任何数据集使用或再分发权。
- 公共发布未包含原服务器访问配置、模型密钥、SSH 私钥、用户会话、模型权重、真实评估日志或缓存。

本文件记录来源与范围，不改变各上游许可。
