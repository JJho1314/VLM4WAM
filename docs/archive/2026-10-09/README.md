# 三个 Ola 工作目录的汇总与追溯

采集日期：2026-10-09。原目录均保留，未删除、重置或提交其中的用户修改。

| 来源 | 原 HEAD | 处理方式 |
| --- | --- | --- |
| VLM4WAM_baton | 4a58318 | 保留 Git 历史；与最终代码不同的有效文件存入同名子目录 |
| VLM4WAM_baton_robofollow_autoresearch | 54dc350 | 保留旧实现、自进化设计、背景状态和独有源码 |
| VLM4WAM_robofollow_q35 | 59ad231 | 当前代码主体，包括所有采集到的未提交有效修改 |
| HPC3 official3750 部署 | source_head 59ad231 + 部署修改 | 覆盖对应文件为真实部署版本，保留 code.sha256 和训练辅助脚本 |

`source-map.json` 逐文件记录 source、原相对 path、SHA256、saved_as。相同文件复用当前文件；不同文件存入来源子目录；入口 README 的原文放在 entry-documents。只有 `.git`、`.superpowers`、Python/Pixi 缓存和 AppleDouble `._*` 元数据不纳入该有效源码清单。来源快照遵循原仓库 Git 跟踪和 untracked/nonignored 清单；原本被忽略的数据、权重与运行产物不属于源码汇总。

归档是证据，不是可独立执行的工作树；要复原完整版本，需将清单中同一 source 的所有 saved_as 文件按原 path 恢复，而非只拷贝此处的差异目录。训练环境、数据和模型不包含在内。旧源码不自动并入当前模型，以免旧实现覆盖新修复；后续按具体研究需求复用。

原目录在 Ola 上共用 VLM4WAM_baton/.git。HPC3 最终仓库使用独立 .git，不依赖这个旧路径。全分支 Git bundle 存放于 HPC3 仓库外的 VLM4WAM_consolidation_backup_20261009，以保存未发布的旧分支历史；不是 GitHub 需要下载的训练资产。

`SELF_EVOLUTION_DESIGN.md`/`SELF_EVOLUTION_PLAN.md` 中的待审设计保留原文，不代表已实现。早期 BACKGROUND_STATUS 是时间快照，不代表现在仍运行。
