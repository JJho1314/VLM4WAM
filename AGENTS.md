# 项目约定

- 简体中文沟通。当前研究入口为 README.md、docs/HPC3_ROBOFOLLOW_SUMMARY.md 和 docs/HPC3_RUNBOOK.md。
- 后续训练只用 HPC3 jhe724，最多 2 nodes / 16 GPUs；不使用 Qianhai。优先 acd_u，不自动升级高价分区。
- 登录域名 hpc3login.hpc.hkust-gz.edu.cn；不要固定旧 IP。不要绕过 SSH 主机密钥核验。
- 权威工作目录 /data/user/jhe724/workspace/VLM4WAM_robofollow_official3750。
- 所有远程 Python 经 /data/user/jhe724/.pixi/bin/pixi 和 autoresearch/runtime/pixi.toml 运行。
- 不修改排队/运行作业使用的文件；先检查 squeue、jobid 和 .agent/official3750/code.sha256。新实验使用隔离目录及独立输出。
- 数据、权重、输出、凭据、环境目录不入 Git。旧目录不因汇总完成而自动删除。
- docs/archive/ 为历史证据；历史路径和预算不作为当前配置。保留失败结果，严格区分 Intent、Execution、Completion、smoke 和完整 benchmark。
