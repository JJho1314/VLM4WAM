# 迭代实验工作规约

每轮开始先读取 `EXPERIENCE.md`、`REPORT.md` 和 registry 中的 baseline、thresholds、best、budget、所有 evidence 及失败日志。把历史现象、当前实测与待验证假设分开。当前优先目标是 RoboFollow 指令遵循；LIBERO 离线动作 MSE 是辅助诊断。

1. 从现有证据提出一个可证伪假设。若 Intent/Execution 都为零，先检查 action/state 归一化、14维投影初始化、相机顺序和历史/未来时序，并做小批次拟合；此时语言差值为零不能证明模型不依赖语言。
2. 每个候选只改变一个清楚的方法因素；保留相同数据版本、初始化、steps、seed、评测协议和 scorer 目标。切换完整数据版本要重建 R0，不能把多出的训练数据算作 planner 收益。
3. 写会失败的接口/回归测试，修正实现，完成真实 GPU 小规模验证后才登记候选。R0–R3 为首阶段注册候选；R4 需要已有 evidence 支持，新增代码及配置必须另做验证，不接受自由 shell 文本。
4. 经控制器产生不可混淆的代码快照、来源哈希和独立 run。保持 8 GPUh 首阶段总预算、2h 每候选、最多2 GPU；不停止其他用户作业，不通过新 registry 隐藏本阶段消耗。
5. 训练完成只是 `trained`。必须完成所有 correct/shuffle 开发 trials，确认 round/seed 配对、官方 scorer、runtime/asset 指纹一致、数值有限，才做比较。teacher future features 仅用于训练。
6. Intent 为主，Execution/CR 非退化；best 初始 baseline 不是已优化模型。正确指令分数提升但打乱分数相近时，记录“语言依赖尚无证据”。只有四任务开发结果时，不宣称完整 benchmark 或 L1–L3 泛化。
7. 保存失败与无收益 run，将假设、实际配置、指标、不确定性、保留/拒绝理由和下一步回写 evidence。经验文档引用 evidence 路径和真实日志；训练 loss 或 smoke 不代替 benchmark。
8. 用未参与选择的任务/等级和更多 seeds 复核最佳候选，再单独安排完整官方验收。完成后记录覆盖范围、资源消耗和未完成事项。

`loop` 是有限的注册候选执行器；研究 agent 负责根据 evidence 修正假设、实现并验证下一项实验。它不声称只靠脚本即可保证模型能力持续提升。
