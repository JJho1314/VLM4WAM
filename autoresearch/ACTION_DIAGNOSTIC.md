# 动作与语言能力的后续诊断

这份清单是下一轮假设，不是已经获得提升的结果。先读REPORT.md与实际evidence；首阶段8GPUh停止条件继续有效。

已核对的契约：原始joint_action只shift一次，state[t]对应action[t+1]；head/left_wrist/right_wrist和left6+gripper+right6+gripper固定。loader返回4历史+54未来动作，generic trainer实际使用最后54个，pipeline采样54个，官方每次执行前50个；当前state输入[B,1,14]。这些接口检查和normalize/denormalize回归已通过，不把未经证实的接口猜测当失败原因。

14D action projection的4个weight/bias来自显式新初始化，其他source weights严格迁移。300步、batch1的文本基线最后loss约1.77，仅是有限预算训练，不能表示动作头收敛。当前已完成的scene1官方诊断出现wrong_arm_intervened、首次关闭抓取源对象不正确及后续阶段门控。这些是真实评测日志观察，不能单独证明归一化错误或语言模块失效。

建议顺序：

1. 在train split内固定少量跨场景episodes，核对实际joint targets和normalized targets的round-trip、数值范围、静止关节/夹爪、训练与采样动作切片；检查预测初始几行与当前joint state差异。使用train-only统计，不修改冻结dev协议。
2. 固定同一初始化、噪声与样本，先测试新14D action projection能否在小样本上拟合。把“仅warm-up动作头”和“同时更新backbone”设为独立因素，记录动作loss与实际可执行轨迹；需要足够steps，不能以300步上限内的小loss差代替收敛。此离线拟合仅作诊断，不作benchmark分数。
3. 建立正的动作Execution后，比较正确/打乱/空指令和同场景相反指令；保持scorer原目标、相同seed、噪声及动作预算。检查conditioning梯度以及matched instruction改变是否导致对应动作改变。
4. 对predicted/teacher gap先离线比较相同样本的feature误差、conditioning分布，再比较mix概率或query/MLP joint+anchor。teacher未来RGB只在训练使用；部署始终当前公开观测和指令。
5. 开发集收益后再使用未参与选择的L1–L3任务和更多种子，最后独立完成官方5540trials验收。不能用规则解析任务名、privileged目标或降低strict scorer来替代模型指令能力。

任何新数据版本（包括补回4条视觉不同的示范）必须重建manifest/stats与R0，旧v3阈值不沿用。新实验需先登记单因素假设、验证代码和GPU预算，不能通过新root重置本阶段账本。

R1 first scene diagnostic: `20261003_131008_R1/paired_eval/correct/scene1/results.json` round1 source select0, closest yellow_block_2 while target is yellow_block_1, source_xy23.9cm, wrong arm participated; subsequent relative-place/finish stages gated. Both scene1 rounds and language modes have Intent/Execution0. R1 training reached300 steps, final batch loss1.97, phasewall542.738seconds including model loading, peak allocated37.805GB. These are actual observations, not proof of a normalization bug or planner effectiveness.
