# VLM4WAM 工作约定

## 训练和评测优先把性能拉满

任何训练或评测任务在正式长跑之前，先确认 GPU 是满载的、数据读取不是瓶颈；吞吐没优化好就不要开长任务。

- **先测再跑**：正式提交前先跑几十步，记录 s/it、GPU 利用率（`nvidia-smi`）、显存峰值和 CPU 占用。GPU 利用率长期低于约 90%，或步速波动大，先查原因再提交。
- **显存用足**：在不 OOM 的前提下把每卡 batch 调大，用梯度累积保持全局 batch 不变。Stage 2 从每卡 1 提到每卡 4、并把 SigLIP2 teacher 预处理放到 GPU（`BATON_TEACHER_GPU_PREPROCESS=1`）后，2 卡上快了 3.5 倍（13.7 s/step 对 48.0 s/step）。
- **数据读取不能拖后腿**：DataLoader 的 worker 数和 CPU 配额要和卡数匹配；集群规格按每卡分到的 CPU 核数选，不要只看 GPU 数。worker 内存持续上涨时在 collate 里调用 `malloc_trim(0)`。
- **共享文件系统**：在 AFS / quarkfs 这类网络文件系统上读 HDF5，必须设 `HDF5_USE_FILE_LOCKING=FALSE`，否则会偶发文件锁错误（errno 9），导致 DataLoader worker 崩溃、整个多卡任务卡在 NCCL 超时。
- **别和别人抢资源**：同一容器或节点上有其他高 CPU 任务时，数据加载会被拖慢（实测 13 s/step 掉到 71 s/step）。长训要放在独占的节点或任务上。
- **训练中持续监控**：长训期间定期看步速和 GPU 利用率，速度明显下降就排查，不要等跑完才发现。
