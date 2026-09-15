# 实验运行手册

正式实验按阶段执行，任何阶段失败都不会删除此前已完成的原子发布目录：

```text
preprocess → single → joint → pcgrad → lora → fullft → analysis
```

## 先做本地 smoke

```bash
ftrec-smoke --config configs/smoke.yaml --output-dir results/smoke-final
```

它只在 CPU 上使用合成五域数据，每个训练分支只有一个 step，但会真实执行 5 个 Single、Joint、PCGrad、50 个 LoRA、10 个 FullFT 和分析，共 67 个模型运行。smoke 使用 sampled 评测，仅证明管线可运行，不能用于支持或否定科学假设。

## 正式服务器流程

先查看计划，不写 checkpoint：

```bash
bash scripts/run_preprocess.sh --dry-run
bash scripts/run_single.sh --dry-run
bash scripts/run_joint.sh --dry-run
bash scripts/run_pcgrad.sh --dry-run
bash scripts/run_lora.sh --dry-run
bash scripts/run_fullft.sh --dry-run
ftrec-analyze --runs-root runs --output-dir results/analysis --dry-run
```

确认后依次去掉 `--dry-run` 执行：

```bash
bash scripts/run_preprocess.sh
bash scripts/run_single.sh
bash scripts/run_joint.sh
bash scripts/run_pcgrad.sh
bash scripts/run_lora.sh
bash scripts/run_fullft.sh
bash scripts/run_analysis.sh
```

生产配置使用 seeds 42、43、44、full-catalog 评测和按验证集 macro NDCG@10 早停。默认矩阵为：15 个 Single，3 个 Joint，3 个 PCGrad，150 个 LoRA（2 backbones × 5 domains × 5 ranks × 3 seeds）和 30 个 FullFT。LoRA rank 为 1、2、4、8、16，且只训练每层 Q/V adapter。

## 恢复与单组合调试

每个组合独立写入 staging 目录，成功后原子发布并生成 `COMPLETE.json`。再次执行时，配置、数据和 checkpoint 指纹一致的已完成组合会跳过；指纹冲突会停止并要求显式 `--force`。因此服务器中断后直接重新执行同一阶段脚本即可从未完成组合继续。

训练矩阵可以按“已完成组合”恢复，但单次预处理目前不能从 SQLite 中间点续跑。旧版 `run_preprocess.sh` 已运行很久时，不要直接按 `Ctrl+C`，否则 staging 数据会被清理；应先决定是让旧进程完成，还是明确接受重新开始后再切换到优化版。

单组合示例：

```bash
ftrec-pretrain --config configs/experiment/pcgrad.yaml --seed 42 --device cuda
ftrec-adapt --config configs/experiment/lora.yaml --pretrain-method pcgrad --domain 0 --rank 2 --seed 42
```

## 产物

每个训练目录包含 resolved lineage、`batch_manifest.json`、逐 epoch `metrics.jsonl`、`best.pt`、`last.pt`、`result.json` 和 `COMPLETE.json`。`batch_manifest.json` 是小型确定性配方（seed、算法、总 step 和各域样本指纹），实际 batch 在训练时流式生成，不再保存数千万个样本编号。Joint/PCGrad 另有 `gradient_conflicts.jsonl`。LoRA checkpoint 只保存 adapter 张量和 backbone SHA-256。

分析目录包含 `results.csv`、`summary.csv`、`recovery.csv`、`warnings.json`，以及四类图的 PNG/PDF：预训练对比、LoRA rank 曲线、Recovery 曲线和梯度冲突热图。Recovery 不裁剪；分母接近零或 FullFT 低于预训练时会保留 NaN/符号并写入警告。

## 性能与进度日志

预处理会向 stderr 显示三类 `tqdm` 进度条：每个域 gzip 压缩字节的读取进度、joint k-core 的轮次及当轮删除量、Parquet/sequence 的 interaction 导出进度。阶段完成时还会输出 JSON，包括累计行数、吞吐、elapsed 和 ETA。训练每个 epoch 报告 elapsed、ETA 和验证指标。保留日志的推荐方式：

```bash
bash scripts/run_preprocess.sh 2>&1 | tee preprocess.log
bash scripts/run_joint.sh 2>&1 | tee joint.log
```

非交互任务如果不希望输出进度条，可使用：

```bash
bash scripts/run_preprocess.sh --no-progress
```

正式配置默认 `evaluation_batch_size: 128`、`evaluation_chunk_size: 4096`。显存不足时先把 evaluation batch 调到 64 或 32；评测仍慢但显存充足时再增到 256。单张 4090D 默认顺序跑实验组合：SASRec 较小不代表多进程一定更快，Joint/PCGrad 和 full-catalog 评测通常已经能占满 GPU。只有通过 `nvidia-smi dmon` 确认 GPU 长期空闲、且单进程显存明显不足总显存的一半时，才值得手工测试两个组合并发。
