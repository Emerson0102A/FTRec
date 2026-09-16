# 实验运行手册

正式实验按阶段执行，任何阶段失败都不会删除此前已完成的原子发布目录：

```text
import GMFlowRec → single → joint/pcgrad → lora → fullft → analysis
```

## 先做本地 smoke

```bash
ftrec-smoke --config configs/smoke.yaml --output-dir results/smoke-final
```

它只在 CPU 上使用合成五域数据，每个训练分支只有一个 step，但会真实执行 5 个 Single、Joint、PCGrad、50 个 LoRA、10 个 FullFT 和分析，共 67 个模型运行。smoke 使用 sampled 评测，仅证明管线可运行，不能用于支持或否定科学假设。

## 正式服务器流程

作者发布的三个 Parquet 文件放在 `data/MDSR-Amazon/` 后，先转换为 FTRec
原生格式。源目录已被 `.gitignore` 排除，不需要上传 GitHub：

```bash
bash scripts/run_import_gmflowrec.sh --dry-run
bash scripts/run_import_gmflowrec.sh
```

导入器会核对论文中的 580,329 用户、14,308,937 条交互及五域统计，把作者从
0 开始的 item id 整体加 1（为 SASRec padding id 0 留位），并一次性生成验证集和
测试集各自固定的 1 正样本 + 999 个同域未交互负样本。后续每个 epoch 复用同一份
候选矩阵，不会重新采样。

如果目标是先快速判断核心假设是否值得继续，先运行单 seed pilot。默认只运行
seed 42 的 Joint、PCGrad，以及两个 backbone 在五个 domain 上的 LoRA rank
1/2/4，共 32 个模型：

```bash
bash scripts/run_pilot.sh --dry-run
bash scripts/run_pilot.sh
```

单张 4090D 可先尝试两个预训练并发，以及两个 LoRA 作业并发：

```bash
bash scripts/run_pilot.sh --pretrain-workers 2 --adapt-workers 2 2>&1 | tee pilot.log
```

runner 会先并行完成 Joint/PCGrad，再把每个「backbone × domain × rank」拆成独立
LoRA 作业并发执行。不同阶段仍保持依赖顺序。若显存不足或总速度反而下降，把对应
worker 调回 1；tmux 不是必需的，但仍建议把整条命令放在一个 tmux 会话中防止 SSH
断线。

初步 LoRA 结果有信号后，补同一 seed 的 10 个 FullFT，已完成的 32 个组合会
按指纹自动跳过：

```bash
bash scripts/run_pilot.sh --with-fullft
ftrec-analyze --runs-root runs --output-dir results/pilot-seed-42
```

pilot 只用于筛选研究方向，不能替代多 seed 正式结论。需要复核随机稳定性时，
再执行下面的完整矩阵。

## 从 pilot 切换到正式 seed 42

正式训练前先保留整个 pilot 目录，再让新的 100-epoch checkpoint 使用默认的
`runs/` 路径。这样后续 LoRA/FullFT 会自动读取正式 checkpoint，而不会把 pilot
与正式结果混在一起：

```bash
PILOT_ARCHIVE="runs-pilot-seed-42-$(date +%Y%m%d-%H%M%S)" &&
test ! -e "$PILOT_ARCHIVE" &&
mv -- runs "$PILOT_ARCHIVE" &&
mkdir -p runs logs &&
printf 'pilot archived at %s\n' "$PILOT_ARCHIVE"
```

Joint 和 PCGrad 的正式配置均为最多 100 epoch、`steps_per_epoch: auto`、按验证集
macro NDCG@10 早停（patience 10）。不要通过 `run_pilot.sh` 启动正式训练，因为
pilot runner 会显式覆盖 epoch 和 step 数。可在两个 tmux 会话中分别执行：

```bash
ftrec-pretrain --config configs/experiment/joint.yaml --seed 42 \
  2>&1 | tee logs/joint-seed-42.log

ftrec-pretrain --config configs/experiment/pcgrad.yaml --seed 42 \
  2>&1 | tee logs/pcgrad-seed-42.log
```

每轮训练的 `metrics.jsonl` 同时保留数字域字段，并额外提供 `domain_names`、
`domain_losses_by_name` 和 `validation_by_name`。最终 `result.json` 额外提供
`test_metrics_by_name` 与 `validation_metrics_by_name`，方便直接按 Health、Clothing、
Beauty、Grocery、Sports 阅读；数字域字段继续保留，以兼容已有分析代码。

先查看计划，不写 checkpoint：

```bash
bash scripts/run_import_gmflowrec.sh --dry-run
bash scripts/run_single.sh --dry-run
bash scripts/run_joint.sh --dry-run
bash scripts/run_pcgrad.sh --dry-run
bash scripts/run_lora.sh --dry-run
bash scripts/run_fullft.sh --dry-run
ftrec-analyze --runs-root runs --output-dir results/analysis --dry-run
```

确认后依次去掉 `--dry-run` 执行：

```bash
bash scripts/run_import_gmflowrec.sh
bash scripts/run_single.sh
bash scripts/run_joint.sh
bash scripts/run_pcgrad.sh
bash scripts/run_lora.sh
bash scripts/run_fullft.sh
bash scripts/run_analysis.sh
```

生产配置使用 seeds 42、43、44、45、46，按论文协议对固定 1000 候选计算
HR/NDCG@5,@10，并按验证集 macro NDCG@10 早停。默认矩阵为：25 个 Single、
5 个 Joint、5 个 PCGrad、250 个 LoRA（2 backbones × 5 domains × 5 ranks × 5
seeds）和 50 个 FullFT。LoRA rank 为 1、2、4、8、16，且只训练每层 Q/V adapter。

## 恢复与单组合调试

每个组合独立写入 staging 目录，成功后原子发布并生成 `COMPLETE.json`。再次执行时，配置、数据和 checkpoint 指纹一致的已完成组合会跳过；指纹冲突会停止并要求显式 `--force`。因此服务器中断后直接重新执行同一阶段脚本即可从未完成组合继续。

训练矩阵可以按“已完成组合”恢复。GMFlowRec 导入不再执行旧版原始评论数据的
joint k-core；作者发布的数据已经过论文预处理，因此无需继续运行旧
`run_preprocess.sh`。

单组合示例：

```bash
ftrec-pretrain --config configs/experiment/pcgrad.yaml --seed 42 --device cuda
ftrec-adapt --config configs/experiment/lora.yaml --pretrain-method pcgrad --domain 0 --rank 2 --seed 42
```

## 产物

每个训练目录包含 resolved lineage、`batch_manifest.json`、逐 epoch `metrics.jsonl`、`best.pt`、`last.pt`、`result.json` 和 `COMPLETE.json`。`batch_manifest.json` 是小型确定性配方（seed、算法、总 step 和各域样本指纹），实际 batch 在训练时流式生成，不再保存数千万个样本编号。Joint/PCGrad 还会从 PCGrad 投影前的 raw domain gradients 生成：

- `gradient_conflicts.jsonl`：每次采样点的完整矩阵；
- `gradient_conflict_pairs.csv`：可直接分析的 domain pair 长表；
- `gradient_conflict_summary.json`：最终 EMA group/domain/layer 汇总；
- `gradient_conflict_by_domain_layer.csv`：`domain,layer,q_conflict,v_conflict,qv_conflict`，可与 LoRA gain 直接 join。

这些统计包含动态生成的 `block_l_q`、`block_l_v`、`block_l_qv`，同时保留原有
attention、FFN 和 full groups。LoRA checkpoint 只保存 adapter 张量和 backbone
SHA-256。

LoRA 与 FullFT 会先把尚未更新的模型作为 `epoch=0` 做一次验证，并把它纳入
early stopping 与最佳 checkpoint 选择。若微调始终未超过初始模型，`best.pt` 会保留
epoch 0，避免强制采用负迁移的 checkpoint。`metrics.jsonl` 的首行以
`phase: "initial_validation"` 标识该记录；`result.json` 额外保存
`initial_validation_metrics`、`best_validation_metrics` 和
`best_validation_ndcg`。原有 `validation_metrics` 仍表示最后一次训练 epoch 的验证值，
以保持旧分析脚本兼容。epoch 0 未执行 optimizer step，因此该行的 `loss` 与
`gradient_norm` 均记为 0。

分析目录包含 `results.csv`、`summary.csv`、`recovery.csv`、`warnings.json`，以及四类图的 PNG/PDF：预训练对比、LoRA rank 曲线、Recovery 曲线和梯度冲突热图。Recovery 不裁剪；分母接近零或 FullFT 低于预训练时会保留 NaN/符号并写入警告。

## 性能与进度日志

预处理会向 stderr 显示三类 `tqdm` 进度条：每个域 gzip 压缩字节的读取进度、joint k-core 的轮次及当轮删除量、Parquet/sequence 的 interaction 导出进度。训练阶段显示每个模型 run 的 optimizer step 进度，以及每次 full/sampled evaluation 的用户进度；LoRA 和 FullFT 还会显示整个组合矩阵的已完成 run 数。阶段完成时继续输出 JSON，包括 elapsed、ETA、loss 和验证指标。保留日志的推荐方式：

```bash
bash scripts/run_preprocess.sh 2>&1 | tee preprocess.log
bash scripts/run_joint.sh 2>&1 | tee joint.log
```

所有训练命令同样支持 `--no-progress`。非交互任务如果不希望输出进度条，可使用：

```bash
bash scripts/run_preprocess.sh --no-progress
```

正式配置默认 `evaluation_batch_size: 256`、`evaluation_chunk_size: 4096`。显存不足时先把 evaluation batch 调到 128、64 或 32。并行作业不是无条件加速：两个进程会争用同一张 GPU 的算力、显存和 PCIe；先用 worker=2 观察吞吐与显存，再决定 LoRA 是否升到 3 或 4。
