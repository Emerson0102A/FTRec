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

单组合示例：

```bash
ftrec-pretrain --config configs/experiment/pcgrad.yaml --seed 42 --device cuda
ftrec-adapt --config configs/experiment/lora.yaml --pretrain-method pcgrad --domain 0 --rank 2 --seed 42
```

## 产物

每个训练目录包含 resolved lineage、`batch_manifest.json`、逐 epoch `metrics.jsonl`、`best.pt`、`last.pt`、`result.json` 和 `COMPLETE.json`。Joint/PCGrad 另有 `gradient_conflicts.jsonl`。LoRA checkpoint 只保存 adapter 张量和 backbone SHA-256。

分析目录包含 `results.csv`、`summary.csv`、`recovery.csv`、`warnings.json`，以及四类图的 PNG/PDF：预训练对比、LoRA rank 曲线、Recovery 曲线和梯度冲突热图。Recovery 不裁剪；分母接近零或 FullFT 低于预训练时会保留 NaN/符号并写入警告。
