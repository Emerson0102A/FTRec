# Joint 最优轮次与目标域微调最优轮次

## 口径

在相同 `joint_proportional` 预训练协议下，对 ID-SASRec（`id`）和
structured-title-fused（`semantic`）分别训练至 300 epoch，并保存指定轮次模型。
两个模型都使用 `attribute_structured_title_fused_pretrain.yaml`，仅模型配置不同。
预训练 patience 设为 301，保证晚期快照不会被早停截断。

每个快照分别在五个目标域做 10 epoch FullFT。目标域微调使用 target-only 历史、
至少两项目标域行为、31 个训练负样本，以及固定的 999 个评测负样本；微调 patience
设为 11，确保完成全部 10 epoch。主要微调指标是**第 10 轮结束时的 validation
NDCG@10**；10 轮内由 validation 选出的最佳值仅作为辅助指标。汇总器检查每次
微调都实际完成了 10 轮。

对每种模型分别报告预训练 validation macro NDCG@10 与微调后 validation macro
NDCG@10 在快照轮次上的 Spearman、两个指标各自的最佳轮次，以及逐域结果。
所有快照的比较和相关分析只用 validation；`test_metrics` 不参与选择。
相关系数和最佳轮次只能说明本次观测到的训练轨迹，不能证明全局参数最优点不同。

## 服务器准备

把本分支代码同步到服务器后，在仓库根目录运行：

```bash
cd /root/autodl-tmp/FTRec
conda activate ftrec
python -m pip install -e . --no-deps
test -f data/processed/gmflowrec-amazon/COMPLETE.json
test -d data/attribute_experiment/structured-title
```

先查看正式矩阵，不访问数据、不启动训练：

```bash
ACTION=plan bash scripts/run_checkpoint_transfer.sh
```

默认会生成 2 个预训练 run 和 120 个微调 run。快照轮次为
`5 10 20 30 40 50 75 100 150 200 250 300`。

## 小规模试跑

先用独立输出目录跑 2 个模型、3 个快照、2 个目标域，共 12 次微调：

```bash
RUN_ROOT=runs-attributes/checkpoint-transfer-pilot \
  PRETRAIN_EPOCHS=50 SNAPSHOT_EPOCHS="5 20 50" DOMAINS="0 4" \
  bash scripts/run_checkpoint_transfer.sh \
  2>&1 | tee checkpoint-transfer-pilot.log

PYTHONPATH=src python -m ftrec.analysis.checkpoint_transfer \
  --root runs-attributes/checkpoint-transfer-pilot \
  --epochs 5 20 50 --domains 0 4 --ft-epochs 10 \
  --output results/checkpoint-transfer-pilot.json
```

## 正式运行

正式实验使用新的输出目录，避免用 50-epoch pilot 的配置覆盖 300-epoch 运行：

```bash
PHASE=pretrain bash scripts/run_checkpoint_transfer.sh \
  2>&1 | tee checkpoint-transfer-pretrain.log

PHASE=adapt bash scripts/run_checkpoint_transfer.sh \
  2>&1 | tee checkpoint-transfer-adapt.log

PYTHONPATH=src python -m ftrec.analysis.checkpoint_transfer \
  --root runs-attributes/checkpoint-transfer \
  --output results/checkpoint-transfer-seed42.json
```

预训练快照位于 `.../pretrain/joint_proportional/all-domains/seed-42/snapshots/epoch-XXXX.pt`。
每个微调运行写入独立目录。相同配置的完整运行会被现有 CLI 跳过；服务器中断后可重新
执行对应 `PHASE`，但中断中的单个 run 会从头开始。如需更小矩阵，可设置
`ARMS`、`SEEDS`、`DOMAINS`、
`SNAPSHOT_EPOCHS`、`PRETRAIN_EPOCHS` 与 `FT_EPOCHS`；汇总时传入相同的
`--epochs`、`--domains`、`--seed`、`--ft-epochs`。

`results/checkpoint-transfer-seed42.json` 中的 `spearman_macro`、
`best_joint_epoch` 和 `best_ft_epoch` 是主要结果，`spearman_by_domain` 和
`best_ft_by_domain` 用于检查差异是否集中在少数域。`ft_macro_ndcg10` 是固定预算
结束值，`selected_ft_macro_ndcg10` 是同一预算内选出的最佳验证值。seed 42 只适合作为试探；正式
论断需重复多个预训练 seed，并报告跨 seed 的稳定性。
