# 商品语义与目标域微调收益消融

## 问题与对照

比较五域 `joint_proportional` 预训练后，目标域 LoRA 的 NDCG@10 增益。所有组使用
64 维、2 层、2 个 attention head 的单塔 SASRec；预训练沿用
`attribute_structured_title_fused_pretrain.yaml`，微调沿用
`attribute_structured_title_fused_target_only_lora.yaml`（目标域历史、31 个训练负样本、
999 个固定评测负样本、rank 5 all-linear LoRA、epoch 0 参与 validation 选模）。

| 组 | 商品输入 | 解释 |
| --- | --- | --- |
| `id` | 可学习 ID embedding | 传统基线；预训练参数量不同，不能单独用于因果归因 |
| `random` | 固定随机方向、逐向量保留原始范数和空属性槽 | 检查固定稠密向量能否产生收益 |
| `shuffled` | 同域、同 present 状态内对 title+attributes 成对置换 | 主负对照；精确保留每域联合分布、维度、模型参数和训练预算 |
| `attribute` | 原始 attribute bank；title 分支输出被屏蔽 | 属性单独提供的信息；title 参数保留在结构中但不接收梯度 |
| `semantic` | 原始 structured-title-fused bank | 完整语义组 |

`shuffled` 对每个域、每种 present 状态分别随机成环置换，固定 seed 42；至少两个商品的组没有固定点。
title 与属性一起移动，padding 行保持零。随机组使用本地 RNG，不改变模型参数初始化的
全局 RNG。`content_ablation`、seed 和域索引文件写入模型配置与 run 指纹；微调读取
checkpoint 时还会核对预训练的模型配置，避免错配外置、冻结的内容 bank。

关键检验是相同 domain/seed 的配对差值：

```text
gain(arm, domain) = adapted_test_NDCG@10 - epoch0_test_NDCG@10
alignment_effect(domain) = gain(semantic, domain) - gain(shuffled, domain)
```

先报告五域逐域值及五域等权均值。`semantic` 与 `shuffled` 共享同一内容编码器和
SASRec 结构，差异是商品与语义的正确对应。比较 gain 时也要同时看两个组的
epoch 0 绝对 NDCG、适配后绝对 NDCG 和 `best_epoch`；`gain` 的差异本身仍不能证明
唯一机制是语义结构，例如预训练起点与优化轨迹也可能相互作用。seed 42 是 pilot；
正式结论应补多个预训练/适配 seed，并给出跨 seed 不确定性。

## 运行

需要 `data/processed/gmflowrec-amazon/` 和
`data/attribute_experiment/structured-title/` 的完整数据。seed 42 默认复用已有的
`runs-lr1e-4` ID 预训练 checkpoint、`runs-attributes/structured-title-fused`
语义预训练 checkpoint，以及已完成的 structured-title-fused target-only rank-5
`lora_all` 微调结果。脚本不会复制或覆盖这些产物。先打印计划，不访问数据：

```bash
ACTION=plan bash scripts/run_semantic_ft_ablation.sh
```

若上述已有产物都在服务器上，计划只新增 random、shuffled、attribute 的 3 个
预训练，以及这三组和 ID 的 20 个目标域微调。若语义微调缺少完成标记，脚本会在
新目录补跑该域。执行矩阵时，完成的新增组合由现有 runner 按指纹跳过：

```bash
bash scripts/run_semantic_ft_ablation.sh
```

可以用 `ARMS="shuffled semantic"`、`PHASE=pretrain`、`DOMAINS="0 1"`、
`SEED=43` 限定范围。其他 seed 默认复用同一套固定随机向量/置换，只改变训练随机性。
其他 seed 只有在对应的 ID 和 semantic 预训练 checkpoint 已存在时才能复用；
本脚本不会悄悄用新训练覆盖它们。默认已有结果路径可用 `ID_BASE_ROOT`、
`SEMANTIC_BASE_ROOT` 和 `SEMANTIC_ADAPT_ROOT` 覆盖。
在完整数据存在后，`ACTION=dry-run PHASE=pretrain` 可检查新组预训练的实际 run
判定。新组的 checkpoint 生成后，再用 `ACTION=dry-run PHASE=adapt` 检查微调；
缺少基线 checkpoint 时脚本会直接报错。

完成后汇总，并自动检查五组使用相同目标域 cohort 与训练协议：

```bash
python -m ftrec.analysis.semantic_ablation \
  --root runs-attributes/semantic-ft-ablation \
  --output results/semantic-ft-ablation-seed42.json
```

汇总器会在新目录没有 semantic 结果时读取原有的 target-only `lora_all` 结果；
若其根目录不同，用 `--semantic-root` 指定。已有 ID 预训练的最多 epoch / patience
是 100 / 10，best epoch 为 2；语义预训练的最多 epoch / patience 是 300 / 20，
best epoch 为 293。因此 ID 是辅助基线，不能把 ID 与语义组的差异归因于语义
结构。主对照的 shuffled 按语义组相同的预训练配置和预算训练；解释两者差异时
需披露语义组复用了历史 checkpoint。

结果中 `macro_gain` 是五域等权微调收益，`aligned_minus_shuffled_gain` 是主要对照。
如果只有 semantic 的收益为正而 shuffled 接近零，再检查跨 seed 一致性、绝对指标
和随机组。若两者均有效，则当前实验不足以把收益完全归因于 item 与语义的对应。
