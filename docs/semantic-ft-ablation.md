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
`data/attribute_experiment/structured-title/` 的完整数据。先打印 5 个预训练和
25 个微调命令，不访问数据：

```bash
ACTION=plan bash scripts/run_semantic_ft_ablation.sh
```

执行全矩阵，完成的组合会由现有 runner 按指纹跳过：

```bash
bash scripts/run_semantic_ft_ablation.sh
```

可以用 `ARMS="shuffled semantic"`、`PHASE=pretrain`、`DOMAINS="0 1"`、
`SEED=43` 限定范围。其他 seed 默认复用同一套固定随机向量/置换，只改变训练随机性。
在完整数据存在后，`ACTION=dry-run` 可检查现有 runner 的实际 run 判定；如果预训练
checkpoint 尚未生成，微调部分会报告 `missing-base`，属于预期。

完成后汇总，并自动检查五组使用相同目标域 cohort 与训练协议：

```bash
python -m ftrec.analysis.semantic_ablation \
  --root runs-attributes/semantic-ft-ablation \
  --output results/semantic-ft-ablation-seed42.json
```

结果中 `macro_gain` 是五域等权微调收益，`aligned_minus_shuffled_gain` 是主要对照。
如果只有 semantic 的收益为正而 shuffled 接近零，再检查跨 seed 一致性、绝对指标
和随机组。若两者均有效，则当前实验不足以把收益完全归因于 item 与语义的对应。
