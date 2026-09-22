# Structured fused 的目标域微调

## 研究问题

固定已经完成的五域 `joint_proportional` Structured fused checkpoint，依次把
Health、Clothing、Beauty、Grocery、Sports 当作目标域。微调、验证和测试只保留
目标域历史，回答：在相同 domain-only cohort 和固定候选集上，目标域参数更新是否
优于 epoch 0 的未微调模型。

本实验不重新训练属性提取 LLM，也不使用 LLM2Attr、dual tower 或 ID embedding。
Structured-LLM 导出的 title/attribute embedding bank 始终冻结。

## 为什么采用内容 adapter + SASRec LoRA

Structured fused 先用可训练投影和属性选择器把冻结的 LLM embedding 映射到 64 维
商品空间，再交给单塔 SASRec。旧的 `lora_all` 只修改 SASRec，无法直接纠正目标域
商品语义空间。

实验采用零初始化的 residual content adapter：

```text
frozen Structured fused item vector
    -> x + Up(GELU(Down(x)))
    -> SASRec with optional all-linear LoRA
```

`Down: 64 -> 16`，`Up: 16 -> 64`，且 `Up` 初始化为零，所以 epoch 0 与原始
checkpoint 完全等价。adapter 同时作用于历史商品和候选商品，不会让序列输入空间与
打分空间失配。

这一选择与以下证据一致：

- UniSRec 使用轻量语义 adaptor 将文本表示迁移到下游域；其官方微调代码在
  `fix_enc=True` 时冻结 position embedding 和 Transformer，但继续训练文本
  adaptor：<https://arxiv.org/abs/2206.05941>、
  <https://github.com/RUCAIBox/UniSRec/blob/master/finetune.py>。
- PeterRec 表明冻结预训练主干、学习小型残差 patch 可以接近 FullFT，并在数据较少时
  更稳定：<https://arxiv.org/abs/2001.04253>。
- LoRA 用零初始低秩增量适配 Transformer，避免为五个域复制完整主干：
  <https://arxiv.org/abs/2106.09685>。
- 内容序列推荐的近期研究发现，完全冻结内容表示可能欠适配，而自由全量更新又可能
  破坏冷门商品的语义结构；小的可训练 delta 是更稳妥的折中：
  <https://arxiv.org/abs/2507.19473>。

## 受控实验矩阵

三个核心实验臂只改变可训练参数：

| 实验臂 | 可训练参数 | 作用 |
| --- | --- | --- |
| `content_adapter` | 64→16→64 内容残差 | 只适配目标域商品语义空间 |
| `lora_all` | SASRec Q/K/V/O 与 FFN 的 rank-5 LoRA | 只适配目标域序列模式 |
| `lora_all_content_adapter` | 上述两者 | 主方案，联合适配语义与序列 |

可选 `fullft` 是容量上界，不是主方案。它使用较低的 `1e-4` 学习率，并仍由 epoch 0
参与 checkpoint 选择。

所有实验臂共享：

- `context_mode: target_only`；
- `min_domain_sequence_length: 2`，即至少一个目标域历史和一个 target；
- 每个正样本配 31 个同域、互不重复且未交互的训练负样本；
- 固定的 999 个同域评测负样本；
- validation NDCG@10 选择 checkpoint，test 不参与选择；
- epoch 0 作为合法 best checkpoint，微调无收益时保留原模型。

## 运行

先确认五域 checkpoint、数据和输出路径：

```bash
ACTION=dry-run bash scripts/run_structured_fused_target_only_adapt.sh
```

运行 seed 42 的三个核心实验臂，共 15 个 run：

```bash
bash scripts/run_structured_fused_target_only_adapt.sh \
  2>&1 | tee structured-fused-target-only-seed42.log
```

只有核心实验出现正向信号后再运行 FullFT 上界：

```bash
INCLUDE_FULLFT=1 bash scripts/run_structured_fused_target_only_adapt.sh
```

脚本使用以下配置：

- `configs/experiment/attribute_structured_title_fused_target_only_content.yaml`；
- `configs/experiment/attribute_structured_title_fused_target_only_lora.yaml`；
- `configs/experiment/attribute_structured_title_fused_target_only_joint.yaml`；
- `configs/experiment/attribute_structured_title_fused_target_only_fullft.yaml`。

## 判定

每个 `result.json` 都在同一批 domain-only test 用户和候选集上保存：

```text
pretrain_metrics.NDCG@10  # 参数更新前
test_metrics.NDCG@10      # validation 选出的 checkpoint
best_epoch                # 0 表示不微调最好
```

每域的微调收益为：

```text
gain_d = test_metrics.NDCG@10 - pretrain_metrics.NDCG@10
```

先报告五域逐域 gain 和五域等权 Macro gain。seed 42 只用于筛选方案；如果联合方案
Macro gain 为正且至少四个域同方向，再把三份核心配置的 `seeds` 扩为
`[42, 43, 44, 45, 46]`。最终结论使用五个 seed 的 mean±std，并检查：

1. 联合方案是否优于 epoch 0；
2. 联合方案是否同时优于 content-only 和 LoRA-only；
3. 最佳 epoch 是否经常为 0；
4. FullFT 是否提供更高上界，还是出现过拟合。

若 LoRA-only 无效而 content-only/联合方案有效，说明过去 ID+SASRec 的“微调无效”
不能推广到内容模型，关键适配位置在商品语义空间。若三个核心实验臂都无法超过
epoch 0，才支持 Structured fused 在当前数据与目标下不需要域内微调。
