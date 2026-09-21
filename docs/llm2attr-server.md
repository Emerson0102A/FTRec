# LLM2Attr 服务器运行

默认项目目录是 `/root/autodl-tmp/FTRec`，完整 LLM2Attr 目录是
`/root/autodl-tmp/FTRec/LLM2Attr`。脚本会自动定位：

- MNTP：`output/mntp/Qwen2.5-0.5B/checkpoint-10000`
- 属性模型：`output/attr/Qwen2.5-0.5B/checkpoint-3000`

属性模型目录必须保留 `title_prompt_encoder` 与 `attr_prompt_encoder`。

## 安装与预检

```bash
cd /root/autodl-tmp/FTRec
pip install -e .
pip install -r requirements-llm2attr.txt
bash scripts/check_llm2attr_environment.sh
```

依赖组合固定为 `transformers==4.44.2`、`llm2vec==0.2.3` 和
`peft==0.18.1`。`llm2vec 0.2.3` 不兼容 Transformers 4.46 以上版本，
不要单独升级 Transformers。安装完成后运行 `python -m pip check`，再执行预检。

预检会检查 CUDA、BF16、依赖、checkpoint 完整性，并通过
`HF_ENDPOINT=https://hf-mirror.com` 验证 `Qwen/Qwen2.5-0.5B`。

## 先处理 64 个商品

```bash
bash scripts/run_llm2attr_export.sh
```

默认只处理 64 个商品，输出到 `data/attribute_experiment/llm2attr-smoke`。
配置为 BF16、PyTorch SDPA、batch size 16，不要求安装 FlashAttention。

若预检确认实际可见显存约 80GB，可测试 batch size 32：

```bash
BATCH_SIZE=32 LLM2ATTR_OUTPUT=data/attribute_experiment/llm2attr-smoke-b32 \
  bash scripts/run_llm2attr_export.sh
```

## 全量导出

```bash
MODE=full BATCH_SIZE=32 bash scripts/run_llm2attr_export.sh
```

全量输出为 `data/attribute_experiment/llm2attr`。如果显存不足，把 batch size
降回 16。输出目录已存在时脚本会拒绝覆盖，除非在底层命令中显式使用
`--force`。

主实验固定为 3 个属性，因为现有 LLM2Attr checkpoint 就是按 3 个属性训练的。
代码支持其他数量，但必须显式传入 `--allow-ood-attribute-count`，并作为 OOD
消融单独报告，不能与主实验混在一起。

可覆盖的路径变量包括 `FTREC_ROOT`、`LLM2ATTR_ROOT`、`MDSR_SOURCE_DIR`、
`AMAZON_META_DIR`、`ATTRIBUTE_CATALOG` 与 `LLM2ATTR_OUTPUT`。

## Structured-LLM 对照组

先运行只读取标题的公平对照；它与 LLM2Attr 使用完全相同的 catalog、item ID、
属性数量和 artifact 格式：

```bash
bash scripts/run_structured_attribute_export.sh
MODE=full BATCH_SIZE=16 bash scripts/run_structured_attribute_export.sh
```

增强组可以额外读取分类、features、description、details 和 store。它接收了更多
信息，必须单独报告，不能替代 title-only 主对照：

```bash
MODE=full EVIDENCE=enhanced BATCH_SIZE=16 \
  bash scripts/run_structured_attribute_export.sh
```

## 2×2 训练消融

两个主实验都沿用 `data/processed/gmflowrec-amazon` 中已有的 MDSR
train/valid/test 划分，不重新划分，也都不使用 ID embedding。每种属性提取器
运行两种推荐架构：

- `dual`：严格沿用 MyRec。Title 和 Attr 各自经过一套完整且参数独立的
  SASRec；训练损失是两塔 BCE 之和，推理分数是
  `0.5 * title_score + 0.5 * attr_score`。
- `fused`：Title 与 Attr 先融合成一个纯内容 item embedding，只经过一套
  SASRec。

先运行四个预训练：

```bash
ftrec-pretrain \
  --config configs/experiment/attribute_llm2attr_pretrain.yaml \
  --model-config configs/model/sasrec_llm2attr.yaml

ftrec-pretrain \
  --config configs/experiment/attribute_structured_title_pretrain.yaml \
  --model-config configs/model/sasrec_structured_title.yaml

ftrec-pretrain \
  --config configs/experiment/attribute_llm2attr_fused_pretrain.yaml \
  --model-config configs/model/sasrec_llm2attr_fused.yaml

ftrec-pretrain \
  --config configs/experiment/attribute_structured_title_fused_pretrain.yaml \
  --model-config configs/model/sasrec_structured_title_fused.yaml
```

四组内容模型默认最多训练 300 epoch，并使用 patience 20。如果已有旧版
100-epoch 完整运行，不要使用 `--force` 从头覆盖；用 `--resume` 从同一目录的
`last.pt` 原地接续。它会恢复模型、优化器、随机数状态、全局步数、历史最优值和
前 100 轮 `metrics.jsonl`，其中 300 表示总 epoch 上限：

```bash
ftrec-pretrain --resume \
  --config configs/experiment/attribute_llm2attr_pretrain.yaml \
  --model-config configs/model/sasrec_llm2attr.yaml

ftrec-pretrain --resume \
  --config configs/experiment/attribute_structured_title_pretrain.yaml \
  --model-config configs/model/sasrec_structured_title.yaml

ftrec-pretrain --resume \
  --config configs/experiment/attribute_llm2attr_fused_pretrain.yaml \
  --model-config configs/model/sasrec_llm2attr_fused.yaml

ftrec-pretrain --resume \
  --config configs/experiment/attribute_structured_title_fused_pretrain.yaml \
  --model-config configs/model/sasrec_structured_title_fused.yaml
```

再分别运行五个目标域的 LoRA 微调：

```bash
ftrec-adapt \
  --config configs/experiment/attribute_llm2attr_adapt.yaml \
  --model-config configs/model/sasrec_llm2attr.yaml

ftrec-adapt \
  --config configs/experiment/attribute_structured_title_adapt.yaml \
  --model-config configs/model/sasrec_structured_title.yaml

ftrec-adapt \
  --config configs/experiment/attribute_llm2attr_fused_adapt.yaml \
  --model-config configs/model/sasrec_llm2attr_fused.yaml

ftrec-adapt \
  --config configs/experiment/attribute_structured_title_fused_adapt.yaml \
  --model-config configs/model/sasrec_structured_title_fused.yaml
```

适配阶段使用 `lora_all`：ContentEncoder（包括 Title/Attr adapter 与属性选择器）
全部冻结，只训练 SASRec 各线性层中的 LoRA 参数。双塔有两套 SASRec，所以同一
rank 下 LoRA 参数量约为单塔的两倍；结果表必须同时报告总参数量和可训练参数量，
不能把这项架构差异隐藏起来。建议先完成 seed 42 的可行性实验，再扩为相同的 5 个
seeds，并额外报告按训练频次分桶的 head/medium/tail 指标。

## 主干训练后的内容诊断

四组主干结束后，在微调前统一运行：

```bash
bash scripts/run_content_diagnostics.sh
```

每个 `best.pt` 同目录会生成 `content-diagnostics.json`。双塔报告
`fusion`、`title`、`attribute` 三套结果，单塔报告 `fused`。所有结果复用训练时
保存的同一组 999 个同域负样本，并按目标物品在训练 cohort 中的出现次数报告
`0`、`1-4`、`5-14`、`15-49`、`50-99`、`100+` 六个桶。频次统计明确排除
valid/test cohort，因此不会用测试数据定义流行度。

双塔还会先在 validation 上扫描属性权重，并用坐标搜索选择每个候选物品频次桶的
权重，然后冻结这些权重，只运行一次 test。权重由每个候选物品自己的训练频次决定，
不会根据测试正样本所属桶给整行候选加权，因此不泄漏目标身份。脚本也会对 ID-SASRec
运行相同的频次诊断。

## 三属性聚合消融（先于序列消融）

原始 `sasrec_llm2attr_fused.yaml` 保留 MyRec 的 `hard_top1`：训练时用 hard
Gumbel、评估时用 argmax，从三个属性中只选一个。它现在只作为旧基线，不再假设
硬选择一定最优。新增三种只改变属性聚合、仍然不使用 ID embedding 的单塔模型：

- `mean_all`：对所有非零属性等权平均；
- `soft_attention`：由 title 生成 query，对所有非零属性做可微 softmax；
- `domain_title_attention`：借鉴 MyModel4，由 item 所属 domain 的可学习 query 与
  domain-specific title query 共同产生 softmax 权重。

三种软聚合都先把每个属性独立投影到 SASRec hidden size，再融合。注意力参数从零
初始化，因此初始权重等同于 `mean_all`，训练后才学习偏好；缺失属性会被 mask。
领域来自 `items.csv.gz` 的 item-domain 映射，只描述候选物品自身，不读取 target、
valid/test 频次或用户标签。为了隔离“聚合机制”的贡献，这里没有移植 MyModel4 的
correction branch，也没有额外加入 domain embedding。

先训练三个新模型（seed 42、joint proportional、最多 300 epoch、patience 20）：

```bash
bash scripts/run_attribute_pooling_ablation.sh
```

原 hard-top1 结果复用 `runs-attributes/llm2attr-fused`，不重复训练。三个新结果位于
`runs-attribute-pooling/{mean-all,soft-attention,domain-title-attention}`。完成后运行：

```bash
ACTION=diagnose bash scripts/run_attribute_pooling_ablation.sh
```

选择 validation macro NDCG@10 最好的聚合模型，再与 ID 和 structured-title-fused
一起进入下面的 mixed/domain-only 序列消融；不要先根据 test 挑模型。

## Mixed 与 domain-only 序列消融

不能直接用 `joint_proportional` 对比 domain-only，因为前者包含没有目标域历史的额外
用户，目标 cohort 和上下文长度同时发生了变化。严格的序列消融使用：

- `joint_mixed_matched`：要求存在目标域历史，保留完整五域混合上下文；
- `joint_domain`：使用完全相同的用户和目标，只保留目标域上下文。

核心组包含 ID、LLM2Attr dual 和当前最佳的 structured-title fused，共六个 run：

```bash
bash scripts/run_sequence_context_ablation.sh
```

若需要覆盖全部四种内容组合：

```bash
MODEL_SET=all bash scripts/run_sequence_context_ablation.sh
```

训练完成后运行同口径诊断；双塔会同时执行 validation 自适应融合：

```bash
ACTION=diagnose bash scripts/run_sequence_context_ablation.sh
```

两组训练固定使用相同 seed、matched 目标 cohort、999 个同域候选、最多 300 epoch、
patience 20 和学习率 `1e-4`。报告时核心量是
`joint_mixed_matched - joint_domain`；正值表示跨域历史有帮助，负值才支持跨域负迁移。
