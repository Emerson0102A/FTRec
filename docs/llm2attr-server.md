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

可覆盖的路径变量包括 `FTREC_ROOT`、`LLM2ATTR_ROOT`、`MDSR_SOURCE_DIR`、
`AMAZON_META_DIR`、`ATTRIBUTE_CATALOG` 与 `LLM2ATTR_OUTPUT`。

## Structured-LLM 对照组

先运行只读取标题的公平对照；它与 LLM2Attr 使用完全相同的 catalog、item ID、
属性数量、artifact 格式和下游融合层：

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

## 训练公平对照

两个主实验都沿用 `data/processed/gmflowrec-amazon` 中已有的 MDSR
train/valid/test 划分，不重新划分。先分别预训练：

```bash
ftrec-pretrain \
  --config configs/experiment/attribute_llm2attr_pretrain.yaml \
  --model-config configs/model/sasrec_llm2attr.yaml

ftrec-pretrain \
  --config configs/experiment/attribute_structured_title_pretrain.yaml \
  --model-config configs/model/sasrec_structured_title.yaml
```

再分别运行五个目标域的 LoRA + 内容融合微调：

```bash
ftrec-adapt \
  --config configs/experiment/attribute_llm2attr_adapt.yaml \
  --model-config configs/model/sasrec_llm2attr.yaml

ftrec-adapt \
  --config configs/experiment/attribute_structured_title_adapt.yaml \
  --model-config configs/model/sasrec_structured_title.yaml
```

`lora_all_content` 只训练 Transformer 的 LoRA 参数和共享内容融合层，不更新
LLM，也不更新完整 ID embedding 表。两种提取器的推荐模型结构和训练超参数完全
相同。建议先完成 seed 42 的可行性实验，再扩为相同的 5 个 seeds，并额外报告按
训练频次分桶的 head/medium/tail 指标。
