# Conflict-Aware Pretraining 与低秩适配实验设计

## 1. 目标与边界

本项目验证以下假设：在模型结构、数据、初始化与训练预算一致时，PCGrad 预训练得到的 SASRec backbone 是否比普通 Joint Training backbone 更容易通过低秩 LoRA 适配到各个 domain。

核心比较是：

```text
Joint pretrained backbone + LoRA(rank)
vs.
PCGrad pretrained backbone + LoRA(rank)
```

重点观察 rank 1、2、4 的 NDCG@10、HR@10 和 Recovery，而不是仅观察 PCGrad 预训练绝对性能是否高于 Joint。

本轮交付完整可运行实验管线，但只在本地使用合成五域数据完成端到端 smoke test。Amazon 五域全量预处理和所有高成本训练由用户在 RTX 4090D 服务器上运行。本轮不实现 MAMDR、MDSR-DSL、AutoCDSR、动态 rank、复杂 adapter 或新跨域模型。

已批准的数据预处理语义继续由子规格 `docs/superpowers/specs/2026-09-14-amazon-five-domain-preprocessing-design.md` 定义；完整管线不得修改其中的去重、联合迭代 k-core、确定性映射和全局 leave-one-out 口径。

## 2. 技术路线

采用原生 PyTorch 重写的 vanilla SASRec，显式暴露每层的 `W_Q`、`W_K`、`W_V` 和 `W_O`。不继续围绕旧版 fused `nn.MultiheadAttention.in_proj_weight` 做切片式 LoRA，也不引入 RecBole 或 PEFT。

这样做的原因是：

- Q/V LoRA 插入点是稳定、可测试的一等模块；
- PCGrad 可以按参数和 Transformer 子层准确收集梯度；
- 避免旧代码中的 NumPy/Tensor 往返、多进程无限 sampler、宽泛异常捕获和不可复现评测；
- 保持模型足够原始，使实验差异只来自 Joint/PCGrad 与 adaptation capacity。

旧版根目录 `main.py`、`model.py` 和 `utils.py` 移入 `legacy/pmixer_sasrec/` 并附来源说明，仅作为历史参考。新代码位于标准 `src/ftrec/` 包内，旧实现不参与任何新实验。

## 3. 推荐服务器环境

正式环境固定为：

- Linux x86_64；
- Python 3.12；
- PyTorch 2.13.0 CUDA 13.0 wheel；
- NVIDIA driver 580.65.06 或更高；
- 单卡 RTX 4090D；
- Conda 只管理 Python 和通用依赖，PyTorch 从官方 CUDA 13.0 pip index 安装。

仓库提供 `environment.yml`、锁定的核心依赖文件和环境自检命令。代码不得依赖本机已经安装的系统 CUDA Toolkit；只有 NVIDIA driver 是运行 wheel 的外部要求。

## 4. 项目结构

```text
src/ftrec/
    config.py
    reproducibility.py
    data/
        amazon.py
        preprocessing.py
        datasets.py
        sampling.py
    models/
        attention.py
        sasrec.py
        lora.py
    training/
        objectives.py
        engine.py
        single.py
        joint.py
        pcgrad.py
        adapt.py
        checkpoint.py
    evaluation/
        ranking.py
        metrics.py
    analysis/
        gradients.py
        recovery.py
        results.py
        plotting.py
    cli/
        preprocess.py
        train.py
        adapt.py
        evaluate.py
        analyze.py
        smoke.py
configs/
    data/amazon5.yaml
    model/sasrec.yaml
    experiment/single.yaml
    experiment/joint.yaml
    experiment/pcgrad.yaml
    experiment/lora.yaml
    experiment/fullft.yaml
    smoke.yaml
scripts/
    run_preprocess.sh
    run_single.sh
    run_joint.sh
    run_pcgrad.sh
    run_lora.sh
    run_fullft.sh
    run_analysis.sh
tests/
environment.yml
pyproject.toml
```

每个命令可独立运行和恢复；`smoke` 命令只负责编排这些真实组件，不维护另一套简化实现。

## 5. 数据语义与任务构造

### 5.1 统一序列

五域原始交互经过联合迭代 k-core 后，同一用户的所有交互按 `(timestamp, domain_id, parent_asin)` 排序。全局 leave-one-out 定义为最后两个交互分别作为 valid 和 test，其余交互作为 train。

### 5.2 训练样本

一个训练样本由 `context_items`、`positive_item`、`target_domain` 和 `negative_item` 构成。

- Joint/PCGrad/LoRA/FullFT 使用 mixed-domain context；
- 样本所属任务由 positive item 的 domain 决定；
- 负样本从 target domain 的商品集合中采样，并排除该用户已经交互的商品；
- 每个可用 train target position 形成一个样本，context 截断到最近 `maxlen` 个交互；
- Single-Domain 先把用户序列过滤到目标域，再构造域内 context 和域内 next-item target。

因此 LoRA 的“只使用本 domain 数据”严格解释为：只使用 positive target 属于该 domain 的监督样本；context 仍保留预训练时可见的历史跨域交互。Joint 与 PCGrad 的 LoRA 比较使用完全相同的样本和评测 cohort。

### 5.3 验证与测试 cohort

对全局 valid/test 中 target 属于 domain `d` 的用户，计入 domain `d` 指标。评测输入使用 target 之前的真实 mixed-domain 历史；test 时历史包含 valid 交互。若某用户没有至少一个历史交互则跳过，并在结果中记录跳过数量。

Single-Domain 报告同一 target cohort，但只保留历史中的目标域交互；没有域内历史的用户对所有 Single 比较统一跳过。结果文件同时记录 mixed-domain cohort 数和 Single 可评测 cohort 数，避免把样本覆盖率差异误解为模型收益。

## 6. Vanilla SASRec

### 6.1 结构

模型包含：

- 全局 item embedding，0 为 padding；
- learned position embedding；
- 两层 causal Transformer block；
- 每层显式 `q_proj`、`k_proj`、`v_proj`、`out_proj`；
- PyTorch `scaled_dot_product_attention`；
- SASRec 风格的两层 point-wise FFN，hidden dimension 保持等于 model dimension，激活为 ReLU；
- Pre-LayerNorm 残差结构和最终 LayerNorm；
- 输入 item embedding 与 candidate scoring 权重共享。

生产默认值为 `hidden_size=64`、`num_blocks=2`、`num_heads=2`、`dropout=0.2`、`maxlen=50`。所有方法共享同一份模型配置。

### 6.2 初始化与 padding

线性层和 embedding 使用显式、统一的初始化函数。padding embedding 在初始化后置零，并依靠 `padding_idx=0` 保持无梯度更新。位置编号只覆盖非 padding token。attention 同时使用 causal mask 和 key padding mask。

### 6.3 训练目标

沿用原 SASRec 的 sampled binary logistic objective：对每个 positive target 和一个同域 negative target 分别计算 `BCEWithLogitsLoss`，两项求和。默认不对 item embedding 施加额外 L2；如启用 weight decay，只作用于非 bias、非 LayerNorm 的 dense 参数。

## 7. Single、Joint 与 PCGrad

### 7.1 公平采样

Joint 和 PCGrad 每个 optimizer step 都消费五个 domain micro-batch，每个 domain 的 batch size 相同。较小 domain 的 sampler 按 seed 确定性循环并重新洗牌。两种方法必须读取相同的 batch manifest；该 manifest 记录每步使用的样本索引，使训练差异不能由抽样顺序解释。

同一 seed 的 Joint 与 PCGrad 必须从同一个初始化 checkpoint 启动，并在运行元数据中记录相同的初始化 SHA-256。两者共享最大 epoch、validation 频率、学习率调度和 early-stop patience，均按 validation macro NDCG@10 选择 best checkpoint。

Single-Domain 模型各自使用本域数据训练并独立早停。Single 主要提供负迁移参照，不参与 Joint/PCGrad backbone 的 LoRA 初始化。

### 7.2 Joint

Joint 的 step loss 是五个 domain loss 的算术平均：

```text
L_joint = mean(L_health, L_clothing, L_beauty, L_grocery, L_sports)
```

每个 domain 的 loss 单独记录。共享优化器、学习率调度、梯度裁剪、epoch 定义和 checkpoint 规则与 PCGrad 一致。

### 7.3 PCGrad

PCGrad 对五个 domain 的 backbone 原始梯度执行标准 pairwise projection：当 backbone 内的 `dot(g_i, g_j) < 0` 时，从 `g_i` 中减去其在原始、未投影 `g_j` 上的冲突分量。每个 task 的投影遍历顺序由 `(seed, global_step, task_id)` 确定，随后对投影后的 task gradients 求平均。投影范围包含 position embedding、Q/K/V/O、FFN、LayerNorm/final norm 等所有非 item-embedding 参数。

`item_embedding.weight` 不参与点积、范数、投影系数或梯度修正；五个 domain 的 sparse item-embedding raw gradients直接做算术平均。Joint 对全部参数的五个原始 task gradients 求平均。两种方法都只在聚合完成后执行同一 global gradient clipping，再执行 optimizer step，并继续使用相同的 dense AdamW 与 sparse SparseAdam 参数分组。checkpoint、resolved config 和 `result.json` 必须显式记录 `pcgrad_projection_scope`，不得把旧版 full-scope PCGrad 与 backbone-scope 结果混合比较。

### 7.4 Gradient conflict logging

Joint 与 PCGrad 都记录 projection 之前的 raw task gradients。日志至少包含：

- 每次采样的 5×5 cosine matrix；
- 每对 domain 的负余弦比例；
- full model、backbone、item embedding、position embedding；
- 每层 attention 和 FFN 参数组；
- 采样 step、epoch、seed 和方法。

训练过程日志频率由配置控制，并保留到 `*_training` 汇总中作为轨迹审计。训练停止后必须重新加载 `best.pt`，使用固定诊断 seed 和固定 batch manifest 在该 checkpoint 上计算 raw gradients；不得执行 optimizer step、修改模型参数或依赖训练结束时的 RNG 状态。正式 `gradient_conflict_summary.json` 和 `gradient_conflict_by_domain_layer.csv` 必须来自这个 best-checkpoint-local profile，并记录 checkpoint epoch、诊断 seed/steps 和 profile scope。核心 conflict↔LoRA utility 比较只使用该正式口径；训练轨迹不能替代它。

## 8. LoRA 与 Full Fine-Tuning

### 8.1 LoRA

LoRA 只插入每个 Transformer block 的 `q_proj` 和 `v_proj`：

```text
W(x) = W_base(x) + scale * B(A(x))
```

- rank 取 `{1, 2, 4, 8, 16}`；
- `A` 使用 Kaiming 初始化，`B` 初始化为零；
- `alpha=rank`，使初始 scale 恒为 1；
- base backbone、item embedding、position embedding 和 LayerNorm 全部冻结；
- 每个 domain、pretrain method、rank、seed 拥有独立 adapter checkpoint；
- 训练启动时断言可训练参数集合恰好等于 Q/V LoRA 参数。

adapter checkpoint 保存 base checkpoint SHA-256、rank、alpha、domain、seed、数据 manifest hash 和 LoRA state dict。加载时任何指纹不匹配都必须失败。

### 8.2 Full Fine-Tuning

Joint 与 PCGrad checkpoint 分别在每个 domain 上执行全参数 fine-tuning。FullFT 与对应 LoRA 使用相同 domain sample、验证 cohort、epoch/early-stop 规则和评测协议，但学习率拥有独立配置。它提供 Recovery 分母所需的 adaptation capacity 上界。

## 9. 评测协议

主结果使用 domain 内候选全集评测，并按 item chunk 计算分数以限制显存。对每个用户排除历史中已见候选，保留 ground-truth target。排序分数相同时以全局 item ID 升序作确定性 tie-break。

计算：

- HR@10；
- NDCG@10；
- domain 样本数和跳过数；
- Macro Average，只对存在有效样本的 domain 求平均。

smoke test 可使用固定、持久化的 sampled candidate set 加速，但产物必须标记 `evaluation_protocol=sampled`，不得与 full-catalog 指标合并。

每个 checkpoint 只根据 validation NDCG@10 选择，不允许读取 test 指标做早停或 checkpoint 选择。

## 10. Recovery 与判定

对 HR 和 NDCG 分别计算：

```text
Recovery_d(r) =
    (M_d^LoRA(r) - M_d^Pretrain) /
    (M_d^FullFT - M_d^Pretrain)
```

Recovery 不裁剪到 `[0, 1]`。若分母绝对值小于配置 epsilon，结果写为 NaN 并标记 `undefined_recovery=true`；若 FullFT 低于 Pretrain，也保留带符号结果并发出分析警告。

支持核心假设需要同时观察：

- 小 rank 下 PCGrad+LoRA 的绝对指标高于 Joint+LoRA；
- PCGrad 的 Recovery curve 高于 Joint；
- 提升不能仅由 PCGrad pretrain 的起点差异解释；
- 至少三个 seed 的均值和标准差方向稳定。

若 Joint 与 PCGrad Recovery 接近，或优势完全来自 pretrain gap，结果必须如实报告为不支持或不足以支持假设。

## 11. Checkpoint、结果与恢复

所有运行由规范化配置生成唯一 `run_id`，目录层级包含 stage、method、domain、rank 和 seed。每个运行写出：

- 完整 resolved config；
- 环境信息和 Git commit；
- 数据 manifest hash；
- epoch metrics JSONL；
- best/last checkpoint；
- trainable/total parameter count；
- 完成标记或结构化失败信息。

若完成标记、配置 hash、数据 hash 和 checkpoint 指纹一致，重复运行默认跳过；`--force` 才允许替换精确 run 目录。所有写入先进入 sibling staging path，再原子发布。

统一 `results.csv` 至少包含：

```text
seed, domain, pretrain_method, adapt_method, lora_rank,
split, evaluation_protocol, HR@10, NDCG@10,
num_eval_users, num_skipped_users,
num_trainable_params, num_total_params,
checkpoint_path, config_hash, data_hash
```

## 12. 图表与汇总

分析命令从结构化结果生成：

1. 每域 Single/Joint/PCGrad pretraining HR@10 与 NDCG@10；
2. 每域 LoRA rank 与 NDCG@10，包含对应 FullFT 水平线；
3. Joint/PCGrad Recovery curve；
4. raw gradient cosine heatmap 和 negative-gradient ratio。

同时输出逐 seed 原始表、mean±std 汇总表以及机器可读 JSON。绘图代码不得依赖手工复制的数字。

## 13. 本地端到端 Smoke Test

合成数据包含五个 domain、跨域重叠用户、足够的域内目标、可触发 k-core 的边、重复交互和确定性 timestamp tie。smoke 配置使用：

- CPU；
- `hidden_size=8`；
- `num_blocks=1`；
- `num_heads=1`；
- `maxlen=8`；
- 每个 stage 一个 epoch、极少 optimizer step；
- seed 仅一个；
- 全部五个 LoRA rank；
- sampled candidate evaluation。

单条命令必须实际跑通预处理、五个 Single、Joint、PCGrad、gradient logging、50 个 LoRA adaptation 组合、10 个 FullFT 组合、评测、Recovery 和四类图表。smoke 断言关注接口、冻结参数、checkpoint/结果契约与数值有限性，不把随机微型数据的模型优劣作为正确性条件。

## 14. 服务器运行方式

服务器脚本按依赖顺序分阶段运行，而不是一个不可恢复的长进程：

```text
preprocess -> single -> joint -> pcgrad -> lora -> fullft -> analysis
```

生产默认启用 BF16 autocast；模型参数和梯度累积保持 FP32。不开启 `torch.compile`，避免首次科学基线同时引入编译差异。单 GPU 运行，不实现 DDP。数据加载使用固定 worker seed、pinned memory 和可配置 worker 数。

每阶段提供 dry-run，列出将创建、跳过或恢复的 run；脚本遇到失败立即停止，并保留已完成 checkpoint。服务器完整实验默认 seeds 为 42、43、44。

## 15. 测试与完成标准

测试分为：

- 单元测试：mask、attention、padding、negative sampling、metrics、Recovery、LoRA 初始化与冻结、dense/sparse PCGrad projection；
- 集成测试：checkpoint round-trip、Joint/PCGrad batch identity、domain loss、gradient logs、atomic run publication；
- 端到端测试：完整 smoke command 生成所有预期 checkpoint、CSV/JSON 和图表；
- CPU/GPU 兼容测试：CPU 必须全通过；服务器提供 CUDA/BF16 自检和单 step GPU smoke。

本轮完成必须满足：

1. 新代码不依赖 legacy SASRec；
2. 自动化测试全部通过；
3. CPU 端到端 smoke test 实际完成所有实验分支；
4. Joint 与 PCGrad 使用相同样本 manifest；
5. LoRA 仅 Q/V adapter 可训练，rank 参数量单调增加；
6. FullFT 所有预期参数可训练；
7. 结果表、Recovery、gradient matrix 和四类图表均自动生成；
8. 服务器环境、命令、恢复方式和正式配置完整；
9. 不在本地启动 Amazon 全量训练；
10. 不声称合成 smoke 的指标支持或否定科学假设。
