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

## 参数匹配的 PEFT 诊断

当 Q/V LoRA 与 FullFT 之间存在明显差距时，使用下面三组配置判断瓶颈来自
LoRA 插入范围还是适配结构。配置默认读取 `runs-lr1e-4` 中以 `1e-4` 训练的
标准 `joint` backbone，并以 `1e-4` 进行适配；域内 target、混合历史、
固定 999 个负样本和 epoch-0 checkpoint 选择规则保持不变：

```bash
bash scripts/run_lora_all.sh --pretrain-method joint --seed 42 --dry-run
bash scripts/run_houlsby.sh --pretrain-method joint --seed 42 --dry-run
bash scripts/run_pfeiffer.sh --pretrain-method joint --seed 42 --dry-run

bash scripts/run_lora_all.sh --pretrain-method joint --seed 42
bash scripts/run_houlsby.sh --pretrain-method joint --seed 42
bash scripts/run_pfeiffer.sh --pretrain-method joint --seed 42
```

默认容量经过近似匹配：Q/V LoRA rank 8/16 分别约 4k/8k 参数；all-linear
LoRA 使用 rank 3/5，Houlsby 使用 bottleneck 8/16，Pfeiffer 使用 bottleneck
16/32。`lora_all` 同时适配 Q/K/V/O 与两层 FFN；Houlsby 在 attention 和 FFN
后各放一个残差 bottleneck；Pfeiffer 只适配 FFN。先跑标准 Joint、seed 42，
只有差异明确后再补其余 seeds。`joint_proportional` 应作为独立扩展实验，避免
把主干学习率和采样策略同时改变。

每个 `result.json` 和汇总 CSV 都记录 `target_modules`、`rank`、
`bottleneck_size` 和实际可训练参数量，比较时优先使用实际参数量而不是只看
rank 或 bottleneck 名义值。

## 目标域 item embedding 消融

当 Transformer 侧的 LoRA/Adapter 与未适配主干基本持平时，使用下面两组实验
判断瓶颈是否位于商品表示和打分空间：

```bash
bash scripts/run_embedding.sh --seed 42 --dry-run
bash scripts/run_lora_all_embedding.sh --seed 42 --dry-run

bash scripts/run_embedding.sh --seed 42
bash scripts/run_lora_all_embedding.sh --seed 42
```

两组配置均读取 `runs-lr1e-4` 中的 `joint_proportional` 主干，适配学习率和
embedding 学习率均为 `1e-4`。`embedding` 只训练目标域商品的零初始化残差表；
`lora_all_embedding` 在此基础上同时训练 all-linear LoRA rank 5。非目标域商品
映射到固定的零残差，因此即使历史序列跨域，也不会修改其他域的商品表示。

残差表在 epoch 0 与原主干完全等价，checkpoint 只保存目标域残差而不复制整张
全域 embedding。`result.json` 额外记录 `target_embedding_rows`，用于核对各域
实际参与适配的商品数量。

## 低学习率 FullFT 对照

如果 embedding 消融仍与未适配主干持平，使用下面的 FullFT 对照判断完整模型
是否能够从相同的域内训练数据获得额外信号：

```bash
bash scripts/run_fullft_lr1e4.sh --seed 42 --dry-run
bash scripts/run_fullft_lr1e4.sh --seed 42
```

该配置读取 `runs-lr1e-4` 中的 `joint_proportional` 主干，dense 参数和 item
embedding 均使用 `1e-4`，只运行 seed 42。它与旧的 `configs/experiment/fullft.yaml`
相互独立，不覆盖原来的 `lr=1e-3` FullFT 设计。

## 多负样本 adaptation 对照

若低学习率 FullFT 仍无法提高 NDCG，可将训练目标从每个正样本搭配 1 个负样本
改为搭配 31 个同域、互不重复且未见过的负样本，使训练候选更接近 999 负样本的
sampled evaluation。正样本 BCE 与所有负样本 BCE 的均值各占一项，因此负样本数
增加不会把负项的总权重放大 31 倍。

```bash
bash scripts/run_lora_all_multineg31.sh --seed 42 --dry-run
bash scripts/run_fullft_multineg31.sh --seed 42 --dry-run
bash scripts/run_embedding_multineg31.sh --seed 42 --dry-run
bash scripts/run_lora_all_embedding_multineg31.sh --seed 42 --dry-run

bash scripts/run_lora_all_multineg31.sh --seed 42
bash scripts/run_fullft_multineg31.sh --seed 42
bash scripts/run_embedding_multineg31.sh --seed 42
bash scripts/run_lora_all_embedding_multineg31.sh --seed 42
```

两组配置均读取 `runs-lr1e-4` 的 `joint_proportional` 主干，学习率为 `1e-4`；
LoRA 使用 all-linear rank 5，FullFT 同时以 `1e-4` 更新 dense 参数与 embedding。
`embedding` 只更新目标域商品残差，`lora_all_embedding` 同时更新该残差与
all-linear LoRA rank 5，用于判断 FullFT 的增益是否来自商品表示及其与 Transformer
的协同适配。
评测仍固定使用 999 个负样本。结果单独写入 `runs-multineg31/`，并在
`result.json` 和汇总 CSV 中记录 `num_train_negatives: 31`，不会覆盖单负样本实验。

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

每个训练目录包含 resolved lineage、`batch_manifest.json`、逐 epoch `metrics.jsonl`、`best.pt`、`last.pt`、`result.json` 和 `COMPLETE.json`。`batch_manifest.json` 是小型确定性配方（seed、算法、总 step 和各域样本指纹），实际 batch 在训练时流式生成，不再保存数千万个样本编号。

Joint/PCGrad 的冲突统计分为两个口径。训练轨迹文件用于审计 early stopping 前后的变化：

- `gradient_conflicts.jsonl`；
- `gradient_conflict_pairs.csv`；
- `gradient_conflict_summary_training.json`；
- `gradient_conflict_by_domain_layer_training.csv`。

加载 `best.pt` 后，程序会用固定 `checkpoint_seed` 和固定诊断 batch 重新计算 raw domain gradients，不执行 optimizer step。以下文件是后续 conflict↔LoRA utility 分析的正式口径：

- `gradient_conflicts_best_checkpoint.jsonl`；
- `gradient_conflict_pairs_best_checkpoint.csv`；
- `gradient_conflict_summary.json`；
- `gradient_conflict_by_domain_layer.csv`：`domain,layer,q_conflict,v_conflict,qv_conflict`，可与 LoRA gain 直接 join；
- `gradient_profile_batch_manifest.json`：固定诊断 batch 的可复现配方。

正式配置使用 `gradient_conflict.checkpoint_steps: 100`、`checkpoint_seed: 2026`。分析器在同一 run 同时存在两种日志时优先读取 best-checkpoint 文件。所有记录包含动态生成的 `block_l_q`、`block_l_v`、`block_l_qv`，并保留 attention、FFN、item embedding、`backbone` 与 full groups。`result.json` 记录 profile scope、checkpoint epoch、诊断步数/种子和 PCGrad 投影范围。LoRA checkpoint 只保存 adapter 张量和 backbone SHA-256。

PCGrad 默认 `pcgrad_projection_scope: backbone`：只对 position embedding、Q/K/V/O、FFN、LayerNorm 等非 item-embedding 参数做 gradient surgery；`item_embedding.weight` 的五域 raw gradients 不投影，直接做算术平均。旧版全参数 PCGrad 结果与该口径不同，不能混合比较。

## 共享参数与混合上下文的 2×2 消融

Single 明显优于 Joint 时，差距可能来自两种不同机制：五域共享同一套模型参数，或
输入序列混入其他域的历史行为。以下四种模式只改变这两个因素，不修改 SASRec
结构、学习率、balanced batch、固定候选集或 early stopping：

为了避免把 cohort 差异误认为上下文收益，受控四格都只保留“目标域中至少已有一次
历史行为”的 target。原始 `joint` 会额外包含只有跨域历史的 target，因此保留为外部
参考，但不直接代替 matched 四格中的共享+混合模型：

| method | 模型参数 | 输入上下文 | target cohort | 运行数（单 seed） |
| --- | --- | --- | --- | ---: |
| `single` | 每域独立 | 仅目标域 | matched | 5（已有） |
| `single_mixed` | 每域独立 | 五域混合 | matched | 5 |
| `joint_domain` | 五域共享 | 仅目标域 | matched | 1 |
| `joint_mixed_matched` | 五域共享 | 五域混合 | matched | 1 |
| `joint` | 五域共享 | 五域混合 | 全部可评估 target | 1（已有，仅参考） |

已有 `single` 结果可直接作为受控四格的一角，`joint` 结果保留为全 cohort 参考，
二者都无需重跑。补 seed 42 的另外三个受控模式前可先检查计划：

```bash
bash scripts/run_single_mixed.sh --seed 42 --dry-run
bash scripts/run_joint_domain.sh --seed 42 --dry-run
bash scripts/run_joint_mixed_matched.sh --seed 42 --dry-run
```

确认后可放入 tmux 会话；同一张 GPU 上最多先尝试两个进程并发，是否缩短总墙钟
时间取决于 GPU 利用率和显存。先观察 `nvidia-smi`，若吞吐下降明显就顺序运行：

```bash
mkdir -p logs
bash scripts/run_single_mixed.sh --seed 42 2>&1 | tee logs/single-mixed-seed-42.log

bash scripts/run_joint_domain.sh --seed 42 2>&1 | tee logs/joint-domain-seed-42.log

bash scripts/run_joint_mixed_matched.sh --seed 42 \
  2>&1 | tee logs/joint-mixed-matched-seed-42.log
```

`single_mixed` 依次训练五个独立模型；`joint_domain` 和
`joint_mixed_matched` 各训练一个五域 balanced-batch 共享模型，共新增 7 个 run。
三个配置均为最多 100 epoch、patience 10、学习率和 embedding 学习率 0.001，并
从作者预生成的固定候选矩阵中选择 matched 用户对应的 999 个负样本。冲突日志对
这些纯消融关闭，以免增加额外开销；这不会改变 optimizer update。

完成后重新汇总已有全部结果：

```bash
ftrec-analyze \
  --runs-root runs \
  --output-dir results/context-ablation-seed-42 \
  --force
```

重点比较两组受控差值：

- 上下文干扰：`single_mixed - single`，以及
  `joint_mixed_matched - joint_domain`；
- 参数共享影响：`joint_domain - single`，以及
  `joint_mixed_matched - single_mixed`；
- cohort 扩展效应（仅辅助解释）：`joint - joint_mixed_matched`。

若前两项明显为负，主要问题是混合序列上下文；若后两项明显为负，主要问题是共享
参数造成的负迁移；两类差值都明显为负则说明两种机制同时存在。先用 seed 42 判断
效应方向，只有差值大于评测噪声后再补 43、44 等随机种子。

## 自然比例 Joint 对照

`joint_proportional` 用来区分共享参数带来的迁移与五域均衡重加权带来的收益。它与
`joint` 使用相同的共享 SASRec、混合域上下文、训练 target cohort、固定评测候选集
和 early stopping；唯一变化是训练 batch 的域构成。

配置中的 `batch_size: 256` 仍表示 balanced Joint 的单域基准。每个 optimizer step
的总 batch 固定为 `5 × 256 = 1280`，然后按各域训练样本数用 largest-remainder
方法分配。当前 GMFlowRec-Amazon 数据对应约：

```text
Health       317
Clothing     530
Beauty       224
Grocery      143
Sports        66
total       1280
```

各域 loss 先在域内求平均，随后按该域实际 micro-batch 大小加权合并梯度；这严格
等价于对1280条样本的 pooled loss 求平均。`steps_per_epoch: auto` 仍为384，因此
每轮总训练量为491,520条，和491,445条训练样本近似相等。

该低学习率控制实验使用独立目录，不覆盖已有结果：

```bash
bash scripts/run_joint_proportional.sh --seed 42 \
  2>&1 | tee joint-proportional-lr1e-4-seed42.log
```

输出位于：

```text
runs-lr1e-4/pretrain/joint_proportional/all-domains/seed-42
```

训练启动后可检查 `batch_manifest.json`。其中 `algorithm` 应为
`proportional-shuffle-cycle-v1`，每个域同时记录 `count` 和 `batch_size`。正式比较
前还应检查 Joint、Single 与本实验的 `resolved_config.json`，确保 `lr` 和
`embedding_lr` 都是 `0.0001`。

分析时重点报告：

- 非均衡多域整体效应（参数共享与混合上下文的合计）：
  `joint_proportional - single`；
- 均衡重加权效应：`joint - joint_proportional`；
- 五域等权 Macro NDCG@10，以及按测试用户数加权的总体 NDCG@10。

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
