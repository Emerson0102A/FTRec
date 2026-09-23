# CGRec 官方代码本地验证

此目录保存 [cpark88/CGRec](https://github.com/cpark88/CGRec) 的官方实现，
固定于提交 `9108dd04637decd616448e30d1d604eddb2a3543`。`README.md` 和
`src/` 中的 Python、shell 源文件均保持上游原样。本仓库新增 `verify_cpu.py`、
`parquet_data.py`、`parquet_model.py`、`parquet_eval.py`、`train_parquet.py`、
`run_parquet_sweep.sh` 和 `summarize_parquet.py`；它们提供样例检查及基于
GMFlowRec Parquet 的完整训练/评测入口。

上游仓库没有 `LICENSE` 文件，也没有 README 所说的 `requirements.txt`。
代码的公开可读性不等于获得再分发许可；本目录仅用于本地研究验证，向外发布前应
先确认作者许可。

## 已纳入和未纳入的文件

- 纳入官方 README、全部 Python/shell 源文件、5,000 名用户的官方样例 pickle、
  三个 402 字节的域/类别词表。
- 未纳入架构图片和 PDF，因其与运行无关。
- 未纳入约 30 MB 的 `src/pretrained_model/voca/amazon_voca_item_202312.ep`。
  官方 `run_pretrain.py` 依赖此文件；本仓库的 Parquet 入口直接从 `mappings.pkl`
  获取物品数，不读取该文件。

## 当前机器上的连通性验证

从本 worktree 根目录执行：

```powershell
python FTRec-CGRec/verify_cpu.py --train-users 1 --eval-users 8 --max-seq-length 8 --hidden-size 8
```

验证入口依赖 `torch`、`numpy`、`tqdm` 和 `pandas`；当前 worktree 环境已具备。

入口使用官方样例与 `CausalDataset`、`CausalModel`，保留 Shapley 权重计算，
在 CPU 上做一次参数更新，然后按官方“1 个正例 + 100 个随机负例”的候选形式计算
HR@5/NDCG@5。这里特意缩小隐藏维度和序列长度，以确认代码路径可运行。
种子 42 的本地检查得到训练损失 `2.193335`、HR@5 `0.0`、NDCG@5 `0.0`
（8 名样例用户）。这些数值只表示连通性，不是 CGRec 的论文效果，也不能与
`FTRec-GMFlowRec` 的结果比较。

## 用 GMFlowRec 的 Parquet 在服务器训练

`PARQUET_DIR` 必须包含 `mappings.pkl`、`train_new.parquet`、`valid_new.parquet`
和 `test_new.parquet`。先运行与开源代码数据集行为一致的模式：官方代码将
`cat1` 和 `cat2` 都直接设为域序列，随附的两个类别词表也只有域名。此模式
不需要另外的类别文件。从仓库根目录执行：

```bash
python -m pip install -e .
python FTRec-CGRec/train_parquet.py \
  --parquet_dir /path/to/MDSR-Amazon \
  --official_domain_categories \
  --run_dir runs/cgrec-official-domains \
  --target_domain 0 \
  --device cuda \
  --seed 42
```

先用下面的短程运行检查服务器环境；其中减少样本和负例，因此结果**不能**用于
与 Table 1 比较：

```bash
python FTRec-CGRec/train_parquet.py \
  --parquet_dir /path/to/MDSR-Amazon \
  --official_domain_categories \
  --run_dir runs/cgrec-smoke \
  --target_domain 0 --device cuda --epochs 1 \
  --max_train_examples 128 --max_eval_examples 32 \
  --num_eval_negatives 9
```

确认显卡可用、短程运行完成后，对五个域分别训练五个种子：

```bash
bash FTRec-CGRec/run_parquet_sweep.sh /path/to/MDSR-Amazon runs/cgrec-official-domains \
  --official_domain_categories
python FTRec-CGRec/summarize_parquet.py --run_dir runs/cgrec-official-domains
```

为了检验论文所述的真实两级类别，再运行单独的元数据模式。需要与
`mappings.pkl` 的 ASIN 和 item ID 逐条对齐的 `catalog.jsonl.gz`。本机已有
`data/attribute_experiment/catalog.jsonl.gz`；该文件较大，不在 Git 中，需单独
放到服务器。如服务器有 Amazon 2023 各域 `meta_*.jsonl.gz` 原始元数据，也可用
本仓库的命令生成：

```bash
ftrec-attribute-catalog \
  --dataset-dir /path/to/Amazon2023-metadata \
  --mappings /path/to/MDSR-Amazon/mappings.pkl \
  --output /path/to/catalog.jsonl.gz
```

然后运行元数据层级模式：

```bash
bash FTRec-CGRec/run_parquet_sweep.sh /path/to/MDSR-Amazon runs/cgrec-metadata \
  --category_catalog /path/to/catalog.jsonl.gz
python FTRec-CGRec/summarize_parquet.py --run_dir runs/cgrec-metadata
```

不要将不同类别模式写入同一 `run_dir`；汇总工具会拒绝混合结果。

目标域编号：`0` Health、`1` Clothing、`2` Beauty、`3` Grocery、`4` Sports。
批量脚本依次使用种子 42–46。单次运行把验证集 NDCG@10 最优权重保存为
`runs/cgrec-official-domains/domain-0/seed-42/best.pt`，同时保存 `config.json` 和
`results.json`。批量脚本重启时跳过已有非空 `results.json` 的域/种子，
可继续未完成的组合；被中断的单次训练会从头开始。

默认配置为序列长度 50、embedding 64、2 层、2 个头、dropout 0.1、
Adam 学习率 0.001、batch 256、最多 100 epochs、验证集连续 10 次未改进时停止。
验证和测试均为 1 个正例 + 999 个同域、未见过的负例；负例生成逻辑直接复用
`FTRec-GMFlowRec/gmflowrec_data.py`，固定评测种子 3407。记录四项
HR/NDCG@5/10，汇总脚本将五个种子的均值和样本标准差与 GMFlowRec Table 1
的 CGRec 数值并列展示。默认会预生成验证/测试候选，内存有限时可用
`--no_precompute_eval`，或减小 `--eval_batch_size`。服务器依赖带 CUDA 的
PyTorch；`pip install -e .` 应在已安装兼容 CUDA 版 PyTorch 的环境中执行。

## 与论文 CGRec 的差异

这个入口保留官方序列编码器、三级损失和 Shapley 计算，目标域映射为模型固定
使用的 5，其他四域映射为 6–9。`--official_domain_categories` 按公开代码将
两级类别都设为域 ID。这有助于核对开源实现的结果，但不能当成论文所称的
真实层级类别。Parquet 本身只有物品、域和时间戳。`--category_catalog` 的两级
类别由同一商品 ASIN 对齐的 Amazon 元数据 `categories` 路径恢复：
`cat2 = categories[1]`（域下较粗层），`cat1 = categories[2]`（较细层）。
键包含域和父路径，避免同名类别碰撞。缺失层级用专用标记，不伪造具体类别。
类别 ID 也与官方词表一样保留 0–4 给特殊符号，真实类别从 5 开始。
本地目录对全部 218,846 个物品都有元数据，其中 864 个没有可用的粗类别。
两种类别模式均启用官方的 `hierarhical=y` 路径；要单独运行不使用类别的消融实验，
必须显式指定 `--item_only`。

原作者未公开 Table 1 所用的类别分层映射规则，因此这个由 Amazon 元数据恢复
的层级是**明确记录的近似**，不能声称类别 ID 与原论文完全相同。此外，训练
使用 Parquet 已发布的 `train_new`，验证和测试使用其现成划分，无法证明它与
原作者 Table 1 的私有数据处理和随机负例完全一致。
完整 Shapley 路径每个训练 batch 需要多次编码，正式 5×5 次运行可能耗时很长；
先用单域正式配置测量一个 epoch 的耗时，再安排整组实验。
