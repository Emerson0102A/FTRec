# CGRec 官方代码本地验证

此目录保存 [cpark88/CGRec](https://github.com/cpark88/CGRec) 的官方实现，
固定于提交 `9108dd04637decd616448e30d1d604eddb2a3543`。`README.md` 和
`src/` 中的 Python、shell 源文件均保持上游原样；`verify_cpu.py` 是本仓库新增的
最小验证入口。

上游仓库没有 `LICENSE` 文件，也没有 README 所说的 `requirements.txt`。
代码的公开可读性不等于获得再分发许可；本目录仅用于本地研究验证，向外发布前应
先确认作者许可。

## 已纳入和未纳入的文件

- 纳入官方 README、全部 Python/shell 源文件、5,000 名用户的官方样例 pickle、
  三个 402 字节的域/类别词表。
- 未纳入架构图片和 PDF，因其与运行无关。
- 未纳入约 30 MB 的 `src/pretrained_model/voca/amazon_voca_item_202312.ep`。
  官方 `run_pretrain.py` 依赖此文件；完整 GPU 训练前需从上述提交单独下载并放在
  该路径。`verify_cpu.py` 从样例中推算物品 embedding 的大小，不读取该文件。

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

## 完整效果验证所需工作

1. 官方脚本强制使用 CUDA/NCCL 和分布式训练；当前机器安装的 PyTorch 为 CPU 版。
   原始 `train.sh` 还使用了与解析器不一致的 `--epoch`、`--num_attention_head` 参数。
2. CGRec 在模型中固定目标域为 `5`、源域为 `6–9`。要与 GMFlowRec 做公平对比，
   需先将两者映射到同一组域、相同的用户划分、负采样与评测协议。
3. 官方 `datasets.py` 当前把 `cat1`、`cat2` 从域序列读取，而非独立类别序列。
   正式复现前需核对作者预期数据格式并修正这一点。
4. 目前只有官方小样例，没有对应的完整 CGRec 论文实验数据与运行结果。

因此，本目录已经可以审查模型实现和运行小规模训练检查；完整论文指标与
同数据对比仍需解决上述数据、脚本和算力条件。
