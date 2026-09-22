# GMFlowRec 论文复现说明

本实现依据 Ye 等人的 *Gaussian Mixture Flow Matching with Domain Alignment
for Multi-Domain Sequential Recommendation*（arXiv:2510.21021v1）完成，入口为
`main_gmflowrec.py`，模型与数据代码分别为 `gmflowrec.py` 和
`gmflowrec_data.py`。

本实现没有复制、导入或依赖 `Lakonik/GMFlow` 的源码。该仓库及其对应论文仅用于核对
“以高斯混合参数化速度分布”和“少步 ODE 采样”这两个上游概念；GMFlowRec 的推荐模型、
损失和数据管线均依据 GMFlowRec 论文在本仓库中独立实现。

## 已对应的论文组件

- 式 (1)(2)：同一个 Transformer 权重分别使用全局因果掩码和“同域 + 因果”掩码，得到
  domain-invariant 与 domain-specific 表示。
- 式 (3)：取目标域最近一次历史状态；目标域冷启动时只使用目标域嵌入。
- 式 (4)(5)：在目标域完整物品词表上计算 domain-aligned prior 的交叉熵。
- 式 (6)--(9)：插值状态与 invariant prior 融合，由 MLP 输出 K 个混合权重、均值和
  球形标准差。
- 式 (10)--(12)：GMM 负对数似然、直接作用于 GMM 期望速度 `mu` 的推荐损失，
  以及 prior 损失联合训练。推荐损失不再使用含真实目标物品 embedding 的插值终点，
  避免训练态目标泄漏。
- Algorithm 2：从 invariant prior 出发，以一阶反向 Euler 求解器生成目标物品表示。
- 论文评测协议：正样本加同域随机负样本，报告 HR/NDCG@5、@10，并同时输出每个域结果。

## 原论文未定义或互相矛盾之处

v1 原稿没有足够信息支持逐比特复现，以下不是实现疏漏，而是必须显式作出的选择：

1. 式 (8) 把 GMM 写成速度分布，但式 (10) 又称其似然目标为 `x0`；Algorithm 2 则把其
   均值作为速度相加。本实现采用与路径 `x_t=(1-t)x0+t*x1` 一致的反向速度
   `x0-x1` 作为 GMM 目标。
2. 式 (11) 直接用速度预测物品，而 Algorithm 2 返回积分后的 `x0`。本实现严格按式 (11)
   在训练时使用 GMM 期望速度做推荐，推理时按 Algorithm 2 使用积分后的终态做推荐。
3. 式 (7) 没有写入时间 `t`，式 (10) 却显式以 `t` 为条件，架构图也含 `t`。本实现为
   `t` 使用两层 MLP 时间嵌入。
4. 论文没有公布 Transformer 层数、头数、MLP 宽度、融合系数 lambda、ODE 步数、
   GMM 方差参数化和 early-stopping patience。本实现全部做成命令行参数，默认值为：
   2 层、2 头、`lambda=0.9`、8 个混合分量、8 个 ODE 步。
5. 附录把 beta 搜索集合写成 `0,00001`，按排版意图解释为 `0.00001`。默认采用
   `alpha=0.1, beta=0.0001`，正式实验应按论文集合调参。

因此，这是一份可运行、可测试、数学自洽的 paper-driven reproduction；在作者代码或
补充材料公开前，不应声称能够严格复现论文表 1 的数值。

## 运行

先做 CPU 小规模连通性检查：

```powershell
python main_gmflowrec.py `
  --parquet_dir ..\..\data\MDSR-Amazon `
  --device cpu `
  --run_dir runs/gmflowrec-smoke `
  --epochs 1 `
  --batch_size 8 `
  --eval_batch_size 8 `
  --num_eval_negatives 9 `
  --hidden_units 16 `
  --num_heads 2 `
  --num_mixtures 2 `
  --ode_steps 2 `
  --max_train_examples 32 `
  --max_eval_examples 16 `
  --amp false
```

论文规模的默认训练（建议在 GPU 上执行）：

```powershell
python main_gmflowrec.py `
  --parquet_dir ..\..\data\MDSR-Amazon `
  --run_dir runs/gmflowrec-k8-seed2026 `
  --device cuda
```

默认使用论文正文的 `d=64, maxlen=50, batch=256, lr=0.0001, epochs=100`，验证集
NDCG@10 选择最佳 checkpoint。每次执行验证评估时也会同步执行一次测试集评估并写入
`history[*].test`；checkpoint 选择仍然只依据验证集，测试集不会参与 early stopping。
为了复核论文结果，应对 `alpha、beta、K、dropout` 和
上述未公布参数做验证集搜索，并至少运行五个随机种子。

上面的 `..\..\data\MDSR-Amazon` 是当前 `.worktrees/gmflowrec-reproduction`
相对主工作区共享数据集的路径；如果将该分支单独检出到其他位置，应改为当地的数据路径。
