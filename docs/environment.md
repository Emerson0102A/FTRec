# 服务器环境

正式训练环境固定为 Linux x86_64、RTX 4090D、Python 3.12 和 PyTorch 2.13.0 的 CUDA 13.0 wheel。NVIDIA 驱动至少为 580.65.06；不要求单独安装系统 CUDA Toolkit，因为 PyTorch wheel 自带运行时。

## 安装

```bash
conda env create -f environment.yml
conda activate ftrec
python -m pip install -r requirements-cu130.txt
python -m pip install -e . --no-deps
bash scripts/check_environment.sh
```

PyTorch 必须显示 `2.13.0+cu130`，`torch.version.cuda` 必须显示 13.0，且自检必须看到 GPU、BF16 支持并完成一次 CUDA 张量计算。Conda 只管理 Python；不要再安装 `cudatoolkit` 或 Conda 版 PyTorch，以免覆盖官方 cu130 wheel。

## 数据放置

把 Amazon Reviews 2023 的五个 CSV gzip 文件放入仓库的 `Dataset/`：

```text
Health_and_Household.csv.gz
Clothing_Shoes_and_Jewelry.csv.gz
Beauty_and_Personal_Care.csv.gz
Grocery_and_Gourmet_Food.csv.gz
Sports_and_Outdoors.csv.gz
```

每个文件至少需要 `user_id,parent_asin,rating,timestamp` 四列。预处理会保留最早的重复 user-item 交互，执行五域联合迭代 k-core，并生成确定性的全局 ID 与 leave-one-out 切分。

## 常见诊断

- `torch.cuda.is_available() == False`：先检查 `nvidia-smi` 和驱动版本，再确认没有装到 CPU wheel。
- Torch 版本不含 `+cu130`：重新执行 requirements 文件中的官方 wheel 安装，不要混用 Conda 的 PyTorch 包。
- 显存不足：先降低 batch size；评测已使用候选分块，不应改成抽样评测作为正式结果。
- 结果目录冲突：先用 `--dry-run` 核对配置和数据指纹；只有确实要替换同一路径时才使用 `--force`。
