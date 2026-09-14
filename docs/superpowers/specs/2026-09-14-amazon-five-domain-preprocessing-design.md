# Amazon 五域数据预处理设计

## 1. 目标与范围

本阶段只构建并验证 Amazon Reviews 2023 五域数据预处理管线，为后续 Single、Joint、PCGrad、LoRA 和 Full Fine-Tuning 实验提供同一份可复现的数据基础。本阶段不实现或运行任何模型训练。

输入域及其固定顺序如下：

1. `Health`：`Dataset/Health_and_Household.csv.gz`
2. `Clothing`：`Dataset/Clothing_Shoes_and_Jewelry.csv.gz`
3. `Beauty`：`Dataset/Beauty_and_Personal_Care.csv.gz`
4. `Grocery`：`Dataset/Grocery_and_Gourmet_Food.csv.gz`
5. `Sports`：`Dataset/Sports_and_Outdoors.csv.gz`

每个输入文件必须包含 `user_id`、`parent_asin`、`rating` 和 `timestamp`。评分值不用于阈值判断；每条有效评价均视为一次隐式正反馈。

## 2. 数据语义

### 2.1 商品身份

商品身份定义为 `(domain, parent_asin)`，而不是裸 `parent_asin`。即使两个域出现相同 ASIN，也会得到不同的全局商品 ID，从而保证各 category 的商品空间互不碰撞。

### 2.2 去重

同一 `(user_id, domain, parent_asin)` 若出现多条记录，只保留时间戳最早的一条。时间戳相同时保留首次读到的记录。去重发生在 k-core 过滤之前。

### 2.3 有效行

以下记录视为无效并跳过：

- 缺少用户 ID、商品 ID、域或时间戳；
- 时间戳无法解析为整数；
- CSV 结构与声明的表头不一致。

无效记录不会中止整次运行，但必须按原因计数并写入汇总。缺少必需列属于输入文件级错误，必须立即失败并指出文件和缺失列。

## 3. 过滤算法

五个域去重后合并成同一张用户—商品二部图，执行联合迭代 k-core：

- 用户最少交互数：10；
- 商品最少交互数：15。

每轮先基于当前图统计度数，再同时删除本轮中度数不达标的用户和商品关联边。重复执行，直到某一轮没有边被删除。阈值可由命令行覆盖，以便合成测试使用；上述数值是正式数据默认值。

收敛后的数据必须满足：

- 每个保留用户至少有 10 条交互；
- 每个保留商品至少有 15 条交互；
- 不存在重复的 `(user, domain, item)`；
- 每条交互属于且仅属于五个声明域之一。

## 4. 确定性映射、排序与切分

### 4.1 ID 映射

- 用户按原始 `user_id` 字典序映射为从 1 开始的连续整数；0 保留给 padding。
- 商品先按固定 domain 顺序、再按原始 `parent_asin` 字典序映射为从 1 开始的连续全局整数；0 保留给 padding。
- 域 ID 按第 1 节列出的固定顺序映射为 0 至 4。

相同输入和配置必须产生完全相同的映射。

### 4.2 序列排序

同一用户的五域交互合并后按以下键升序排列：

1. `timestamp`；
2. 固定 domain 顺序；
3. 原始 `parent_asin`。

这一定义消除相同时间戳造成的非确定性。

### 4.3 Leave-one-out

对每个收敛后的用户序列执行：

- `train = sequence[:-2]`；
- `valid = sequence[-2]`；
- `test = sequence[-1]`。

`position` 是用户完整序列中从 0 开始的位置；`split` 取 `train`、`valid` 或 `test`。由于正式用户阈值为 10，每位用户都必须恰有一个 valid 和一个 test 交互。

## 5. 实现架构

### 5.1 入口与配置

主入口为 `src/data/preprocess_amazon.py`，支持至少以下参数：

- 输入目录；
- 输出目录；
- 用户与商品最小交互阈值；
- SQLite 临时数据库位置；
- CSV 读取批大小；
- 是否覆盖已有完整输出。

默认配置保存在 `configs/preprocess.yaml`。核心逻辑放在可导入函数中，命令行入口只负责参数解析、日志和退出码。

### 5.2 处理阶段

1. 校验五个输入文件和表头。
2. 流式解压并把规范化记录写入 SQLite staging 表。
3. 通过唯一键聚合保留最早交互。
4. 在 SQLite 中执行联合迭代 k-core，并记录每轮删除量。
5. 创建稳定 ID 映射。
6. 按用户和确定性排序键生成序列与 split。
7. 计算域统计、用户重叠和全局汇总。
8. 写入临时输出目录，完成全部验证后再原子化发布为正式输出。

SQLite 仅是受控内存占用的中间处理引擎，不是正式实验产物。成功后删除临时数据库；失败时保留其路径和失败阶段信息，便于诊断。

## 6. 输出契约

默认输出目录为 `data/processed/amazon5/`。

### 6.1 `interactions.parquet`

每行一条保留交互，至少包含：

- `user_id`：映射后的整数用户 ID；
- `item_id`：映射后的全局整数商品 ID；
- `domain_id`：0 至 4；
- `domain`：规范域名；
- `timestamp`：整数毫秒时间戳；
- `position`：用户序列位置；
- `split`：`train`、`valid` 或 `test`。

行顺序固定为 `user_id, position`。

### 6.2 `sequences.jsonl.gz`

每行一个用户对象，包含 `user_id`，以及同长度的 `item_ids`、`domain_ids`、`timestamps` 和 `splits` 数组。它是后续序列加载器的直接输入。

### 6.3 映射文件

- `users.csv.gz`：`user_id, raw_user_id`；
- `items.csv.gz`：`item_id, domain_id, domain, parent_asin`。

### 6.4 统计文件

- `domain_stats.csv`：每个域的用户数、商品数、交互数和域内活跃用户平均序列长度；
- `user_overlap.csv`：每一对域的共有用户数和 Jaccard overlap；
- `summary.json`：输入、去重、过滤轮次、全局规模、序列长度与 split 统计；
- `manifest.json`：配置、输入文件大小与修改时间、输出 schema 版本及除 manifest 自身之外各输出文件的 SHA-256。

统计全部基于 k-core 收敛后的数据。某域的平均序列长度定义为该域交互数除以该域活跃用户数。

数据产物不写入运行时长、当前时间或随机临时路径等非确定性字段；运行时长仅输出到控制台。gzip 文件使用固定的 header 时间，使相同输入与配置产生相同内容哈希。

## 7. 可恢复性与覆盖策略

输出先写入与目标同级的唯一 staging 目录。只有当所有文件生成并通过校验后，才将 staging 目录发布为正式目录。

- 若正式输出不存在，直接发布。
- 若正式输出已存在且未指定 `--force`，立即失败，避免静默覆盖实验数据。
- 若指定 `--force`，仅替换精确的目标输出目录；不删除输入目录或其他实验结果。
- 失败时不得留下部分正式输出。

## 8. 测试与验证

### 8.1 自动化测试

使用小型合成五域压缩 CSV 覆盖：

- 最早交互去重；
- 域命名空间隔离；
- 需要多轮收敛的联合 k-core；
- 相同时间戳的稳定排序；
- 连续且稳定的用户/商品映射；
- leave-one-out 切分；
- 域统计与 pairwise overlap；
- 坏行计数和缺列失败；
- 已有输出的保护行为。

测试使用较小可配置阈值，避免依赖正式数据或 GPU。

### 8.2 全量数据验证

正式五域运行后执行独立验证程序，至少检查：

- k-core 阈值、唯一性和连续 ID；
- 每位用户恰有一个 valid 和一个 test；
- `position` 连续且排序键单调；
- Parquet、JSONL、映射表和汇总计数相互一致；
- overlap 矩阵对称，对角线等于各域用户数；
- manifest 中所有输出哈希可复算；
- 使用相同输入与配置再次运行时，输出内容哈希一致。

全量处理产生的数据文件不提交到 Git；源代码、配置、测试和设计文档提交。

## 9. 完成标准

只有同时满足以下条件，Step 1 才算完成：

1. 自动化测试全部通过；
2. 五个正式输入文件均完成全量处理；
3. 所有全量不变量检查通过；
4. 输出统计包括每域 user/item/interaction/平均序列长度和域对 user overlap；
5. 提供可复现命令、运行时长、过滤轮次和最终数据规模；
6. 未开始任何模型训练或后续实验步骤。
