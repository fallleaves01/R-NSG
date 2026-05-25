# R-NSG 工程背景：实验推进与代码修改指南

本文档面向需要**接手实验、改参数、对齐脚本产出**的协作者（含大模型）。侧重仓库布局、构建方式、二进制入口、数据流与常见工作流，不重复基础向量检索概念。

---

## 1. 项目是什么

- **语言**：C++（`xmake` 构建），Python 脚本用于网格实验、对比与检验。
- **核心目标**：在**带范围约束（label / time-like）**的近似最近邻检索上，构建 **TDF（时间/范围相关）图索引**并做 **beam 搜索**。
- **两条主线**：
  - **Baseline / R-NSG 系**：`include/RNSG/`、`src/main_rnsg.cpp` 等。
  - **Enhanced R-NSG**：`include/EnhancedRNSG/Builder.hpp`、`Worker.hpp`，入口 `src/main_enhanced_rnsg.cpp`，子命令 `knng` / `build` / `query` / `groundtruth`。

详细算法与伪代码见同目录 **`ALGORITHM_PSEUDOCODE.md`**。

---

## 2. 目录与职责速查

| 路径 | 内容 |
|------|------|
| `xmake.lua` | 目标、源文件、依赖；新增 `main_*.cpp` 时常需在此注册 |
| `include/EnhancedRNSG/Builder.hpp` | 增强建图主逻辑（并行、profile JSON） |
| `include/EnhancedRNSG/Worker.hpp` | CLI 选项与 `knng`/`build`/`query`/`groundtruth` 实现 |
| `include/RNSG/Searcher.hpp` | `beam_search` / `beam_search_range` 及边选择策略 |
| `include/RNSG/Builder.hpp` | 基线 RNSG 与 `nn_descent`、共享的 `reorder_prefix_edges` 等 |
| `include/Graph/` | 图索引类型、`TDGraphIndexBase`、子图算子 |
| `scripts/*.py` | 参数扫描、与 iRange 等对比、`search_summary` 类汇总 |
| `src/main_analyzer.cpp` | 更重的分析/精确 trace（与默认 query 路径分离） |
| `doc/` | 算法说明与本文档 |

---

## 3. 构建与运行

```bash
# 在项目根目录
xmake
xmake install  # 若配置需要

# 典型产物路径（架构/模式可能不同）
./build/linux/x86_64/release/enhanced_rnsg
```

若路径不一致：`find build -name enhanced_rnsg -type f`。

**增强索引典型调用形态**（具体路径与参数以脚本为准）：

1. `enhanced_rnsg knng -d <db.fvecs> -k <K> -g <knng.graph>`
2. `enhanced_rnsg build -s <range_step> -d <db> -k <knng> -l <labels.json> -i <out.graph> -m <ef_max> [大量 Enhanced 选项]`
3. `enhanced_rnsg query -d <db> -i <index> -q <queries> -l <labels> -Q <qrange.json> -n <k> -s <beam> -t <trunc> -r <result.json> [nav/pick 选项]`

`Worker::query` 在写结果前会把节点 id **从重排坐标映回原 id**（`ord`），与 groundtruth 对齐时注意脚本是否也在同一坐标系。

---

## 4. 数据约定（实验复现关键）

- **向量**：如 `database_vectors.fvecs`、`query_vectors.fvecs`。
- **标签**：JSON 数组，与数据库行一一对应；建图会按标签排序**重排**向量。
- **查询范围**：`qrange` 通常为扁平数组 `[l0,r0,l1,r1,...]`（见 `Worker` 读取方式）。
- **Ground truth**：JSON；query 结果与之比 recall 时在 CPU 侧做距离 tie 处理（见 `Worker::query` 中 hash 计数逻辑）。
- **KNNG 文件**：磁盘上的图格式由 `Graph::GraphIndex` 读写；Enhanced 建图会按 `index,pos` **映射到重排下标**。

外部大规模路径（如 `/mnt/win-dai/Vectors/...`）在部分机器上挂载；**以当前环境实际路径为准**，脚本里常通过变量传入。

---

## 5. 实验产物与命名

常见模式（与参数扫描脚本一致）：

- 索引：`enh_<tag>.graph` 或脚本约定前缀
- 日志：`build_<tag>.log`、`query_<tag>.log`
- 结果：`result_<tag>.json`
- 汇总：`search_summary.json` / `search_summary.md`

**Tag 示例**：`sc{seedcap}_sk{seedkeep}_rm{reverse_mode}` —— 具体分隔符以脚本为准。

分析目录常包含时间戳子目录；用 `glob` 搜索 `search_summary.*` 可定位最近一次参数搜索。

---

## 6. 修改代码时的注意事项

1. **只改请求范围**：Enhanced 选项多，改 `BuildOptions` / CLI 时需同时更新 `Worker::build()` 赋值与 `Builder::build` 日志、`build_profile_json` 字段（若需要）。
2. **保持与 RNSG 共用层一致**：三角剪枝、`reorder_prefix_edges` 在 `RNSG::` 命名空间；改语义可能影响 baseline 与增强两边。
3. **并行**：建图使用 OpenMP `parallel for`；新增 per-node 状态注意线程安全（已有模式：`profiles[omp_get_thread_num()]`）。
4. **查询热路径**：`Searcher::beam_search*` 的内层循环；改前用小规模数据对比 QPS/recall。
5. **新增可执行文件**：在 `xmake.lua` 添加 target 与 `add_files("src/xxx.cpp")`。

---

## 7. iRange 对比基线（iRangeGraph）

**iRange 实现不在本仓库**：实验脚本默认指向 sibling 工程 **`FANNBench/iRangeGraph`**（需单独克隆并用其 CMake 编译出 `buildindex` / `search`）。本仓库通过 **Python 脚本** 调外部二进制，与 R-NSG / Enhanced 使用**同一套 prepared 数据**（二进制向量 + int32 属性 + qrange/gt 等），以保证可比性。

### 7.1 二进制与默认路径

| 角色 | 典型路径（可覆盖） |
|------|----------------------|
| 建索引 | `--irange-build-bin` → `.../iRangeGraph/build_local/tests/buildindex` |
| 查询 | `--irange-search-bin` → `.../iRangeGraph/build_local/tests/search` |

脚本：`scripts/arxiv_r_prepare_and_compare.py`、`scripts/arxiv_r_grid_compare.py`（后者假设索引已存在，只跑 search）。一键脚本支持 **`--skip-irange`**（只跑 R-NSG）、**`--force-irange-build`**（强制重建索引）等开关。

### 7.2 数据准备（与脚本一致）

`arxiv_r_prepare_and_compare.py` 中 `prepare_assets()` 在 `work_root/prepared/` 下生成（默认命名）：

- **`data_base.bin`** / **`data_query.bin`**：由 fvecs 转换，首部 `int32 n, dim` + 连续 float32 向量；
- **`data_attr_update_date.bin`**：库点属性（int32，与 `database_attributes.jsonl` 的 `update_date` 一致）；
- **`data_qrange_r.bin`**：查询范围（pairs of int32 `[l,r]`）；
- **`gt_r_k{topk}.bin`**：groundtruth（`ivecs` 转存）；
- 同步写出 **`attr_update_date.json`**、**`qrange_r.json`**、**`gt_r_k{topk}.json`** 供 R-NSG / 人工检查。

**R-NSG** 建图/query 用 JSON；**iRange** `buildindex`/`search` 用上述 **bin**（脚本里 `PreparedPaths.base_bin` 等字段）。

### 7.3 建图：`buildindex`

以下为脚本中实际拼接的参数字面（单核 `taskset` 仅为调度隔离，与 R-NSG 同脚本对齐）：

```text
buildindex
  --data_path      <base_bin>        # 库向量二进制
  --index_file     <输出索引路径>     # 如 work_root/irange/index/arxiv_r_M64
  --attr_file      <attr_bin>        # 点属性（与向量行对齐）
  --id2od_file     <输出 id2od>      # 序重排映射，search 必需
  --M              <M>               # 构图参数，默认 64（与 --irange-M 一致）
  --N              <N>               # 库点数
  --ef_construction <efC>            # 默认 200（--irange-ef-construction）
  --threads        <T>               # 默认 16（--irange-build-threads）
```

**产物**：`--index_file` 索引目录/文件；**`--id2od_file`** 必须保留给查询阶段（内部有序域与原始 id 的映射）。若索引与 id2od 已存在且未 `--force-irange-build`，脚本会跳过建图。

### 7.4 查询：`search`

```text
search
  --data_path              <prepared.base_bin>
  --query_path             <prepared.query_bin>
  --range_saveprefix       <prepared.qrange_bin>       # 脚本传入完整路径，语义以 iRange 测试程序为准
  --groundtruth_saveprefix <prepared.gt_bin>
  --index_file             <与建图相同>
  --result_saveprefix      <输出 CSV 路径前缀>       # 如 .../csv/arxiv_irange_r_
  --id2od_file             <与建图相同>
  --attr_file              <prepared.attr_bin>
  --M                      <与建图相同>
  --N                      <prepared.n>
  --Nq                     <prepared.nq>
  --K                      <top-K>
  --ef_search              <逗号分隔 ef 列表>         # 一条命令扫一条 QPS–recall 曲线
```

**输出**：在 `result_saveprefix` 下生成 **`arxiv_irange_r_0.csv`**（脚本将前缀设为 `.../arxiv_irange_r_`）。`parse_irange_csv()` 按列解析：`ef, recall, qps, dco, hop, cmp`（至少 6 列），用于与 R-NSG 汇总表对齐。

### 7.5 与 R-NSG 对比时的注意点

- **ef 曲线**：iRange 通常一次 `search` 传入 **`--ef_search` 列表**，得到多点；R-NSG 侧常扫 `trunc` 或多组 query 参数，对比时在 Markdown/JSON 里按 **recall 阈值取 max QPS**（脚本已做）。
- **查询范围语义**：iRange 内部会对 range 做与有序属性一致的 **redirect**（见 `iRangeGraph/tests/search.cpp` 中逻辑）；本仓库的 `scripts/fannbench_irange_analyzer.py` 中 **`redirect_qranges()`** 用于离线复现该映射、分析有效范围宽度，**不等价于** R-NSG 的 JSON `qrange` 直接二分标签数组（实现路径不同，比 recall 时以各自 groundtruth 为准）。
- **线程**：脚本常设 `OMP_NUM_THREADS=1` 测 query；iRange `search` 本身以其实现为准（多为单线程热路径）。

### 7.6 相关脚本与离线分析

| 脚本 | 作用 |
|------|------|
| `arxiv_r_prepare_and_compare.py` | 准备资产、可选建 iRange 索引、跑 iRange 曲线 + R-NSG 曲线 |
| `arxiv_r_grid_compare.py` | 在已有 iRange 索引上跑 grid / 与 R-NSG 变体对比 |
| `run_rnsg_vs_irange_opt.py` | 同工作流中联跑 R-NSG 与 iRange（见脚本说明） |
| `fannbench_irange_analyzer.py` | **不修改 iRange 源码**，离线解析索引文件与检索 CSV、与 R-NSG 导出结果对照 |
| `irange_rnsg_deep_checker.py` | 深度一致性/曲线检验（对接时读其输入输出约定） |

深入理解 iRange **检索层**可选读外部仓库 `iRangeGraph/include/iRG_search.h`（层选择、`SelectEdge` 等）；本仓库不负责维护该代码。

---

## 8. Python 脚本怎么用（概念层）

- **网格 / 对比**：`scripts/arxiv_r_grid_compare.py`、`arxiv_r_prepare_and_compare.py` —— 批量生成命令、收集日志与 JSON；iRange 命令行见 **§7**。
- **检验**：`method_comp_checker.py`、`irange_rnsg_deep_checker.py` —— 与 iRange 或其它方法对齐时阅读其**输入输出格式**再对接。
- 运行前确认：`python3` 版本、依赖（numpy 等）、以及**二进制路径**指向当前 `build/.../enhanced_rnsg` 与 **`iRangeGraph` 的 buildindex/search**。

不要假设脚本内的绝对路径在所有机器上存在；优先通过环境变量或命令行参数覆盖。

---

## 9. 文档与算法

- **实现级流程、伪代码、参数表**：`doc/ALGORITHM_PSEUDOCODE.md`
- **用户提供的目录索引**（实验路径、数据路径）可作为检索线索；若与仓库不同步，以 `Worker` CLI 与脚本为准。

---

## 10. Revision 进度快照（2026-04-27）

本节记录当前 revision 实验状态，防止后续接手时丢失上下文；更细的算法语义见 `doc/ALGORITHM_PSEUDOCODE.md`。

### 10.1 新增查询调度器

- `scripts/run_rnsg_query_plan.py`：JSON 驱动的外部查询调度器。它按 `dataset + graph + job` 生成 `rnsg-new query_batch` 参数 CSV，每张图只加载一次数据和索引，再批量扫 `beam_size / trunc_size / qrange / gt`，最后合并为 `all_points.csv`。
- `doc/review/revision_experiments_2026-04-24/tools/rnsg_query_plan_large_trunc_sel0p5.json`：SIFT1M、selectivity=0.5、large-trunc 对比计划示例。
- 已验证命令：

```bash
python3 scripts/run_rnsg_query_plan.py \
  doc/review/revision_experiments_2026-04-24/tools/rnsg_query_plan_large_trunc_sel0p5.json \
  --only-graph expanded_cap64 \
  --dry-run \
  --out-dir /tmp/rnsg_query_plan_dryrun_check
```

### 10.2 当前关键图与构建成本

| tag | 图路径 | 构建时间 | 平均度 | 说明 |
|-----|--------|----------|--------|------|
| `orig_full` | `/mnt/win-dai/Vectors/rnsg_review_runs/sift1m/indexes/full_k300_rs1500_m200.graph` | 1660.79s | 116.06 | 较早版本默认 full 图；高 recall 查询最好，但构建慢 |
| `expanded_cap64` | `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_new_seed_collect_eval_sift1m_20260427_010135/expanded_cap64/expanded_cap64.graph` | 130.93s | 102.53 | `expanded + seed_search_beam_size=32 + seed_search_knng_cap=64 + knng_degree_cap=128`；构建很快，低/中 recall 查询略优 |
| iRangeGraph SIFT1M | `/home/zhouzhiqiu/code/FANNBench/FANNBench/buildtime.json` | 467.15s | N/A | 外部 baseline；构建约为 `expanded_cap64` 的 3.6 倍、约为 `orig_full` 的 0.28 倍 |

`orig_full` 历史构建日志中的关键选项包括：`range_step=1500`、KNNG `k=300`、`ef_max=200`、`reverse_refine=true`、`reverse_mode=incoming`、`seed_mode=beam`、`seed_keep=32`、`seed_expand=24`、`seed_knng_cap=0`、`knng_cap=0`、`prefix_policy=dist`。当前源码默认值已经变化，复现实验时必须以当次日志为准。

### 10.3 已完成的 SIFT1M 查询结论

`orig_full` 与旧 FANNBench RNSG 的可比点：SIFT1M 10% / 25% selectivity 上，当前 `rnsg-new orig_full` 在相同 recall 阈值下整体快于旧 RNSG。示例：10% selectivity 下，`>=0.99` recall 约 `2746 QPS`（当前）对 `1770 QPS`（旧）；25% selectivity 下，`>=0.995` recall 约 `1993 QPS`（当前）对 `1309 QPS`（旧）。50% selectivity 的旧 RNSG 曲线缺失，不能做严格历史对比。

SIFT1M selectivity=0.5、`k=10`、`query_batch` 网格：

| recall 阈值 | 当前最好结论 |
|-------------|--------------|
| `0.95` - `0.98` | `expanded_cap64` 略优，约 4%-5% QPS 优势 |
| `0.99` - `0.9995` | `orig_full` 更优；高 recall 下 `expanded_cap64` 需要更大 beam 或更宽 trunc，QPS 落后 |

主要产物：

- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_new_orig_vs_expanded_sel0p5_batch_20260427_021538/sel0p5_curve_report.md`
- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_new_orig_vs_expanded_sel0p5_batch_20260427_021538/sel0p5_all_points_batch.csv`

### 10.4 Seed collect 与 large-trunc 结论

Seed collect 实验目录：

- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_new_seed_collect_eval_sift1m_20260427_010135/report.md`
- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_new_seed_collect_eval_sift1m_20260427_010135/summary.csv`

已测配置中，`evaluated` 会收集大量候选并增加平均度，但没有带来查询收益；`expanded_cap64` 构建最快，`discovered_cap64/128` 与 `evaluated_cap*` 目前没有证明能替代 `orig_full` 的高 recall 曲线。

Large-trunc 探针目录：

- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_query_plan_large_trunc_sel0p5_20260427_025054/large_trunc_compare.md`
- `/mnt/win-dai/Vectors/rnsg_review_runs/analysis/rnsg_query_plan_large_trunc_sel0p5_20260427_025054/all_points.csv`

SIFT1M selectivity=0.5 下，`trunc=128` 相比 `trunc=70/100` 几乎没有 recall 收益，只明显降低 QPS。例如 `orig_full b160/t128` 为 `recall=0.99976, qps=615.99`，而既有 `b160/t70` 为 `recall=0.99974, qps=704.00`。当前判断：查询有效前缀大致落在 `50-100`，下一步更值得做 build-time 存储前缀压缩（优先 cap 70 / 100），而不是继续增大 query trunc。

### 10.5 保留与禁忌

- 不要删除上述 `.graph` 临时图，除非用户明确要求；这些图仍用于后续曲线与构图策略对比。
- 不要在本机重复处理 SpaceV100M 远程包；该长跑属于另一台大内存机器的问题域。
- baseline 必须通过 `$HOME/code/FANNBench` 的脚本/工具接口运行，不要绕过 FANNBench 自己手搓 baseline。

---

## 11. 快速索引（key → 关注点）

| key | 说明 |
|-----|------|
| `algorithm:enhanced_rnsg` | `src/main_enhanced_rnsg.cpp` + `EnhancedRNSG::` |
| `module:build` | `include/EnhancedRNSG/Builder.hpp` |
| `module:query` | `include/RNSG/Searcher.hpp` + `Graph::TDGraphIndexBase` |
| `module:cli` | `include/EnhancedRNSG/Worker.hpp` |
| `tool:analyzer` | `src/main_analyzer.cpp` |
| `baseline:irange` | 外部 `FANNBench/iRangeGraph`；调用见 **§7**、`arxiv_r_prepare_and_compare.py` |
| `scripts:paramsearch` | `scripts/*.py` 中含 `search_summary` 或 grid 的脚本 |

---

*若你新增长期约定（命名规范、默认数据集路径、iRange 二进制路径），建议只更新本段、`ALGORITHM_PSEUDOCODE.md` 中参数表与 **§7 iRange**，避免分散在多处 README。*
