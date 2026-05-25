# R-NSG / Enhanced R-NSG：实现级算法说明与伪代码

本文档面向算法设计与优化，描述**当前仓库实现**中的数据流、决策点与可调参数。约定：向量已按**范围标签单调重排**，节点下标 `i` 与标签顺序一致；距离为向量空间距离（实现中常用 L2 / 内积变体，以 `VectorList` 为准）。

**主实现位置**：`include/EnhancedRNSG/Builder.hpp`（建图）、`include/RNSG/Searcher.hpp`（查询 beam）、`include/EnhancedRNSG/Worker.hpp`（CLI 与选项绑定）。**精确轨迹分析**见 `src/main_analyzer.cpp`（非 `enhanced_rnsg query` 默认路径）。

---

## 1. 总体管线

### 1.1 输入 / 输出

| 符号 | 含义 |
|------|------|
| `N, D` | 库大小、维度 |
| `label[i]` | 节点 `i` 的范围属性（uint64），建图前会生成**按 label 排序的置换**并重排数据 |
| `KNNG` | 外部或 `knng` 子命令生成的近似 KNN 图（每行邻居已映射到重排后下标） |
| `ef_max` | 每个节点**最终出度上界**（经剪枝 + 可选 reserve + reorder 后仍可能受实现细节影响） |
| `range_step` | 标签轴上的**局部窗口**半径（与 `enable_range_augmentation` 联用） |

**输出**：`Graph::TDGraphIndexBase`——带 header（质心、有序标签等）的邻接表；查询阶段再按 `qrange` 取子图或做 range 过滤。

### 1.2 建图阶段顺序（与代码一致）

1. **预处理**：求质心 `center`；`label → index, pos`；**重排数据集**使下标沿 label 单调。
2. **父链**：对每个 `i` 计算 `centroid_dist[i] = dist(i, center)`，在数组上做单调栈得到 `left_parent[i]`、`right_parent[i]`（“左右更小邻居”链，用于 seed）。
3. **KNNG 映射**：`sorted_knng[i]` = 重排后节点 `i` 的 KNN 邻居表（可 `knng_degree_cap` 截断每行度数）。
4. **第一遍（并行 per-node）**：`core_candidates` ← KNNG 行 + 可选 label 窗口；`bridge_candidates` ← **centroid 单调链 seed** + KNNG 上 pq/beam 扩展；合并 → **三角剪枝**（`prune_candidates` 或 quota 合并）→ 可选 **role 选择** 或 **witness / support / tail reserve** → `reorder_prefix_edges`。
5. **反向精炼（可选）**：`enable_reverse_refine`。先构建全局 `incoming[v]` = `{ u \| v ∈ first_adj[u] }`。第二遍对每个 `i` 再收集候选并剪枝，**覆盖** `final_adj[i]`（不是简单补一条反向边）。`incoming` / `full` 两种模式见 §3.5。
6. **落盘**：`init_header` + `add_neighbours`。

### 1.3 查询阶段顺序

1. 查询向量与库使用同一度量；库保持**重排后下标**，结果写出前乘 `ord` 映回原 id（见 `Worker::query`）。
2. 对每个 query，由 `qrange` 在**有序标签数组**上二分得到 `[range_l, range_r)`。
3. **`range_scan_mode`**：`subgraph` 在**诱导子图**上 beam；`direct` 在**全图邻接**上扩展但邻居需满足 label ∈ range。
4. **种子**：header 给出的入口节点列表（或 fallback 下标）。
5. **`beam_search` / `beam_search_range`**：维护长度为 `beam_size` 的**有序**候选（距查询点最近在前）；对 beam 中**每一个槽位**各做一轮“扩展当前节点 → 从邻接取子集 → 算距 → 尝试插入 top beam”；含 **nav 限宽**、**pick 多扫**、**early stop** 等。

---

## 2. 核心数据结构（实现语义）

- **`sorted_knng[i]`**：无向 KNN 邻接的近似 oracle；扩展时可用 `seed_search_knng_cap` 只读每行前 `cap` 个邻居。
- **`seen` / `bridge_support`**：第一遍中为去重；`bridge_support[v]` 记录节点 `v` 作为 bridge 被不同 seed 路径“见证”的次数，供 support reserve / role 使用。
- **Beam 查询中的 `candidates`**：长度固定为 `beam_size` 的数组，存 `(distance, node_id)`，**升序**；`node_id` 在实现里用 `+dataset.size()` 标记是否已访问等（见 `Searcher.hpp`）。
- **`BeamScratch`**：每查询清空 epoch，记录 `visited_nodes` 供可选导出。

---

## 3. 建图：分步伪代码

### 3.1 标签序与父链

```text
procedure PREPROCESS(V, label):
    (index, pos) ← ORDER_BY_LABEL(label)
    REORDER_DATASET(V, index)
    sorted_label ← SORT(label)
    center ← MEAN(V)
    for i in 0..N-1:
        centroid_dist[i] ← dist(V[i], center)
    left_parent  ← BUILD_PREV_SMALLER(centroid_dist)   // 单调栈，左侧最近更小下标
    right_parent ← BUILD_NEXT_SMALLER(centroid_dist) // 右侧最近更小下标
```

### 3.2 局部标签窗口（core 增强）

当 `enable_range_augmentation` 且 `range_step > 0`：

```text
procedure ADD_LOCAL_WINDOW(src, dst, range_step, N):
    for j from max(0, src - range_step) to src-1:
        dst.push(j)
    for j from src+1 to min(N-1, src + range_step):
        dst.push(j)
```

### 3.3 Centroid 单调链 + KNNG 上 seed 收集

实现函数：`collect_seed_chain` + `collect_from_seed_batch_pq` / `collect_from_seed_batch_beam`。

```text
procedure COLLECT_SEED_CHAIN(src, left_side, seen, bridge_dst, support):
    if not enable_centroid_seed_search: return

    cur ← (left_side ? (src>0 ? src-1 : -1) : (src+1 < N ? src+1 : -1))
    batch_size ← max(1, seed_batch_size)
    chain ← FOLLOW_PARENT_CHAIN(cur, left_parent/right_parent)
    if monotone_seed_policy == "far" and monotone_seed_limit > 0:
        chain ← farthest monotone_seed_limit nodes from chain
    else if monotone_seed_limit > 0:
        chain ← nearest monotone_seed_limit nodes from chain

    for seed_batch in BATCH(chain, batch_size):
        for seed in seed_batch:
            support[seed]++   // 若 support 非空
            if seen.insert(seed): bridge_dst.push(seed)

        scaled_keep   ← (seed_collect_keep == 0 ? 0 : seed_collect_keep * |seed_batch|)
        scaled_expand ← (seed_collect_max_expand == 0 ? 0 : seed_collect_max_expand * |seed_batch|)

        if seed_collect_mode == "pq":
            expanded ← COLLECT_FROM_SEED_BATCH_PQ(src, seed_batch, scaled_keep, scaled_expand,
                                                  seed_search_knng_cap, seed_collect_policy)
        else:
            expanded ← COLLECT_FROM_SEED_BATCH_BEAM(src, seed_batch, scaled_keep, scaled_expand,
                                                    seed_search_beam_size, seed_search_knng_cap,
                                                    seed_collect_policy)

        for v in expanded:
            if v == src: continue
            support[v]++
            if seen.insert(v): bridge_dst.push(v)
```

**PQ 模式**（`collect_from_seed_batch_pq`）：最小堆按 `dist(target, ·)`；反复弹出当前最近未扩展节点，将其 KNNG 邻居入堆，直到扩展次数或收集数量达到上限。

**Beam 模式**（`collect_from_seed_batch_beam`）：维护按距离排序的**候选列表**；每次取出下一个最近节点并扩展其 KNN 邻居，将新点按距离**插入有序表**（同层 beam 宽语义由“有序列表 + 依次 pop”实现）。`seed_search_beam_size` 限制 seed search frontier 宽度；`0` 表示跟随 `seed_collect_max_expand`，两者都为 `0` 时实现用 1024 作为内部上限。

**`seed_collect_policy` 返回语义**：

- `expanded`：只把真正弹出并展开过的节点作为 bridge 候选。
- `discovered`：把 seed search 中插入候选队列的节点作为 bridge 候选。
- `evaluated`：把 seed search 中被计算过 `dist(src, ·)` 的 KNNG 邻居也作为 bridge 候选；候选量最大，可能显著增大平均度。

**预算语义**：

- `seed_collect_keep=0`：不再额外截断由 `seed_collect_policy` 选出的候选。
- `seed_collect_max_expand=0`：seed search 跑到 frontier 耗尽或被 beam 内部上限限制。
- `seed_search_knng_cap=0`：seed search 展开时读取完整 KNNG 行；否则只读每行前 `cap` 个邻居。
- `knng_degree_cap=0`：加载/使用 KNNG 时不做全局行度截断；非零值会截断 KNNG 行，影响 core 与 seed search 可见候选。

### 3.4 第一遍 per-node 合并与剪枝

```text
procedure BUILD_FIRST_PASS_NODE(i):
    seen ← empty
    core ← KNNG_ROW(i) 去重、去自环
    ADD_LOCAL_WINDOW(i, core, ...)

    bridge ← empty
    bridge_support ← empty map
    COLLECT_SEED_CHAIN(i, true,  seen, bridge, bridge_support)
    COLLECT_SEED_CHAIN(i, false, seen, bridge, bridge_support)

    prune_budget ← (role_select ? ef_max + role_pool_extra : ef_max)

    if candidate_merge_mode == "quota":
        pruned ← SELECT_QUOTA_CANDIDATES(i, core, bridge, prune_budget, side_split, core_ratio)
    else:
        merged ← core ∪ bridge
        pruned ← PRUNE_CANDIDATES(i, merged, prune_budget, side_split)  // 见下
```

**`PRUNE_CANDIDATES`（legacy）**（对应 `detail::prune_candidates`）：

1. 将候选按相对 `src` 分为左侧（下标 `< src`）与右侧（`> src`）。
2. 左侧按下标降序、右侧升序去重（保持与 RNSG 一致的“标签邻近”顺序）。
3. 计算到 `src` 的距离，并加上 `||V[src]||²` 项用于与 `RNSG::prune` 的三角检验一致。
4. 若 `side_split_pruning`：左右各调用 `RNSG::prune` 再合并到 `ef_max`；否则合并后一次 `RNSG::prune(..., ef_max)`。

**`SELECT_QUOTA_CANDIDATES`**：先在 core / bridge 桶内用 `check_valid`（三角不等式筛选）按 `core_ratio` 分配额度，再填满 `ef_max`。

### 3.5 后处理：reorder、witness、tail

非 role 路径下典型顺序：

```text
    APPEND_SUPPORT_RESERVE(pruned, bridge_support, ...)   // 策略 dist 或 bridgegap
    APPEND_BRIDGE_WITNESS(pruned, bridge_candidates, ...)   // 按 |index gap| 的对数分桶均衡抽样
    REORDER_FINAL(pruned)   // RNSG::reorder_prefix_edges：prefix_policy 等
    APPEND_TAIL_RESERVE(pruned, core, bridge, tail_reserve)
```

`REORDER_FINAL` 会按距离排序后调用 `reorder_prefix_edges`（`dist|mix|score|...` 等 CLI `--prefix_policy`）。

### 3.6 反向精炼（第二遍）

关闭方式：`--disable_reverse_refine`（实现里没有名为 `none` 的 mode，而是 flag）。

```text
procedure REVERSE_REFINE(first_adj):
    incoming ← 转置(first_adj)   // incoming[v] = { u | v ∈ first_adj[u] }

    for each node i in parallel:
        seen ← empty
        core ← copy(first_adj[i])
        if reverse_refine_mode == "full":
            core ∪= sorted_knng[i]
            ADD_LOCAL_WINDOW(i, core, ...)

        bridge ← empty
        if reverse_incoming_quota > 0:
            bridge ← SELECT_QUOTA(incoming[i], policy dist|bridgegap, reverse_incoming_quota)
            // 并写入 reverse_support
        else:
            bridge ← incoming[i]

        merged / pruned 同第一遍
        // role 或 witness / tail 路径与第一遍对称
        final_adj[i] ← pruned
```

**语义要点**：第二遍会**重新剪枝整张出边表**，`incoming` 模式仅表示 core 不再包含“全量重开 KNNG+窗口”，但 bridge 仍携带反向入边候选；`full` 模式与第一遍收集力度更接近，成本高。

---

## 4. 查询：beam 伪代码

与 `RNSG::Searcher::beam_search_range` 一致（range 版仅多 `range_l ≤ id ≤ range_r` 过滤）。

```text
procedure BEAM_SEARCH_RANGE(query, k, seeds, beam_size, trunc_size, range_l, range_r):
    candidates ← SORT_BY_DIST(query, seeds) ∩ range
    MARK_VISITED(candidates)
    RESIZE candidates to beam_size (padding 远距离占位)

    stall_rounds ← 0
    for uid from 0 to beam_size-1:
        if slot uid empty: continue
        current ← candidates[uid].node

        local_budget ← NAV_WIDTH(uid, trunc_size, nav_degree, nav_tail_degree, stall_rounds, ...)
        scan_limit   ← max(trunc_size * pick_scan_factor, nav_scan_limit(...))

        raw ← first scan_limit unvisited neighbors from graph(current) in adjacency order, range-filtered
        neighbours ← if |raw| ≤ local_budget then raw
                     else SELECT_FROM_RAW(current, raw, local_budget, pick_front_keep, edge_pick_policy, ...)

        for each (dist, nbr) in DIST(query, neighbours):
            MARK_VISITED(nbr)
            if dist improves worst of beam:
                POP worst, INSERT (dist, nbr) sorted, possibly rewind uid
                improved ← true

        stall_rounds ← improved ? 0 : stall_rounds+1
        if early_stop triggered: break

    return TOP_K_BY_DISTANCE(candidates, k)
```

**策略要点（与实现对齐）**：

- **nav**：`nav_degree < trunc_size` 时先用窄扇出，若连续 `nav_stall_rounds` 轮无改进则放宽到 `trunc_size`；后半 beam 槽位可再被 `nav_tail_degree` 限制。
- **pick**：`pick_scan_factor > 1` 时先沿邻接表多读候选，再 `select_from_raw`（`prefix|side|reciprocal|corebridge`）。
- **自适应宽查询**：`nav_width_split` + `nav_degree_wide` 在范围宽度较大时切换更激进 nav。

---

## 5. 参数索引（建图）

| 参数 / CLI | 作用 |
|------------|------|
| `ef_max` | 最终度数主预算 |
| `range_step` + `disable_range_augmentation` | 是否注入标签局部窗 |
| `disable_side_split_pruning` | 单侧三角剪枝 vs 合并剪枝 |
| `candidate_merge_mode` / `core_ratio` | legacy 合并 vs 分桶 quota |
| `disable_centroid_seed_search` / `monotone_seed_policy` / `monotone_seed_limit` | 是否走父链、取近端还是远端 seed、每侧步数上限；`monotone_seed_limit=0` 表示沿 full chain 收集所有起点 |
| `seed_collect_mode` pq/beam | KNNG 上扩展方式 |
| `seed_collect_policy` expanded\|discovered\|evaluated | seed search 中哪些节点进入 bridge 候选；见 §3.3 |
| `seed_collect_keep` / `seed_collect_max_expand` | 每批 seed 的收集与扩展预算（批量时按批大小缩放）；`0` 分别表示不截断收集结果 / 展开到 frontier 耗尽 |
| `seed_batch_size` | 沿父链批处理 seed |
| `seed_search_beam_size` | beam 模式下 seed search 的 frontier 宽度 |
| `seed_search_knng_cap` / `knng_degree_cap` | seed search 行扫描截断 / KNNG 全局行度截断 |
| `disable_reverse_refine` / `reverse_refine_mode` full\|incoming | 第二遍是否启用及 core 是否重开 KNNG+窗 |
| `reverse_incoming_quota` / `reverse_incoming_policy` | 反向边候选压缩 |
| `bridge_witness_reserve` / `support_reserve*` / `tail_reserve` | 剪枝后追加连通/见证边 |
| `role_select_policy` + `role_*` | 建图期按 gap/support 角色选边 |
| `prefix_policy` + 相关 | `reorder_prefix_edges` 边序 |
| `profile_build_json` | 输出各阶段耗时与候选统计 JSON |

---

## 6. 参数索引（查询）

| 参数 | 作用 |
|------|------|
| `beam_size` | beam 宽度 |
| `trunc_size` | 每槽位最多考虑的邻居数（与 nav 组合） |
| `nav_degree`, `nav_scan_factor`, `nav_stall_rounds` | 导航扇出与回落 |
| `nav_front_keep`, `nav_tail_degree`, `nav_early_stop_rounds` | 前缀保留、后半收紧、早停 |
| `nav_width_split`, `nav_degree_wide` | 宽范围时切换 nav |
| `pick_scan_factor`, `pick_front_keep` | 多扫邻接与前缀保留 |
| `edge_pick_policy`, `edge_pick_recip_depth`, `edge_pick_core_ratio` | 从 raw 中选入 neighbours 的规则 |
| `range_scan_mode` subgraph\|direct | 子图 vs 全图+range 检查 |

---

## 7. 与“通用文档”的差异（避免误读）

1. **单调链**：实现是 **centroid 距离数组上的单调栈父链** + 从 `src±1` 起步，不是任意“时间序列”专用抽象。
2. **反向精炼**：第二遍是**整表重剪枝**，不是仅“若缺边则 add `u→v`”。
3. **剪枝**：主机制是 **RNSG 三角剪枝 + ef 截断**，不是独立的 `keep_ratio` 三选一字符串（若脚本层有类似语义，需对照脚本）。
4. **exact_trace**：在 `main_analyzer` 工具链中；默认 `enhanced_rnsg query` 走并行 beam，不保证逐步可复现日志。

---

## 8. Revision 经验记录（2026-04-27）

这些结论来自 SIFT1M 本地实验，属于当前实现的经验约束，不能替代跨数据集最终结论。

1. **`expanded_cap64` 与 `orig_full` 的定位不同**：`expanded_cap64`（`seed_collect_policy=expanded`、`seed_search_beam_size=32`、`seed_search_knng_cap=64`、`knng_degree_cap=128`）构建约 130.93s，平均度约 102.53，低/中 recall 略快；历史 `orig_full` 构建约 1660.79s，平均度约 116.06，高 recall 曲线更强。
2. **`evaluated` 不等于更好**：`evaluated_cap*` 会显著增大候选与平均度，但当前 SIFT1M 测试没有带来查询曲线优势；它更像候选上限验证，不应作为默认策略。
3. **large trunc 收益很低**：selectivity=0.5 下，`trunc=128` 相比 `trunc=70/100` 几乎不提升 recall，却降低 QPS。后续优先验证 build-time 前缀压缩（候选 cap 70 / 100），而不是继续扩大查询 trunc。
4. **边序优化的主要瓶颈仍是距离计算次数**：此前对外层 range 过滤、二分/扫描等热路径优化收益有限；曲线主要随 `beam_size × 有效邻居数` 的距离计算量变化。

---

## 9. 优化时可优先动刀的位置（实现锚点）

| 目标 | 建议阅读 |
|------|----------|
| 降低建图时间 | `collect_seed_chain` 批处理与 `seed_*` cap；`reverse_refine_mode=incoming`；减小 `ef_max` |
| 改善连通 / 召回 | `bridge_witness_reserve` / `tail_reserve` / `full` 反向；`core_ratio` 与 prefix 策略 |
| 查询 QPS | `beam_search_range` 热循环、`nav_*` 早停、`range_scan_mode`、子图构建开销 `index(...)` |
| 可重复实验 | `main_analyzer.cpp` 与 groundtruth 格式、`Worker::query` 输出映射 `ord` |

---

*文档版本与仓库：`include/EnhancedRNSG/Builder.hpp`, `include/RNSG/Searcher.hpp`, `include/EnhancedRNSG/Worker.hpp`。*
