# R-NSG

This repository contains the RNSG implementation used for range-filtered
approximate nearest-neighbor search. The maintained executable for the latest
revision experiments is `rnsg-new`; it provides index construction, querying,
batch querying, KNNG construction, and brute-force ground-truth generation.

## Dependencies and Compilation

This project is built using [**xmake**](https://xmake.io/#/). xmake will
automatically download and manage all required dependencies, but you need to
install xmake beforehand.

> **Compiler requirement:**
> The code uses **C++20** features. Please ensure your compiler supports C++20
> (for example, GCC >= 11, Clang >= 13, or MSVC >= 19.30).

After installing xmake, run the following commands in the project root:

```bash
xmake f -m release
xmake -r rnsg-new
```

Optional targets used by the paper experiments:

```bash
xmake -r rnsg-2d-fast    # 2D/multi-label range filtering prototype
xmake -r TDFANN_ablation # construction ablation target
```

The release build enables `-O3`, `-march=native`, and LTO for the final RNSG
targets. For reproducible performance measurements, build on the same CPU
family on which the binary will run.


## Generic Curve Runner

For reproducibility, the recommended public interface is the generic curve
runner. A user only needs to provide the vector files, label/range files, a run
tag, and optionally ground truth:

```bash
python3 scripts/run_rnsg_new_curve.py all \
  --tag sift1m_sel10 \
  --dataset-file /path/to/base.fvecs \
  --query-file /path/to/query.fvecs \
  --label-file /path/to/labels.json \
  --qrange-file sel10=/path/to/qrange_sel10.json \
  --groundtruth-file sel10=/path/to/gt_sel10_k10.json \
  --threads 32
```

If `--groundtruth-file` is omitted, the runner generates exact ground truth with
`rnsg-new groundtruth` and then runs the curve. Use `--qps-only` to skip recall
calculation and measure QPS only.

For convenience, `--base-file`, `--range-file`, and `--gt-file` are aliases for
`--dataset-file`, `--qrange-file`, and `--groundtruth-file`.

Multiple workloads can be queried in one run by repeating `--qrange-file` and,
when available, `--groundtruth-file` with matching labels:

```bash
python3 scripts/run_rnsg_new_curve.py all \
  --tag sift1m_main \
  --dataset-file /path/to/base.fvecs \
  --query-file /path/to/query.fvecs \
  --label-file /path/to/labels.json \
  --qrange-file sel1=/path/to/qrange_sel1.json \
  --qrange-file sel10=/path/to/qrange_sel10.json \
  --qrange-file sel50=/path/to/qrange_sel50.json \
  --groundtruth-file sel1=/path/to/gt_sel1_k10.json \
  --groundtruth-file sel10=/path/to/gt_sel10_k10.json \
  --groundtruth-file sel50=/path/to/gt_sel50_k10.json \
  --threads 32
```

Outputs are written to `artifacts/rnsg_new_curves/<tag>/` by default:

```text
rnsg_new_curve.csv          # all recall/QPS curve points
rnsg_new_curve_pareto.csv   # non-dominated recall/QPS points
rnsg_new_curve_pareto.md    # compact Markdown table for quick inspection
query_batch_params.csv      # exact beam/trunc/query settings
run_manifest.json           # input paths, build/query parameters, commands
results/*.json              # per-setting neighbor results
```

After the query stage, the runner also prints the Pareto frontier key points,
for example:

```text
Pareto frontier key points:
  [sel10pct] 2 point(s)
    recall=0.9930 qps=28296.96 topk=10 beam=24 trunc=8 run=sel10pct_b24_t8
    recall=1.0000 qps=16595.97 topk=10 beam=48 trunc=16 run=sel10pct_b48_t16
```

The default construction uses the latest lightweight no-seed path:

```bash
--knng-k 128 --range-step 1500 --ef-max 200 --knng-degree-cap 128 \
--seed-search-knng-cap 32 --reverse-refine-mode incoming
```

The default query curve disables the experimental navigation policy and sweeps
only `beam_size`/`trunc_size`. Override it with `--query-grid`, for example:

```bash
--query-grid 24:8,32:12,48:16,64:24,96:32,128:48,192:80
```

## Quick Demo: SIFTsmall

The quickest end-to-end check is the SIFTsmall demo wrapper:

```bash
python3 scripts/run_siftsmall_rnsg_new_demo.py all --threads 8
```

This command will:

* compile `rnsg-new`;
* find SIFTsmall under `/mnt/win-dai/Vectors/siftsmall`, or download it from
  the TexMex mirror if it is not available locally;
* generate deterministic synthetic labels and query ranges;
* build a KNNG and an RNSG index using the latest no-seed construction path;
* generate brute-force ground truth;
* run a small beam/trunc query sweep through `rnsg-new query_batch`.

Generated files are kept out of Git by default:

```text
dataset/siftsmall_demo/
artifacts/siftsmall_rnsg_new_demo/
```

The final query outputs are:

```text
artifacts/siftsmall_rnsg_new_demo/query_batch_results.csv
artifacts/siftsmall_rnsg_new_demo/query_batch_results_pareto.csv
artifacts/siftsmall_rnsg_new_demo/query_batch_results_pareto.md
```

The demo prints the Pareto frontier key points directly in the terminal, so a
first run can be checked without plotting anything.

To inspect commands without running them:

```bash
python3 scripts/run_siftsmall_rnsg_new_demo.py all --dry-run
```

To run the stages manually:

```bash
python3 scripts/run_siftsmall_rnsg_new_demo.py prepare --threads 8
python3 scripts/run_siftsmall_rnsg_new_demo.py knng --threads 8
python3 scripts/run_siftsmall_rnsg_new_demo.py build --threads 8
python3 scripts/run_siftsmall_rnsg_new_demo.py groundtruth --threads 8
python3 scripts/run_siftsmall_rnsg_new_demo.py query --threads 8
```

Useful demo options:

```bash
python3 scripts/run_siftsmall_rnsg_new_demo.py all \
  --source-data-dir /path/to/siftsmall \
  --selectivities 0.01,0.1,0.5 \
  --query-grid 24:8,32:12,48:16,64:24,96:32,128:48 \
  --threads 8
```

The demo intentionally uses smaller construction settings than the 1M-scale
paper runs so that it finishes quickly. The default paper-style 1M construction
settings are typically:

```bash
--range_step 1500 --ef_max 200 --knng_degree_cap 128 \
--disable_centroid_seed_search --reverse_refine_mode incoming
```

## Command-Line Interface

`rnsg-new` exposes the following subcommands:

```bash
./build/linux/x86_64/release/rnsg-new knng --help
./build/linux/x86_64/release/rnsg-new build --help
./build/linux/x86_64/release/rnsg-new groundtruth --help
./build/linux/x86_64/release/rnsg-new query --help
./build/linux/x86_64/release/rnsg-new query_batch --help
```

Minimal manual workflow:

```bash
BIN=./build/linux/x86_64/release/rnsg-new

$BIN knng \
  --dataset_file dataset.fvecs \
  -k 128 \
  --graph_file dataset_k128.knng.graph \
  --threads 32

$BIN build \
  --dataset_file dataset.fvecs \
  --knng_file dataset_k128.knng.graph \
  --label_file labels.json \
  --index_file rnsg.graph \
  --range_step 1500 \
  --ef_max 200 \
  --disable_centroid_seed_search \
  --reverse_refine_mode incoming \
  --knng_degree_cap 128 \
  --threads 32

$BIN groundtruth \
  --dataset_file dataset.fvecs \
  --query_file query.fvecs \
  --label_file labels.json \
  --qrange_file ranges.json \
  --qnumber 10 \
  --result_file gt.json \
  --threads 32

$BIN query \
  --dataset_file dataset.fvecs \
  --index_file rnsg.graph \
  --query_file query.fvecs \
  --label_file labels.json \
  --qrange_file ranges.json \
  --groundtruth_file gt.json \
  --qnumber 10 \
  --beam_size 80 \
  --trunc_size 24 \
  --result_file result.json \
  --nav_degree 0 \
  --threads 32
```

For query sweeps, prefer `query_batch`; it loads the dataset and graph once and
then runs multiple query settings from a CSV file. The CSV header is:

```text
run_id,selectivity,selectivity_label,topk,beam_size,trunc_size,qrange_file,groundtruth_file,result_file
```

## Input Formats

Supported vector files:

* `.fvecs` for float vectors;
* `.u8bin` for unsigned 8-bit vectors loaded as float vectors;
* `.i8bin` for signed 8-bit vectors with SimSIMD-backed int8 distance kernels.

Labels and query ranges are JSON arrays:

* `labels.json`: one integer label per base vector;
* `ranges.json`: a flat array `[l0, r0, l1, r1, ...]`, one closed range per
  query.

## Repository Hygiene

Large datasets, generated graphs, experiment logs, revision reports, and paper
plot tables are ignored by `.gitignore`. Keep reproducible source code, build
configuration, and compact examples in Git; keep generated experiment artifacts
under `dataset/`, `artifacts/`, or external storage such as `/mnt/win-dai/Vectors`.
