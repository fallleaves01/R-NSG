#!/usr/bin/env python3
"""Generic rnsg-new recall/QPS curve runner.

The intended public interface is data-oriented: provide vector files, labels,
query ranges, a tag, and optionally ground-truth files. The script builds or
reuses the KNNG/index, generates missing ground truth unless --qps-only is set,
and runs a query_batch beam/trunc sweep to produce a curve CSV.

Example:

  python3 scripts/run_rnsg_new_curve.py all \
    --tag sift1m_sel10 \
    --dataset-file /path/sift_base.fvecs \
    --query-file /path/sift_query.fvecs \
    --label-file /path/labels.json \
    --qrange-file sel10=/path/qrange_sel10.json \
    --groundtruth-file sel10=/path/gt_sel10_k10.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rnsg_curve_utils import summarize_curve_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "build/linux/x86_64/release/rnsg-new"
DEFAULT_OUT_ROOT = REPO_ROOT / "artifacts/rnsg_new_curves"
DEFAULT_GRID = "24:8,32:12,48:16,64:24,80:32,96:40,128:48,160:64,192:80,256:96"


@dataclass(frozen=True)
class RangeSpec:
    label: str
    path: Path
    selectivity: str


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path
    knng_file: Path
    index_file: Path
    batch_params: Path
    batch_csv: Path
    result_dir: Path
    manifest_file: Path
    build_profile: Path


def shell_join(cmd: list[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd: list[str | os.PathLike[str]], *, dry_run: bool, manifest: list[str]) -> None:
    rendered = shell_join(cmd)
    manifest.append(rendered)
    print(f"$ {rendered}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def parse_grid(raw: str) -> list[tuple[int, int]]:
    grid: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"invalid grid item {item!r}; expected beam:trunc")
        beam_s, trunc_s = item.split(":", 1)
        beam, trunc = int(beam_s), int(trunc_s)
        if beam <= 0 or trunc <= 0:
            raise argparse.ArgumentTypeError("beam/trunc values must be positive")
        grid.append((beam, trunc))
    if not grid:
        raise argparse.ArgumentTypeError("query grid cannot be empty")
    return grid


def clean_label(raw: str) -> str:
    label = raw.strip()
    if not label:
        raise argparse.ArgumentTypeError("empty label")
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    return label.strip("_") or "range"


def infer_selectivity(label: str) -> str:
    low = label.lower()
    m = re.search(r"([0-9]+(?:[._p][0-9]+)?)\s*pct", low)
    if m:
        return str(float(m.group(1).replace("p", ".").replace("_", ".")) / 100.0)
    m = re.search(r"sel(?:ectivity)?[_-]?([0-9]+(?:[._p][0-9]+)?)", low)
    if m:
        value = float(m.group(1).replace("p", ".").replace("_", "."))
        if value > 1.0:
            value /= 100.0
        return str(value)
    return ""


def split_labeled_path(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return clean_label(label), Path(path).expanduser()
    if ":" in value and not value.startswith("/"):
        label, path = value.split(":", 1)
        return clean_label(label), Path(path).expanduser()
    path = Path(value).expanduser()
    return None, path


def parse_qranges(values: list[str]) -> list[RangeSpec]:
    specs: list[RangeSpec] = []
    seen: set[str] = set()
    for raw in values:
        label, path = split_labeled_path(raw)
        if label is None:
            label = clean_label(path.stem)
        if label in seen:
            raise argparse.ArgumentTypeError(f"duplicate qrange label: {label}")
        seen.add(label)
        specs.append(RangeSpec(label=label, path=path.expanduser().resolve(), selectivity=infer_selectivity(label)))
    return specs


def parse_groundtruth(values: list[str], ranges: list[RangeSpec]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not values:
        return out
    range_labels = {r.label for r in ranges}
    if len(values) == 1 and len(ranges) == 1 and "=" not in values[0] and ":" not in values[0]:
        out[ranges[0].label] = Path(values[0]).expanduser().resolve()
        return out
    for raw in values:
        label, path = split_labeled_path(raw)
        if label is None:
            label = clean_label(path.stem)
        if label not in range_labels:
            raise argparse.ArgumentTypeError(
                f"groundtruth label {label!r} does not match any qrange label {sorted(range_labels)}"
            )
        out[label] = path.expanduser().resolve()
    return out


def ensure_inputs(args: argparse.Namespace, ranges: list[RangeSpec], gt: dict[str, Path]) -> None:
    if args.dry_run:
        return
    required = [args.dataset_file, args.query_file, args.label_file] + [r.path for r in ranges]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.qps_only:
        for path in gt.values():
            if not path.exists():
                raise FileNotFoundError(path)


def make_paths(args: argparse.Namespace) -> RunPaths:
    out_dir = (args.out_dir or (args.out_root / args.tag)).resolve()
    knng_file = args.knng_file or out_dir / f"{args.tag}_k{args.knng_k}.knng.graph"
    index_file = (
        args.index_file
        or out_dir / f"{args.tag}_rnsg_new_k{args.knng_k}_rs{args.range_step}_m{args.ef_max}.graph"
    )
    return RunPaths(
        out_dir=out_dir,
        knng_file=knng_file.resolve(),
        index_file=index_file.resolve(),
        batch_params=(out_dir / "query_batch_params.csv").resolve(),
        batch_csv=(out_dir / "rnsg_new_curve.csv").resolve(),
        result_dir=(out_dir / "results").resolve(),
        manifest_file=(out_dir / "run_manifest.json").resolve(),
        build_profile=(out_dir / "build_profile.json").resolve(),
    )


def auto_gt_path(paths: RunPaths, label: str, topk: int) -> Path:
    return paths.out_dir / f"groundtruth_{label}_k{topk}.json"


def maybe_build_knng(args: argparse.Namespace, paths: RunPaths, manifest: list[str]) -> None:
    if args.stage not in {"all", "knng"}:
        return
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    if (
        args.stage == "all"
        and paths.index_file.exists()
        and not args.force_index
        and not args.force_knng
    ):
        print(f"reuse index: {paths.index_file}; skip KNNG build")
        return
    if paths.knng_file.exists() and not args.force_knng:
        print(f"reuse KNNG: {paths.knng_file}")
        return
    run_cmd(
        [
            args.binary,
            "--verbose",
            "knng",
            "--dataset_file",
            args.dataset_file,
            "-k",
            args.knng_k,
            "--graph_file",
            paths.knng_file,
            "--threads",
            args.threads,
        ],
        dry_run=args.dry_run,
        manifest=manifest,
    )


def maybe_build_index(args: argparse.Namespace, paths: RunPaths, manifest: list[str]) -> None:
    if args.stage not in {"all", "build"}:
        return
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    if paths.index_file.exists() and not args.force_index:
        print(f"reuse index: {paths.index_file}")
        return
    cmd: list[str | os.PathLike[str]] = [
        args.binary,
        "--verbose",
        "build",
        "--range_step",
        args.range_step,
        "--dataset_file",
        args.dataset_file,
        "--knng_file",
        paths.knng_file,
        "--label_file",
        args.label_file,
        "--index_file",
        paths.index_file,
        "--ef_max",
        args.ef_max,
        "--reverse_refine_mode",
        args.reverse_refine_mode,
        "--seed_search_knng_cap",
        args.seed_search_knng_cap,
        "--knng_degree_cap",
        args.knng_degree_cap,
        "--profile_build_json",
        paths.build_profile,
        "--threads",
        args.threads,
    ]
    if args.no_seed:
        cmd.append("--disable_centroid_seed_search")
    if args.range_window_cap is not None:
        cmd.extend(["--range_window_cap", args.range_window_cap])
    run_cmd(cmd, dry_run=args.dry_run, manifest=manifest)


def maybe_groundtruth(
    args: argparse.Namespace,
    paths: RunPaths,
    ranges: list[RangeSpec],
    gt: dict[str, Path],
    manifest: list[str],
) -> dict[str, Path]:
    resolved = dict(gt)
    if args.qps_only:
        return resolved
    if args.stage not in {"all", "groundtruth", "query"}:
        return resolved
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in ranges:
        if spec.label not in resolved:
            resolved[spec.label] = auto_gt_path(paths, spec.label, args.topk)
        if resolved[spec.label].exists() and not args.force_groundtruth:
            print(f"reuse groundtruth[{spec.label}]: {resolved[spec.label]}")
            continue
        run_cmd(
            [
                args.binary,
                "--verbose",
                "groundtruth",
                "--dataset_file",
                args.dataset_file,
                "--query_file",
                args.query_file,
                "--label_file",
                args.label_file,
                "--qrange_file",
                spec.path,
                "--qnumber",
                args.topk,
                "--result_file",
                resolved[spec.label],
                "--threads",
                args.threads,
            ],
            dry_run=args.dry_run,
            manifest=manifest,
        )
    return resolved


def write_batch_params(
    args: argparse.Namespace,
    paths: RunPaths,
    ranges: list[RangeSpec],
    gt: dict[str, Path],
) -> None:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.result_dir.mkdir(parents=True, exist_ok=True)
    with paths.batch_params.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(
            [
                "run_id",
                "selectivity",
                "selectivity_label",
                "topk",
                "beam_size",
                "trunc_size",
                "qrange_file",
                "groundtruth_file",
                "result_file",
            ]
        )
        for spec in ranges:
            for beam, trunc in args.query_grid:
                run_id = f"{spec.label}_b{beam}_t{trunc}"
                writer.writerow(
                    [
                        run_id,
                        spec.selectivity,
                        spec.label,
                        args.topk,
                        beam,
                        trunc,
                        spec.path,
                        gt.get(spec.label, ""),
                        paths.result_dir / f"{run_id}.json",
                    ]
                )


def maybe_query(
    args: argparse.Namespace,
    paths: RunPaths,
    ranges: list[RangeSpec],
    gt: dict[str, Path],
    manifest: list[str],
) -> None:
    if args.stage not in {"all", "query"}:
        return
    write_batch_params(args, paths, ranges, gt)
    cmd: list[str | os.PathLike[str]] = [
        args.binary,
        "--verbose",
        "query_batch",
        "--dataset_file",
        args.dataset_file,
        "--index_file",
        paths.index_file,
        "--query_file",
        args.query_file,
        "--label_file",
        args.label_file,
        "--batch_params_file",
        paths.batch_params,
        "--batch_csv",
        paths.batch_csv,
        "--batch_result_dir",
        paths.result_dir,
        "--seed_policy",
        args.query_seed_policy,
        "--nav_degree",
        args.nav_degree,
        "--nav_scan_factor",
        args.nav_scan_factor,
        "--nav_stall_rounds",
        args.nav_stall_rounds,
        "--nav_front_keep",
        args.nav_front_keep,
        "--nav_tail_degree",
        args.nav_tail_degree,
        "--nav_early_stop_rounds",
        args.nav_early_stop_rounds,
        "--threads",
        args.threads,
    ]
    if args.sorted_range_index:
        cmd.extend(["--build_sorted_idx", "--use_sorted_range_idx"])
    run_cmd(cmd, dry_run=args.dry_run, manifest=manifest)


def write_manifest(
    args: argparse.Namespace,
    paths: RunPaths,
    ranges: list[RangeSpec],
    gt: dict[str, Path],
    commands: list[str],
) -> None:
    if args.dry_run:
        return
    payload = {
        "tag": args.tag,
        "dataset_file": str(args.dataset_file),
        "query_file": str(args.query_file),
        "label_file": str(args.label_file),
        "ranges": [
            {"label": r.label, "selectivity": r.selectivity, "qrange_file": str(r.path)}
            for r in ranges
        ],
        "groundtruth": {k: str(v) for k, v in gt.items()},
        "outputs": {
            "out_dir": str(paths.out_dir),
            "knng_file": str(paths.knng_file),
            "index_file": str(paths.index_file),
            "batch_params": str(paths.batch_params),
            "batch_csv": str(paths.batch_csv),
            "result_dir": str(paths.result_dir),
            "build_profile": str(paths.build_profile),
        },
        "build_params": {
            "knng_k": args.knng_k,
            "range_step": args.range_step,
            "ef_max": args.ef_max,
            "knng_degree_cap": args.knng_degree_cap,
            "seed_search_knng_cap": args.seed_search_knng_cap,
            "no_seed": args.no_seed,
            "reverse_refine_mode": args.reverse_refine_mode,
        },
        "query_params": {
            "topk": args.topk,
            "query_grid": args.query_grid,
            "query_seed_policy": args.query_seed_policy,
            "nav_degree": args.nav_degree,
            "sorted_range_index": args.sorted_range_index,
        },
        "commands": commands,
    }
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"manifest: {paths.manifest_file}")
    print(f"curve csv: {paths.batch_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["all", "knng", "build", "groundtruth", "query"],
        help="stage to run; all builds missing artifacts and runs the curve",
    )
    parser.add_argument("--tag", required=True, help="short run tag used for output names")
    parser.add_argument("--dataset-file", "--base-file", dest="dataset_file", type=Path, required=True, help="base vector file (.fvecs/.u8bin/.i8bin)")
    parser.add_argument("--query-file", type=Path, required=True, help="query vector file")
    parser.add_argument("--label-file", type=Path, required=True, help="JSON label array for base vectors")
    parser.add_argument(
        "--qrange-file",
        "--range-file",
        dest="qrange_file",
        action="append",
        required=True,
        help="query range file; use LABEL=PATH for stable curve labels; may repeat",
    )
    parser.add_argument(
        "--groundtruth-file",
        "--gt-file",
        dest="groundtruth_file",
        action="append",
        default=[],
        help="optional ground truth; use LABEL=PATH when multiple qrange files are used",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--out-dir", type=Path, help="explicit output directory")
    parser.add_argument("--knng-file", type=Path, help="reuse/write this KNNG path")
    parser.add_argument("--index-file", type=Path, help="reuse/write this RNSG index path")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--knng-k", type=int, default=128)
    parser.add_argument("--range-step", type=int, default=1500)
    parser.add_argument("--ef-max", type=int, default=200)
    parser.add_argument("--knng-degree-cap", type=int, default=128)
    parser.add_argument("--seed-search-knng-cap", type=int, default=32)
    parser.add_argument("--range-window-cap", type=int)
    parser.add_argument("--reverse-refine-mode", default="incoming", choices=["incoming", "full"])
    parser.add_argument("--enable-centroid-seed-search", dest="no_seed", action="store_false")
    parser.set_defaults(no_seed=True)
    parser.add_argument("--query-grid", type=parse_grid, default=parse_grid(DEFAULT_GRID))
    parser.add_argument("--query-seed-policy", default="header")
    parser.add_argument("--nav-degree", type=int, default=0)
    parser.add_argument("--nav-scan-factor", type=int, default=1)
    parser.add_argument("--nav-stall-rounds", type=int, default=0)
    parser.add_argument("--nav-front-keep", type=int, default=0)
    parser.add_argument("--nav-tail-degree", type=int, default=0)
    parser.add_argument("--nav-early-stop-rounds", type=int, default=0)
    parser.add_argument("--no-sorted-range-index", dest="sorted_range_index", action="store_false")
    parser.set_defaults(sorted_range_index=True)
    parser.add_argument("--qps-only", action="store_true", help="do not generate/use ground truth; output QPS only")
    parser.add_argument("--summary-max-points", type=int, default=12, help="max Pareto rows printed per workload")
    parser.add_argument("--no-summary", action="store_true", help="do not write/print Pareto summary after query")
    parser.add_argument("--force-knng", action="store_true")
    parser.add_argument("--force-index", action="store_true")
    parser.add_argument("--force-groundtruth", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.tag = clean_label(args.tag)
    args.dataset_file = args.dataset_file.expanduser().resolve()
    args.query_file = args.query_file.expanduser().resolve()
    args.label_file = args.label_file.expanduser().resolve()
    args.binary = args.binary.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if args.out_dir is not None:
        args.out_dir = args.out_dir.expanduser().resolve()
    if args.knng_file is not None:
        args.knng_file = args.knng_file.expanduser().resolve()
    if args.index_file is not None:
        args.index_file = args.index_file.expanduser().resolve()

    ranges = parse_qranges(args.qrange_file)
    gt = parse_groundtruth(args.groundtruth_file, ranges)
    ensure_inputs(args, ranges, gt)
    paths = make_paths(args)
    commands: list[str] = []

    if not args.dry_run and not args.binary.exists():
        raise FileNotFoundError(f"rnsg-new binary not found: {args.binary}; run xmake -r rnsg-new first")

    maybe_build_knng(args, paths, commands)
    maybe_build_index(args, paths, commands)
    gt = maybe_groundtruth(args, paths, ranges, gt, commands)
    maybe_query(args, paths, ranges, gt, commands)
    if args.stage in {"all", "query"} and not args.dry_run and not args.no_summary:
        summarize_curve_csv(paths.batch_csv, max_print_points=args.summary_max_points)
    write_manifest(args, paths, ranges, gt, commands)


if __name__ == "__main__":
    main()
