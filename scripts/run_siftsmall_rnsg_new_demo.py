#!/usr/bin/env python3
"""End-to-end SIFTsmall demo for the rnsg-new target.

The script keeps all generated files under ignored local directories by default:

  dataset/siftsmall_demo/              input vectors copied/downloaded here
  artifacts/siftsmall_rnsg_new_demo/   labels, ranges, KNNG, index, results

Typical usage:

  python3 scripts/run_siftsmall_rnsg_new_demo.py all --threads 8

Use --dry-run to print the exact commands without executing them.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shlex
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from rnsg_curve_utils import summarize_curve_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "build/linux/x86_64/release/rnsg-new"
DEFAULT_DATA_ROOT = REPO_ROOT / "dataset/siftsmall_demo"
DEFAULT_WORK_DIR = REPO_ROOT / "artifacts/siftsmall_rnsg_new_demo"
DEFAULT_LOCAL_SIFTSMALL = Path("/mnt/win-dai/Vectors/siftsmall")
SIFTSMALL_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
SIFTSMALL_FILES = ("siftsmall_base.fvecs", "siftsmall_query.fvecs")


def shell_join(cmd: list[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def run_cmd(cmd: list[str | os.PathLike[str]], *, dry_run: bool) -> None:
    print(f"$ {shell_join(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True)


def safe_extract(tar: tarfile.TarFile, dst: Path) -> None:
    dst = dst.resolve()
    for member in tar.getmembers():
        member_path = (dst / member.name).resolve()
        if not str(member_path).startswith(str(dst) + os.sep):
            raise RuntimeError(f"unsafe tar entry: {member.name}")
    tar.extractall(dst)


def ensure_siftsmall(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root.resolve()
    data_dir = data_root / "siftsmall"
    data_dir.mkdir(parents=True, exist_ok=True)

    base = data_dir / "siftsmall_base.fvecs"
    query = data_dir / "siftsmall_query.fvecs"
    if base.exists() and query.exists():
        return base, query

    source_dirs = []
    if args.source_data_dir is not None:
        source_dirs.append(args.source_data_dir.resolve())
    source_dirs.append(DEFAULT_LOCAL_SIFTSMALL)

    for source_dir in source_dirs:
        if all((source_dir / name).exists() for name in SIFTSMALL_FILES):
            for name in SIFTSMALL_FILES:
                dst = data_dir / name
                if not dst.exists():
                    print(f"copy {source_dir / name} -> {dst}")
                    if not args.dry_run:
                        shutil.copy2(source_dir / name, dst)
            return base, query

    archive = data_root / "siftsmall.tar.gz"
    print(f"download {SIFTSMALL_URL} -> {archive}")
    if not args.dry_run:
        urllib.request.urlretrieve(SIFTSMALL_URL, archive)
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, data_root)

    return base, query


def fvecs_count(path: Path) -> int:
    with path.open("rb") as fin:
        dim = int.from_bytes(fin.read(4), "little", signed=False)
        file_size = path.stat().st_size
    if dim <= 0:
        raise RuntimeError(f"invalid fvecs dimension in {path}")
    return file_size // ((dim + 1) * 4)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(value, fout, separators=(",", ":"))


def selectivity_tag(sel: float) -> str:
    pct = sel * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"sel{int(round(pct))}pct"
    return f"sel{pct:.3g}pct".replace(".", "p")


def parse_selectivities(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        val = float(item)
        if val <= 0.0 or val > 1.0:
            raise argparse.ArgumentTypeError("selectivities must be in (0, 1]")
        values.append(val)
    if not values:
        raise argparse.ArgumentTypeError("at least one selectivity is required")
    return values


def parse_grid(raw: str) -> list[tuple[int, int]]:
    grid: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        beam_s, trunc_s = item.split(":", 1)
        beam, trunc = int(beam_s), int(trunc_s)
        if beam <= 0 or trunc <= 0:
            raise argparse.ArgumentTypeError("beam/trunc values must be > 0")
        grid.append((beam, trunc))
    if not grid:
        raise argparse.ArgumentTypeError("query grid cannot be empty")
    return grid


def qrange_path(args: argparse.Namespace, sel: float) -> Path:
    return args.work_dir.resolve() / f"qrange_{selectivity_tag(sel)}.json"


def gt_path(args: argparse.Namespace, sel: float) -> Path:
    return args.work_dir.resolve() / f"groundtruth_{selectivity_tag(sel)}_k{args.topk}.json"


def result_path(args: argparse.Namespace, sel: float, beam: int, trunc: int) -> Path:
    return (
        args.work_dir.resolve()
        / "results"
        / f"{selectivity_tag(sel)}_b{beam}_t{trunc}.json"
    )


def prepare(args: argparse.Namespace) -> None:
    base, query = ensure_siftsmall(args)
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run and (not base.exists() or not query.exists()):
        print("dry-run: input vectors are not present yet; skip JSON generation")
        return

    n_base = fvecs_count(base)
    n_query = fvecs_count(query)
    if n_query < args.num_queries:
        raise RuntimeError(f"SIFTsmall only has {n_query} queries")

    rng = random.Random(args.seed)
    label_max = max(args.label_max, n_base * 10)
    labels = [rng.randrange(label_max + 1) for _ in range(n_base)]
    write_json(args.label_file, labels)

    for sel in args.selectivities:
        width = max(1, int(math.floor((label_max + 1) * sel)))
        ranges: list[int] = []
        for _ in range(args.num_queries):
            left = rng.randrange(0, label_max - width + 2)
            ranges.extend([left, left + width - 1])
        write_json(qrange_path(args, sel), ranges)

    print(f"prepared labels/ranges under {work_dir}")
    print(f"base={base}")
    print(f"query={query}")


def compile_target(args: argparse.Namespace) -> None:
    run_cmd(["xmake", "f", "-m", "release"], dry_run=args.dry_run)
    run_cmd(["xmake", "-r", "rnsg-new"], dry_run=args.dry_run)


def build_knng(args: argparse.Namespace) -> None:
    base, _ = ensure_siftsmall(args)
    args.knng_file.parent.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            args.binary,
            "--verbose",
            "knng",
            "--dataset_file",
            base,
            "-k",
            args.knng_k,
            "--graph_file",
            args.knng_file,
            "--threads",
            args.threads,
        ],
        dry_run=args.dry_run,
    )


def build_index(args: argparse.Namespace) -> None:
    base, _ = ensure_siftsmall(args)
    args.index_file.parent.mkdir(parents=True, exist_ok=True)
    profile = args.work_dir.resolve() / "build_profile.json"
    run_cmd(
        [
            args.binary,
            "--verbose",
            "build",
            "--range_step",
            args.range_step,
            "--dataset_file",
            base,
            "--knng_file",
            args.knng_file,
            "--label_file",
            args.label_file,
            "--index_file",
            args.index_file,
            "--ef_max",
            args.ef_max,
            "--disable_centroid_seed_search",
            "--reverse_refine_mode",
            "incoming",
            "--seed_search_knng_cap",
            args.seed_search_knng_cap,
            "--knng_degree_cap",
            args.knng_degree_cap,
            "--profile_build_json",
            profile,
            "--threads",
            args.threads,
        ],
        dry_run=args.dry_run,
    )


def build_groundtruth(args: argparse.Namespace) -> None:
    base, query = ensure_siftsmall(args)
    for sel in args.selectivities:
        run_cmd(
            [
                args.binary,
                "--verbose",
                "groundtruth",
                "--dataset_file",
                base,
                "--query_file",
                query,
                "--label_file",
                args.label_file,
                "--qrange_file",
                qrange_path(args, sel),
                "--qnumber",
                args.topk,
                "--result_file",
                gt_path(args, sel),
                "--threads",
                args.threads,
            ],
            dry_run=args.dry_run,
        )


def write_batch_params(args: argparse.Namespace) -> Path:
    params_file = args.work_dir.resolve() / "query_batch_params.csv"
    params_file.parent.mkdir(parents=True, exist_ok=True)
    with params_file.open("w", newline="", encoding="utf-8") as fout:
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
        for sel in args.selectivities:
            tag = selectivity_tag(sel)
            for beam, trunc in args.query_grid:
                writer.writerow(
                    [
                        f"{tag}_b{beam}_t{trunc}",
                        sel,
                        tag,
                        args.topk,
                        beam,
                        trunc,
                        qrange_path(args, sel),
                        gt_path(args, sel),
                        result_path(args, sel, beam, trunc),
                    ]
                )
    return params_file


def query_batch(args: argparse.Namespace) -> None:
    base, query = ensure_siftsmall(args)
    params_file = write_batch_params(args)
    results_dir = args.work_dir.resolve() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            args.binary,
            "--verbose",
            "query_batch",
            "--dataset_file",
            base,
            "--index_file",
            args.index_file,
            "--query_file",
            query,
            "--label_file",
            args.label_file,
            "--batch_params_file",
            params_file,
            "--batch_csv",
            args.batch_csv,
            "--batch_result_dir",
            results_dir,
            "--seed_policy",
            "header",
            "--nav_degree",
            0,
            "--nav_scan_factor",
            1,
            "--nav_stall_rounds",
            0,
            "--nav_front_keep",
            0,
            "--nav_tail_degree",
            0,
            "--nav_early_stop_rounds",
            0,
            "--build_sorted_idx",
            "--use_sorted_range_idx",
            "--threads",
            args.threads,
        ],
        dry_run=args.dry_run,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "stage",
        choices=["compile", "prepare", "knng", "build", "groundtruth", "query", "all"],
        help="demo stage to run",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--source-data-dir", type=Path)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-max", type=int, default=100_000)
    parser.add_argument("--knng-k", type=int, default=64)
    parser.add_argument("--range-step", type=int, default=256)
    parser.add_argument("--ef-max", type=int, default=96)
    parser.add_argument("--seed-search-knng-cap", type=int, default=32)
    parser.add_argument("--knng-degree-cap", type=int, default=64)
    parser.add_argument(
        "--selectivities",
        type=parse_selectivities,
        default=parse_selectivities("0.1"),
        help="comma-separated selectivities, e.g. 0.01,0.1,0.5",
    )
    parser.add_argument(
        "--query-grid",
        type=parse_grid,
        default=parse_grid("24:8,32:12,48:16,64:24,96:32,128:48"),
        help="comma-separated beam:trunc pairs",
    )
    parser.add_argument("--summary-max-points", type=int, default=12)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.work_dir = args.work_dir.resolve()
    args.label_file = args.work_dir / "labels.json"
    args.knng_file = args.work_dir / f"siftsmall_k{args.knng_k}.knng.graph"
    args.index_file = (
        args.work_dir
        / f"siftsmall_rnsg_new_k{args.knng_k}_rs{args.range_step}_m{args.ef_max}.graph"
    )
    args.batch_csv = args.work_dir / "query_batch_results.csv"
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = normalize_args(parser.parse_args())

    if args.stage == "compile":
        compile_target(args)
    elif args.stage == "prepare":
        prepare(args)
    elif args.stage == "knng":
        build_knng(args)
    elif args.stage == "build":
        build_index(args)
    elif args.stage == "groundtruth":
        build_groundtruth(args)
    elif args.stage == "query":
        query_batch(args)
        if not args.dry_run and not args.no_summary:
            summarize_curve_csv(args.batch_csv, max_print_points=args.summary_max_points)
    elif args.stage == "all":
        if not args.skip_compile:
            compile_target(args)
        prepare(args)
        build_knng(args)
        build_index(args)
        build_groundtruth(args)
        query_batch(args)
        if not args.dry_run and not args.no_summary:
            summarize_curve_csv(args.batch_csv, max_print_points=args.summary_max_points)
        print(f"batch results: {args.batch_csv}")


if __name__ == "__main__":
    main()
