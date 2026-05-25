from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


NUMERIC_FIELDS = {"recall", "qps", "average_query_time_ns", "real_seconds"}
INT_FIELDS = {"topk", "beam_size", "trunc_size", "nav_degree"}


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key in NUMERIC_FIELDS:
        out[key] = _to_float(row.get(key))
    for key in INT_FIELDS:
        out[key] = _to_int(row.get(key))
    return out


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_qps(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def _is_completed(row: dict[str, Any]) -> bool:
    status = str(row.get("status", "")).lower()
    return status in {"", "completed", "ok"}


def _dominates(a: dict[str, Any], b: dict[str, Any], *, has_recall: bool) -> bool:
    aq, bq = a.get("qps"), b.get("qps")
    if aq is None or bq is None:
        return False
    if has_recall:
        ar, br = a.get("recall"), b.get("recall")
        if ar is None or br is None:
            return False
        return ar >= br and aq >= bq and (ar > br or aq > bq)
    return aq > bq


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r for r in rows if _is_completed(r) and r.get("qps") is not None]
    has_recall = any(r.get("recall") is not None for r in valid)
    if not has_recall:
        return sorted(valid, key=lambda r: (-(r.get("qps") or 0.0), str(r.get("run_id", ""))))

    valid = [r for r in valid if r.get("recall") is not None]
    front = []
    for row in valid:
        if not any(_dominates(other, row, has_recall=True) for other in valid if other is not row):
            front.append(row)
    return sorted(front, key=lambda r: (r.get("recall") or -1.0, -(r.get("qps") or 0.0)))


def _sample_for_print(rows: list[dict[str, Any]], max_points: int) -> list[dict[str, Any]]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    if max_points == 1:
        return [rows[-1]]
    last = len(rows) - 1
    picks = sorted({round(i * last / (max_points - 1)) for i in range(max_points)})
    return [rows[i] for i in picks]


def _write_csv(path: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    fields = [
        "selectivity_label",
        "run_id",
        "recall",
        "qps",
        "topk",
        "beam_size",
        "trunc_size",
        "average_query_time_ns",
        "real_seconds",
        "result_file",
        "qrange_file",
        "groundtruth_file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for label in sorted(groups):
            for row in groups[label]:
                writer.writerow({key: row.get(key, "") for key in fields})


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| run_id | recall | qps | topk | beam | trunc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {recall} | {qps} | {topk} | {beam} | {trunc} |".format(
                run_id=row.get("run_id", ""),
                recall=_fmt_float(row.get("recall"), 4),
                qps=_fmt_qps(row.get("qps")),
                topk=row.get("topk", ""),
                beam=row.get("beam_size", ""),
                trunc=row.get("trunc_size", ""),
            )
        )
    return lines


def _write_markdown(path: Path, groups: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# Pareto Frontier Summary", ""]
    for label in sorted(groups):
        rows = groups[label]
        lines.extend([f"## {label}", ""])
        if rows:
            lines.extend(_markdown_table(rows))
        else:
            lines.append("No completed rows with numeric QPS were found.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_summary(groups: dict[str, list[dict[str, Any]]], max_points: int) -> None:
    print("\nPareto frontier key points:")
    for label in sorted(groups):
        rows = groups[label]
        print(f"  [{label}] {len(rows)} point(s)")
        for row in _sample_for_print(rows, max_points):
            print(
                "    recall={recall} qps={qps} topk={topk} beam={beam} trunc={trunc} run={run}".format(
                    recall=_fmt_float(row.get("recall"), 4) or "n/a",
                    qps=_fmt_qps(row.get("qps")) or "n/a",
                    topk=row.get("topk", ""),
                    beam=row.get("beam_size", ""),
                    trunc=row.get("trunc_size", ""),
                    run=row.get("run_id", ""),
                )
            )


def summarize_curve_csv(
    curve_csv: Path,
    *,
    output_prefix: Path | None = None,
    max_print_points: int = 12,
    print_summary: bool = True,
) -> tuple[Path, Path, dict[str, list[dict[str, Any]]]]:
    curve_csv = curve_csv.resolve()
    if output_prefix is None:
        output_prefix = curve_csv.with_suffix("")
    output_prefix = output_prefix.resolve()

    with curve_csv.open(newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)
        rows = [_parse_row(row) for row in reader]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        label = str(row.get("selectivity_label") or row.get("selectivity") or "all")
        grouped[label].append(row)

    fronts = {label: pareto_front(items) for label, items in grouped.items()}
    csv_path = output_prefix.with_name(output_prefix.name + "_pareto.csv")
    md_path = output_prefix.with_name(output_prefix.name + "_pareto.md")
    _write_csv(csv_path, fronts)
    _write_markdown(md_path, fronts)
    if print_summary:
        _print_summary(fronts, max_print_points)
        print(f"Pareto CSV: {csv_path}")
        print(f"Pareto MD:  {md_path}")
    return csv_path, md_path, fronts
