#!/usr/bin/env python3
"""Complete, transparent Valentine schema-matching pilot.

The original handover file contained only the scoring primitives.  This
version runs the ten real Valentine pairs, evaluates name-only and combined
multi-signal Hungarian matching, and writes pair-level plus aggregate metrics.
It is a discovery baseline, not a claim of state-of-the-art performance.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment


def norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def name_score(a: object, b: object) -> float:
    na, nb = norm_col(a), norm_col(b)
    ta, tb = set(na.split()), set(nb.split())
    tok = len(ta & tb) / max(1, len(ta | tb))
    return 0.6 * (ratio(na, nb) / 100.0) + 0.4 * tok


def family(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"
    if pd.api.types.is_numeric_dtype(series):
        return "num"
    sample = series.dropna().astype(str).head(100)
    if len(sample) and sample.str.match(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}").mean() > 0.5:
        return "date"
    return "cat"


def type_score(a: pd.Series, b: pd.Series) -> float:
    fa, fb = family(a), family(b)
    return 1.0 if fa == fb else (0.4 if {fa, fb} == {"num", "cat"} else 0.0)


def sample_vals(series: pd.Series, n: int = 1000) -> pd.Series:
    values = series.dropna()
    if len(values) > n:
        values = values.sample(n, random_state=0)
    return values.astype(str).str.strip().str.lower()


def value_overlap(a: pd.Series, b: pd.Series, n: int = 1000) -> float:
    left, right = set(sample_vals(a, n)), set(sample_vals(b, n))
    return len(left & right) / max(1, len(left | right)) if left or right else 0.0


def range_score(a: pd.Series, b: pd.Series) -> float:
    if family(a) != "num" or family(b) != "num":
        return 0.0
    x = pd.to_numeric(a, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    y = pd.to_numeric(b, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xa, xb = np.percentile(x, [1, 99])
    ya, yb = np.percentile(y, [1, 99])
    intersection = max(0.0, min(xb, yb) - max(xa, ya))
    union = max(xb, yb) - min(xa, ya)
    return intersection / union if union > 0 else 1.0


def format_score(a: pd.Series, b: pd.Series, n: int = 300) -> float:
    def pattern(value: str) -> str:
        value = re.sub(r"\d", "9", value)
        value = re.sub(r"[A-Za-z]", "A", value)
        return re.sub(r"\s+", " ", value)

    left = sample_vals(a, n).map(pattern).value_counts(normalize=True)
    right = sample_vals(b, n).map(pattern).value_counts(normalize=True)
    common = set(left.index) & set(right.index)
    return sum(min(left[key], right[key]) for key in sorted(common))


def score_matrix(source: pd.DataFrame, target: pd.DataFrame, method: str) -> np.ndarray:
    scores = np.zeros((len(source.columns), len(target.columns)))
    for i, left_name in enumerate(source.columns):
        for j, right_name in enumerate(target.columns):
            ns = name_score(left_name, right_name)
            if method == "name":
                scores[i, j] = ns
                continue
            ts = type_score(source[left_name], target[right_name])
            vo = value_overlap(source[left_name], target[right_name])
            rs = range_score(source[left_name], target[right_name])
            fs = format_score(source[left_name], target[right_name])
            scores[i, j] = 0.35 * ns + 0.25 * ts + 0.25 * vo + 0.10 * rs + 0.05 * fs
    return scores


def evaluate(source: pd.DataFrame, target: pd.DataFrame, ground_truth: set[tuple[str, str]], method: str, audit=None) -> dict[str, float | int | str]:
    score = score_matrix(source, target, method)
    rows, cols = linear_sum_assignment(-score)
    predicted = {(source.columns[i], target.columns[j]) for i, j in zip(rows, cols)}
    if audit is not None:
        audit.extend(dict(source_column=source.columns[i],target_column=target.columns[j],
                          score=float(score[i,j]),correct=(source.columns[i],target.columns[j]) in ground_truth,
                          method=method) for i,j in zip(rows,cols))
    true_positive = len(predicted & ground_truth)
    precision = true_positive / max(1, len(predicted))
    recall = true_positive / max(1, len(ground_truth))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    by_source = {left: target.columns[int(np.argmax(score[i]))] for i, left in enumerate(source.columns)}
    top1_hits = sum((left, right) in ground_truth and by_source[left] == right for left, right in ground_truth)
    return {
        "method": method,
        "n_source": len(source.columns),
        "n_target": len(target.columns),
        "n_ground_truth": len(ground_truth),
        "n_predicted": len(predicted),
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "top1_recall": top1_hits / max(1, len(ground_truth)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pairs = sorted(path for path in args.data_dir.iterdir() if (path / "source_table.csv").exists() and (path / "target_table.csv").exists() and (path / "ground_truth.json").exists())
    rows: list[dict[str, float | int | str]] = []
    matches=[]
    for pair in pairs:
        source = pd.read_csv(pair / "source_table.csv", low_memory=False)
        target = pd.read_csv(pair / "target_table.csv", low_memory=False)
        gt_json = json.loads((pair / "ground_truth.json").read_text(encoding="utf-8"))
        ground_truth = {(item["source_column"], item["target_column"]) for item in gt_json["matches"]}
        for method in ("name", "combined"):
            matched=[]
            row = evaluate(source, target, ground_truth, method, matched)
            matches.extend(dict(pair=pair.name,**m) for m in matched)
            row["pair"] = pair.name
            rows.append(row)
        print(f"{pair.name}: done", flush=True)

    detail = pd.DataFrame(rows)
    detail = detail[["pair", "method", "n_source", "n_target", "n_ground_truth", "n_predicted", "true_positive", "precision", "recall", "f1", "top1_recall"]]
    detail.to_csv(args.outdir / "valentine_pilot_pair_results.csv", index=False)
    summary = detail.groupby("method", as_index=False).agg(
        pairs=("pair", "count"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_top1_recall=("top1_recall", "mean"),
    )
    summary.to_csv(args.outdir / "valentine_pilot_summary.csv", index=False)
    pd.DataFrame(matches).to_csv(args.outdir/'matched_columns.csv',index=False)
    paired=detail.pivot(index='pair',columns='method',values='f1')
    delta=(paired['combined']-paired['name']).to_numpy()
    boot=np.random.default_rng(42).choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
    (args.outdir/'config.json').write_text(json.dumps(dict(
        pairs=len(pairs),weights=dict(name=.35,type=.25,values=.25,range=.10,format=.05),
        assignment='forced one-to-one Hungarian, no tuned threshold',sample_seed=0,
        paired_mean_f1_delta=float(delta.mean()),paired_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
        caveat='Local 10-pair pilot; no comparison with all official Valentine methods.'),indent=2),encoding='utf-8')
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
