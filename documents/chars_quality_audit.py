from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import math
import os
from pathlib import Path
import time
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.feather as feather
import pyarrow.parquet as pq
from datetime import date


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_BASELINE_DIR = Path("/home/jianxin/chars/2025chars60")
DEFAULT_OUT_XLSX = ROOT / "documents" / f"chars_{date.today():%Y%m%d}.xlsx"

# File and column configurations
ACCOUNTING_FILES = ("chars_a_accounting.parquet", "chars_q_accounting.parquet")
ACCOUNTING_ID_COLS = ("permno", "gvkey", "ticker", "conm", "datadate", "jdate")

META_BASE = {"gvkey", "permno", "ticker", "conm", "comnam", "sic", "prc", "shrout", "ret"}
META_ACCOUNTING = META_BASE | {"datadate", "jdate", "primaryexch", "conditionaltype", "tradingstatusflg", "sharetype", "securitytype", "securitysubtype", "usincflg", "issuertype", "retx"}
META_FINAL = META_BASE | {"date", "ffi49"}
META_CORR = META_FINAL | {"exchcd", "shrcd", "lag_me", "__index_level_0__"}

# Correlation pairs: (label, current_file, baseline_file, is_rank)
CORR_PAIRS = [
    ("raw_no_impute", "chars_raw_no_impute.parquet", "chars60_raw_no_impute.feather", False),
    ("raw_imputed", "chars_raw_imputed.parquet", "chars60_raw_imputed.feather", False),
    ("rank_no_impute", "chars_rank_no_impute.parquet", "chars60_rank_no_impute.feather", True),
    ("rank_imputed", "chars_rank_imputed.parquet", "chars60_rank_imputed.feather", True),
]

PAIR_LABELS = {"raw_no_impute": "Raw / No Impute", "raw_imputed": "Raw / Imputed", "rank_no_impute": "Rank / No Impute", "rank_imputed": "Rank / Imputed"}
SATELLITE_CHARS = {"abr", "sue", "re"}
MARKET_CHARS = {"age", "baspread", "beta", "dolvol", "dy", "ill", "indmom", "maxret", "me", "mom1m", "mom6m", "mom12m", "mom36m", "mom60m", "rvar_capm", "rvar_ff3", "rvar_mean", "seas1a", "std_dolvol", "std_turn", "svar", "turn", "zerotrade"}

# Outlier policy defaults
# Binary indicators (exclude from outlier detection)
DEFAULT_DUMMY_EXCLUDE = {"rd", "divi", "divo", "sin"}
DEFAULT_LOG_ZSCORE_CHARS = {"me", "me_ia", "ala", "ni"}
DEFAULT_MAD_CHARS = {"mom1m", "seas1a", "chmom", "indmom"}
OUTLIER_Z_THRESHOLD = 3.0
OUTLIER_MAD_THRESHOLD = 3.5
OUTLIER_ABS_CAP = 1e10

EXCEL_MAX_ROWS = 1_000_000  # Safe limit for Excel
CPU_WORKERS = max(1, (os.cpu_count() or 4))
CORR_BATCH_SIZE = 64


def _format_progress(done: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "." * width + "] 0/0"
    filled = int(width * done / total)
    bar = "#" * filled + "." * (width - filled)
    return f"[{bar}] {done}/{total}"


def _print_progress(stage: str, done: int, total: int, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    speed = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / speed if speed > 0 else float("nan")
    eta_txt = f"{eta:,.1f}s" if math.isfinite(eta) else "--"
    print(f"[{stage}] {_format_progress(done, total)} | elapsed {elapsed:,.1f}s | eta {eta_txt}", flush=True)


def normalize_char(val: object) -> str:
    return "" if val is None or (isinstance(val, float) and math.isnan(val)) else str(val).strip()


def classify_source(char: str) -> str:
    return "satellite" if char in SATELLITE_CHARS else ("market" if char in MARKET_CHARS else "accounting")


def load_references(summary_csv: Path, summary_xlsx: Path) -> tuple[pd.DataFrame, set[str]]:
    summary = pd.read_csv(summary_csv)
    summary["char"] = summary["Acronym"].map(normalize_char)
    chars60 = pd.read_excel(summary_xlsx)
    chars60_set = (set(chars60["Acronym"].map(normalize_char)) | set(chars60["New_Acronym"].map(normalize_char))) - {""}
    return summary, chars60_set


def _signed_log1p_expr(col_name: str) -> pl.Expr:
    x = pl.col(col_name)
    return pl.when(x.is_not_null()).then(x.sign() * (x.abs() + 1.0).log()).otherwise(None)


def _clean_numeric_expr(col_name: str, cap_abs: float | None) -> pl.Expr:
    x = pl.col(col_name).cast(pl.Float64, strict=False)
    if cap_abs is not None and cap_abs > 0:
        x = pl.when(x.abs() > cap_abs).then(None).otherwise(x)
    return x


def compute_outlier_summary_polars(
    num_df: pl.DataFrame,
    present: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute counts/outlier stats under mixed policy (z-score + MAD)."""
    if num_df.is_empty() or not present:
        empty = pd.Series(dtype="float64")
        return empty, empty, empty, empty, empty, empty

    count_row = num_df.select([pl.col(c).is_not_null().sum().alias(c) for c in present]).to_dicts()[0]
    counts = pd.Series({c: int(count_row.get(c, 0) or 0) for c in present}, dtype="int64")

    means = pd.Series(np.nan, index=present, dtype="float64")
    stds = pd.Series(np.nan, index=present, dtype="float64")
    medians = pd.Series(np.nan, index=present, dtype="float64")
    mads = pd.Series(np.nan, index=present, dtype="float64")
    outliers = pd.Series(0, index=present, dtype="int64")

    z_chars = [c for c in present if c not in DEFAULT_DUMMY_EXCLUDE and c not in DEFAULT_MAD_CHARS]
    if z_chars:
        z_stat_exprs = []
        z_base_exprs: dict[str, pl.Expr] = {}
        for c in z_chars:
            base = _signed_log1p_expr(c) if c in DEFAULT_LOG_ZSCORE_CHARS else pl.col(c)
            z_base_exprs[c] = base
            z_stat_exprs.append(base.mean().alias(f"{c}__mean"))
            z_stat_exprs.append(base.std(ddof=0).alias(f"{c}__std"))

        z_row = num_df.select(z_stat_exprs).to_dicts()[0]
        z_out_exprs = []
        valid_z = []
        for c in z_chars:
            mu = z_row.get(f"{c}__mean")
            sd = z_row.get(f"{c}__std")
            means[c] = float(mu) if mu is not None else np.nan
            stds[c] = float(sd) if sd is not None else np.nan
            if np.isfinite(stds[c]) and stds[c] > 0 and np.isfinite(means[c]):
                z_out_exprs.append(((z_base_exprs[c] - pl.lit(float(means[c]))).abs() > pl.lit(float(OUTLIER_Z_THRESHOLD * stds[c]))).cast(pl.Int64).sum().alias(f"{c}__out"))
                valid_z.append(c)
        if z_out_exprs:
            z_out_row = num_df.select(z_out_exprs).to_dicts()[0]
            for c in valid_z:
                outliers[c] = int(z_out_row.get(f"{c}__out") or 0)

    mad_active = [c for c in present if c in DEFAULT_MAD_CHARS and c not in DEFAULT_DUMMY_EXCLUDE]
    if mad_active:
        med_row = num_df.select([pl.col(c).median().alias(c) for c in mad_active]).to_dicts()[0]
        for c in mad_active:
            medians[c] = float(med_row[c]) if med_row.get(c) is not None else np.nan

        mad_exprs = []
        mad_valid = []
        for c in mad_active:
            med = medians[c]
            if np.isfinite(med):
                mad_exprs.append((pl.col(c) - pl.lit(float(med))).abs().median().alias(c))
                mad_valid.append(c)
        if mad_exprs:
            mad_row = num_df.select(mad_exprs).to_dicts()[0]
            for c in mad_valid:
                mads[c] = float(mad_row[c]) if mad_row.get(c) is not None else np.nan

        mad_out_exprs = []
        mad_out_valid = []
        for c in mad_active:
            med = medians[c]
            mad = mads[c]
            if np.isfinite(med) and np.isfinite(mad) and mad > 0:
                robust_z = (pl.lit(0.6745) * (pl.col(c) - pl.lit(float(med))).abs() / pl.lit(float(mad)))
                mad_out_exprs.append((robust_z > pl.lit(float(OUTLIER_MAD_THRESHOLD))).cast(pl.Int64).sum().alias(f"{c}__out"))
                mad_out_valid.append(c)
        if mad_out_exprs:
            mad_out_row = num_df.select(mad_out_exprs).to_dicts()[0]
            for c in mad_out_valid:
                outliers[c] = int(mad_out_row.get(f"{c}__out") or 0)

    return counts, means, stds, medians, mads, outliers


def extract_outliers_fast(
    df: pl.DataFrame,
    means: pd.Series,
    stds: pd.Series,
    medians: pd.Series,
    mads: pd.Series,
    file_name: str,
    period: str,
    id_cols: list[str],
    top_k: int | None,
) -> list[dict]:
    """Polars-native outlier extraction under mixed policy (z-score + MAD)."""
    if df.is_empty() or not df.columns:
        return []
    
    records = []
    for char in df.columns:
        if char in id_cols or char in DEFAULT_DUMMY_EXCLUDE:
            continue
        raw_val_col = pl.col(char).cast(pl.Float64, strict=False)
        score_expr = None
        if char in DEFAULT_MAD_CHARS:
            med = medians.get(char, np.nan)
            mad = mads.get(char, np.nan)
            if np.isfinite(med) and np.isfinite(mad) and mad > 0:
                score_expr = (pl.lit(0.6745) * (pl.col("value") - pl.lit(float(med))) / pl.lit(float(mad)))
                threshold = OUTLIER_MAD_THRESHOLD
            else:
                continue
        else:
            mean, std = means.get(char, np.nan), stds.get(char, np.nan)
            if not np.isfinite(std) or std <= 0 or not np.isfinite(mean):
                continue
            base = pl.when(pl.col("value").is_not_null()).then(pl.col("value").sign() * (pl.col("value").abs() + 1.0).log()).otherwise(None) if char in DEFAULT_LOG_ZSCORE_CHARS else pl.col("value")
            score_expr = (base - pl.lit(float(mean))) / pl.lit(float(std))
            threshold = OUTLIER_Z_THRESHOLD

        out = (
            df.select([
                *(pl.col(c) for c in id_cols),
                raw_val_col.alias("value"),
            ])
            .filter(pl.col("value").is_not_null())
            .with_columns([
                score_expr.alias("zscore"),
            ])
            .with_columns([
                pl.col("zscore").abs().alias("abs_zscore"),
            ])
            .filter(pl.col("abs_zscore") > pl.lit(float(threshold)))
            .sort("abs_zscore", descending=True)
        )

        if top_k and top_k > 0:
            out = out.head(top_k)

        rows = out.to_dicts()
        if not rows:
            continue

        for row in rows:
            rec = {
                "file_name": file_name,
                "period": period,
                "char": char,
                "value": float(row["value"]),
                "mean": float(means.get(char, np.nan)) if np.isfinite(means.get(char, np.nan)) else np.nan,
                "std": float(stds.get(char, np.nan)) if np.isfinite(stds.get(char, np.nan)) else np.nan,
                "zscore": float(row["zscore"]),
                "abs_zscore": float(row["abs_zscore"]),
            }
            for c in id_cols:
                rec[c] = row.get(c)
            records.append(rec)
    
    return records


@lru_cache(maxsize=None)
def get_parquet_cols(path: Path) -> tuple[str, ...]:
    return tuple(pq.ParquetFile(path).schema.names)


@lru_cache(maxsize=None)
def get_feather_cols(path: Path) -> tuple[str, ...]:
    return tuple(feather.read_table(path).column_names)


def compute_stats_polars(num_df: pl.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute numeric count/mean/std/outlier_count for all columns via Polars."""
    if num_df.is_empty() or not num_df.columns:
        empty = pd.Series(dtype="float64")
        return empty, empty, empty, empty

    count_exprs = [pl.col(c).is_not_null().sum().alias(f"{c}__count") for c in num_df.columns]
    mean_exprs = [pl.col(c).mean().alias(f"{c}__mean") for c in num_df.columns]
    std_exprs = [pl.col(c).std(ddof=0).alias(f"{c}__std") for c in num_df.columns]
    stat_row = num_df.select(count_exprs + mean_exprs + std_exprs).to_dicts()[0]

    counts = pd.Series({c: int(stat_row.get(f"{c}__count") or 0) for c in num_df.columns}, dtype="int64")
    means = pd.Series({c: float(stat_row[f"{c}__mean"]) if stat_row.get(f"{c}__mean") is not None else np.nan for c in num_df.columns}, dtype="float64")
    stds = pd.Series({c: float(stat_row[f"{c}__std"]) if stat_row.get(f"{c}__std") is not None else np.nan for c in num_df.columns}, dtype="float64")

    outlier_exprs = []
    valid_cols = []
    for c in num_df.columns:
        sd = stds.get(c, np.nan)
        mu = means.get(c, np.nan)
        if np.isfinite(sd) and sd > 0 and np.isfinite(mu):
            outlier_exprs.append((((pl.col(c) - pl.lit(float(mu))).abs() > pl.lit(3.0 * float(sd))) & pl.col(c).is_not_null()).cast(pl.Int64).sum().alias(f"{c}__out"))
            valid_cols.append(c)

    outliers = pd.Series(0, index=num_df.columns, dtype="int64")
    if outlier_exprs:
        out_row = num_df.select(outlier_exprs).to_dicts()[0]
        for c in valid_cols:
            outliers[c] = int(out_row.get(f"{c}__out") or 0)

    return counts, means, stds, outliers


def audit_accounting_file(
    fname: str,
    pdir: Path,
    known: Iterable[str],
    top_k: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit single accounting file with optimized processing."""
    fpath = pdir / fname
    schema = list(get_parquet_cols(fpath))
    chars = [c for c in schema if c not in META_ACCOUNTING]
    ordered = sorted(set(known) | set(chars))
    present = [c for c in ordered if c in chars]

    date_col = "datadate" if "datadate" in schema else "jdate"
    id_cols = [c for c in ACCOUNTING_ID_COLS if c in schema]
    need_cols = list(dict.fromkeys([date_col] + id_cols + present))
    base_df = pl.read_parquet(fpath, columns=need_cols)
    row_total = base_df.height
    
    if present:
        char_df = base_df.select(present)
        raw_counts_row = char_df.select([pl.col(c).is_not_null().sum().alias(c) for c in present]).to_dicts()[0]
        raw_counts = pd.Series({c: int(raw_counts_row.get(c, 0) or 0) for c in present}, dtype="int64")

        num_df = base_df.select([_clean_numeric_expr(c, OUTLIER_ABS_CAP).alias(c) for c in present])
        counts, means, stds, medians, mads, outliers = compute_outlier_summary_polars(
            num_df, present
        )
    else:
        char_df = pl.DataFrame()
        num_df = pl.DataFrame()
        raw_counts = pd.Series(dtype="int64")
        counts = means = stds = medians = mads = outliers = pd.Series(dtype="float64")
    
    # Quality records
    quality = []
    for char in ordered:
        is_present = char in chars
        rec = {"file_name": fname, "char": char, "source_group": classify_source(char), "present_in_file": is_present, "rows_total": row_total if is_present else None, "non_missing": None, "missing_ratio": None, "outlier_rate": None}
        if is_present:
            n = int(raw_counts.get(char, 0))
            numeric_n = int(counts.get(char, 0))
            if char in DEFAULT_DUMMY_EXCLUDE:
                outlier_rate = None
            elif char in DEFAULT_MAD_CHARS:
                mad = float(mads.get(char, np.nan))
                outlier_rate = None if numeric_n == 0 or not np.isfinite(mad) or mad <= 0 else float(outliers.get(char, 0) / numeric_n)
            else:
                std = float(stds.get(char, np.nan))
                outlier_rate = None if numeric_n == 0 or not np.isfinite(std) or std <= 0 else float(outliers.get(char, 0) / numeric_n)
            rec.update({"rows_total": row_total, "non_missing": n, "missing_ratio": (row_total - n) / row_total, "outlier_rate": outlier_rate})
        quality.append(rec)
    
    # Period analysis
    period_rows, sample_rows = [], []
    period_filters = {
        "pre_1980": pl.col(date_col).cast(pl.Date, strict=False).dt.year() < 1980,
        "post_1980": pl.col(date_col).cast(pl.Date, strict=False).dt.year() >= 1980,
    }

    for period, cond in period_filters.items():
        period_base = base_df.filter(cond)
        total = period_base.height
        if total == 0 or not present:
            continue

        period_raw = period_base.select(present)
        p_raw_row = period_raw.select([pl.col(c).is_not_null().sum().alias(c) for c in present]).to_dicts()[0]
        p_raw_counts = pd.Series({c: int(p_raw_row.get(c, 0) or 0) for c in present}, dtype="int64")

        period_num = period_base.select([_clean_numeric_expr(c, OUTLIER_ABS_CAP).alias(c) for c in present])
        p_counts, p_means, p_stds, p_medians, p_mads, p_outliers = compute_outlier_summary_polars(
            period_num, present
        )
        
        for char in present:
            n = int(p_raw_counts.get(char, 0))
            p_numeric_n = int(p_counts.get(char, 0))
            if char in DEFAULT_DUMMY_EXCLUDE:
                outlier_rate = None
            elif char in DEFAULT_MAD_CHARS:
                mad = float(p_mads.get(char, np.nan))
                outlier_rate = None if p_numeric_n == 0 or not np.isfinite(mad) or mad <= 0 else float(p_outliers.get(char, 0) / p_numeric_n)
            else:
                std = float(p_stds.get(char, np.nan))
                outlier_rate = None if p_numeric_n == 0 or not np.isfinite(std) or std <= 0 else float(p_outliers.get(char, 0) / p_numeric_n)
            period_rows.append({"file_name": fname, "period": period, "char": char, "rows_total": total, "non_missing": n, "missing_ratio": (total - n) / total, "outlier_rate": outlier_rate})
        
        # Only extract outliers for post_1980
        if period == "post_1980":
            sample_df = period_base.select([
                *(pl.col(c) for c in id_cols),
                *(_clean_numeric_expr(c, OUTLIER_ABS_CAP).alias(c) for c in present),
            ])
            sample_rows.extend(
                extract_outliers_fast(
                    sample_df,
                    p_means,
                    p_stds,
                    p_medians,
                    p_mads,
                    fname,
                    period,
                    id_cols,
                    top_k,
                )
            )
    
    return pd.DataFrame(quality), pd.DataFrame(period_rows), pd.DataFrame(sample_rows)


def audit_accounting(
    pdir: Path,
    known: Iterable[str],
    top_k: int | None,
    sort_samples: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parallel audit of accounting files."""
    total = len(ACCOUNTING_FILES)
    started_at = time.perf_counter()
    _print_progress("accounting", 0, total, started_at)

    results = []
    with ThreadPoolExecutor(max_workers=min(2, CPU_WORKERS)) as exe:
        future_map = {
            exe.submit(
                audit_accounting_file,
                f,
                pdir,
                known,
                top_k,
            ): f
            for f in ACCOUNTING_FILES
        }
        done = 0
        for fut in as_completed(future_map):
            fname = future_map[fut]
            result = fut.result()
            results.append((fname, result))
            done += 1
            _print_progress("accounting", done, total, started_at)

    results.sort(key=lambda x: ACCOUNTING_FILES.index(x[0]))
    unpacked = [r for _, r in results]
    
    quality = pd.concat([r[0] for r in unpacked], ignore_index=True)
    period = pd.concat([r[1] for r in unpacked if not r[1].empty], ignore_index=True) if any(not r[1].empty for r in unpacked) else pd.DataFrame()
    samples = pd.concat([r[2] for r in unpacked if not r[2].empty], ignore_index=True) if any(not r[2].empty for r in unpacked) else pd.DataFrame()
    
    if not period.empty:
        period = period.sort_values(["period", "missing_ratio", "outlier_rate"], ascending=[True, False, False])
    if not samples.empty and sort_samples:
        samples = samples.sort_values(["file_name", "period", "char", "abs_zscore"], ascending=[True, True, True, False])
    if not samples.empty:
        if len(samples) > EXCEL_MAX_ROWS:
            sheets = (len(samples) + EXCEL_MAX_ROWS - 1) // EXCEL_MAX_ROWS
            print(f"[info] Outliers {len(samples):,} rows exceed one-sheet limit; will split into {sheets} Excel sheets.", flush=True)
    
    return quality, period, samples


def corr_scan_current(path: Path, cols: list[str]) -> pl.LazyFrame:
    return pl.scan_parquet(str(path)).select(cols).with_columns(pl.col("permno").cast(pl.Int64, strict=False), pl.col("date").cast(pl.Date, strict=False))


def corr_scan_baseline(path: Path, cols: list[str]) -> pl.LazyFrame:
    return pl.scan_ipc(str(path)).select(cols).with_columns(pl.col("permno").cast(pl.Int64, strict=False), pl.col("date").cast(pl.Date, strict=False))


def compute_correlation(cur_path: Path, old_path: Path, chars: list[str], is_rank: bool) -> pd.DataFrame:
    """Compute correlation for one pair using Polars."""
    if not chars:
        return pd.DataFrame(columns=["char", "overlap_count", "aligned_overlap_count", "spearman", "pearson"])

    prefix = "rank_" if is_rank else ""
    cur_cols = ["permno", "date"] + [f"{prefix}{c}" for c in chars]
    old_cols = ["permno", "date"] + [f"{prefix}{c}" for c in chars]
    
    cur = corr_scan_current(cur_path, cur_cols)
    old = corr_scan_baseline(old_path, old_cols).rename({"date": "date_old"}).with_columns(pl.col("date_old").alias("date"))
    joined = cur.join(old, on=["permno", "date"], how="inner", suffix="_old")

    records = []
    batch_size = CORR_BATCH_SIZE if len(chars) > CORR_BATCH_SIZE else len(chars)
    for batch_start in range(0, len(chars), batch_size):
        batch_chars = chars[batch_start:batch_start + batch_size]

        exprs = []
        key_map: list[tuple[str, str, str, str]] = []
        for i, char in enumerate(batch_chars):
            col = f"{prefix}{char}"
            old_col = f"{col}_old"
            overlap_key = f"overlap__{i}"
            spearman_key = f"spearman__{i}"
            pearson_key = f"pearson__{i}"
            exprs.extend([
                pl.when(pl.col(col).is_not_null() & pl.col(old_col).is_not_null()).then(1).otherwise(0).sum().alias(overlap_key),
                pl.corr(col, old_col, method="spearman").alias(spearman_key),
                pl.corr(col, old_col, method="pearson").alias(pearson_key),
            ])
            key_map.append((char, overlap_key, spearman_key, pearson_key))

        result = joined.select(exprs).collect(engine="streaming").to_dicts()[0]

        for char, overlap_key, spearman_key, pearson_key in key_map:
            overlap_val = result.get(overlap_key)
            records.append({
                "char": char,
                "overlap_count": overlap_val,
                "aligned_overlap_count": overlap_val,
                "spearman": result.get(spearman_key),
                "pearson": result.get(pearson_key),
            })
    
    return pd.DataFrame(records)


def compute_all_correlations(pdir: Path, bdir: Path) -> pd.DataFrame:
    """Parallel correlation computation."""
    def process_pair(pair):
        label, cur_file, old_file, is_rank = pair
        cur_cols = set(get_parquet_cols(pdir / cur_file))
        old_cols = set(get_feather_cols(bdir / old_file))
        
        if is_rank:
            cur_chars = {c.replace("rank_", "") for c in cur_cols if c.startswith("rank_")}
            old_chars = {c.replace("rank_", "") for c in old_cols if c.startswith("rank_")}
        else:
            cur_chars = cur_cols - META_CORR
            old_chars = old_cols - META_CORR
        
        overlap = sorted(cur_chars & old_chars)
        print(f"[corr] {label}: {len(overlap)} chars", flush=True)
        df = compute_correlation(pdir / cur_file, bdir / old_file, overlap, is_rank)
        df["pair"] = label
        return df
    
    total = len(CORR_PAIRS)
    started_at = time.perf_counter()
    _print_progress("correlation", 0, total, started_at)

    frames_with_label = []
    with ThreadPoolExecutor(max_workers=min(4, CPU_WORKERS)) as exe:
        future_map = {exe.submit(process_pair, pair): pair[0] for pair in CORR_PAIRS}
        done = 0
        for fut in as_completed(future_map):
            label = future_map[fut]
            frames_with_label.append((label, fut.result()))
            done += 1
            _print_progress("correlation", done, total, started_at)

    order = [p[0] for p in CORR_PAIRS]
    frames_with_label.sort(key=lambda x: order.index(x[0]))
    frames = [f for _, f in frames_with_label]
    return pd.concat(frames, ignore_index=True)


def aggregate_corr(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate correlation results."""
    wide = df.pivot(index="char", columns="pair", values="spearman").reset_index()
    wide.columns = ["char"] + [f"corr_{c}_spearman" for c in wide.columns[1:]]
    
    overlap = df.pivot(index="char", columns="pair", values="overlap_count").reset_index()
    overlap.columns = ["char"] + [f"overlap_{c}" for c in overlap.columns[1:]]
    
    aligned = df.pivot(index="char", columns="pair", values="aligned_overlap_count").reset_index()
    aligned.columns = ["char"] + [f"aligned_overlap_{c}" for c in aligned.columns[1:]]
    
    merged = wide.merge(overlap, on="char").merge(aligned, on="char")
    merged["corr_primary"] = merged.apply(lambda r: next((r.get(k) for k in ["corr_rank_imputed_spearman", "corr_rank_no_impute_spearman", "corr_raw_imputed_spearman", "corr_raw_no_impute_spearman"] if pd.notna(r.get(k))), None), axis=1)
    return merged, df


def build_summary(summary: pd.DataFrame, quality: pd.DataFrame, corr: pd.DataFrame, pdir: Path, chars60: set[str]) -> pd.DataFrame:
    """Build final summary table."""
    final_chars = set(get_parquet_cols(pdir / "chars_raw_imputed.parquet")) - META_FINAL
    acc_chars = set()
    for f in ACCOUNTING_FILES:
        acc_chars |= set(get_parquet_cols(pdir / f)) - META_ACCOUNTING
    
    acc_qual = quality[quality["present_in_file"] & (quality["file_name"].isin(ACCOUNTING_FILES))].sort_values(["char", "file_name"]).drop_duplicates("char", keep="first").set_index("char")
    
    rows = []
    for char in sorted(set(summary["char"]) | set(quality["char"])):
        acc = acc_qual.loc[char] if char in acc_qual.index else None
        rows.append({"char": char, "in_current_60_list": char in chars60, "present_in_current_final_output": char in final_chars, "present_in_accounting_output": char in acc_chars, "source_group": classify_source(char), "missing_ratio_main": None if acc is None else acc["missing_ratio"], "outlier_rate_main": None if acc is None else acc["outlier_rate"]})
    
    return summary.merge(pd.DataFrame(rows), on="char", how="left").merge(corr, on="char", how="left")


def save_plots(corr: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Save correlation plots."""
    out_dir.mkdir(parents=True, exist_ok=True)
    created = []
    
    vals = corr.loc[corr["spearman"].notna(), "spearman"].values
    if len(vals) == 0:
        return created
    
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = max(0.01, 0.05 * (vmax - vmin if vmax != vmin else 1))
    
    # Histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    bins = np.linspace(vmin - pad, vmax + pad, 50)
    for idx, (label, _,_, _) in enumerate(CORR_PAIRS):
        ax = axes[idx // 2, idx % 2]
        pvals = corr.loc[(corr["pair"] == label) & corr["spearman"].notna(), "spearman"].values
        if len(pvals) > 0:
            ax.hist(pvals, bins=bins, color="#4C72B0", alpha=0.85)
            ax.axvline(pvals.mean(), color="#DD8452", linestyle="--", label=f"mean={pvals.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_title(PAIR_LABELS[label])
        ax.grid(alpha=0.2)
    fig.suptitle("Spearman Correlation Distribution", fontsize=14)
    fig.tight_layout()
    path = out_dir / "corr_hist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    created.append(path)
    
    return created


def write_excel(summary: pd.DataFrame, corr: pd.DataFrame, period: pd.DataFrame, samples: pd.DataFrame, corr_wide: pd.DataFrame, out: Path, skip_corr: bool, include_outlier_sheet: bool = True) -> None:
    """Fast Excel writing with minimal sheets."""
    sheets = [("missing_outlier", period[["file_name", "period", "char", "rows_total", "missing_ratio", "outlier_rate"]] if not period.empty else pd.DataFrame())]
    if include_outlier_sheet and not samples.empty:
        if len(samples) <= EXCEL_MAX_ROWS:
            sheets.append(("outlier_samples", samples))
        else:
            for i, start in enumerate(range(0, len(samples), EXCEL_MAX_ROWS), start=1):
                chunk = samples.iloc[start:start + EXCEL_MAX_ROWS]
                sheets.append((f"outlier_samp_{i}", chunk))
    
    if not skip_corr:
        corr_display = corr.sort_values(["pair", "char"])
        present_mask = summary["present_in_accounting_output"].eq(True) if "present_in_accounting_output" in summary.columns else pd.Series(False, index=summary.index)
        in60_mask = summary["in_current_60_list"].eq(False) if "in_current_60_list" in summary.columns else pd.Series(False, index=summary.index)
        good = summary[present_mask & in60_mask]

        corr_cols = [c for c in ["pair", "char", "spearman", "aligned_overlap_count"] if c in corr_display.columns]
        good_cols = [c for c in ["Acronym", "char", "source_group", "present_in_accounting_output"] if c in good.columns]

        sheets.extend([
            ("correlations", corr_display[corr_cols] if corr_cols else pd.DataFrame()),
            ("good_not_in_60", good[good_cols].dropna() if good_cols else pd.DataFrame()),
        ])
    
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        for name, df in sheets:
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)


def run_audit(args: argparse.Namespace) -> Path:
    """Main audit pipeline."""
    t_start = time.perf_counter()
    out = args.out_xlsx or (ROOT / "documents" / f"chars_{args.audit_date}.xlsx")
    print(f"[start] audit -> {out}", flush=True)
    
    summary, chars60 = load_references(args.summary_csv, args.summary_xlsx)
    known = sorted(set(summary["char"]))
    top_k = None if args.outlier_all or args.outlier_top_k <= 0 else args.outlier_top_k

    t0 = time.perf_counter()
    quality, period, samples = audit_accounting(
        args.processed_dir,
        known,
        top_k,
        sort_samples=not args.skip_outlier_sheet,
    )
    print(f"[time] accounting phase {time.perf_counter() - t0:,.1f}s", flush=True)
    print(f"[audit] {len(samples):,} outlier samples", flush=True)

    if args.outlier_full_parquet and not samples.empty:
        out_parquet = args.outlier_parquet or out.with_name(f"{out.stem}_outliers.parquet")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(out_parquet, index=False)
        print(f"[outliers] full parquet -> {out_parquet}", flush=True)
    
    if args.skip_corr:
        print("[skip] correlations", flush=True)
        corr_wide, corr_detail = pd.DataFrame({"char": []}), pd.DataFrame(columns=["pair", "char", "spearman"])
    else:
        t1 = time.perf_counter()
        corr_detail = compute_all_correlations(args.processed_dir, args.baseline_dir)
        corr_wide, corr_detail = aggregate_corr(corr_detail)
        print(f"[time] correlation phase {time.perf_counter() - t1:,.1f}s", flush=True)
        if args.skip_plots:
            print("[skip] plots", flush=True)
        else:
            t2 = time.perf_counter()
            save_plots(corr_detail, ROOT / "documents")
            print(f"[time] plot phase {time.perf_counter() - t2:,.1f}s", flush=True)
    
    summary_enriched = build_summary(summary, quality, corr_wide, args.processed_dir, chars60)
    t3 = time.perf_counter()
    write_excel(summary_enriched, corr_detail, period, samples, corr_wide, out, args.skip_corr, include_outlier_sheet=not args.skip_outlier_sheet)
    print(f"[time] excel write phase {time.perf_counter() - t3:,.1f}s", flush=True)
    
    print(f"[done] {out} | total {time.perf_counter() - t_start:,.1f}s", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit characteristics quality")
    p.add_argument("--audit-date", default=f"{date.today():%Y%m%d}")
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    p.add_argument("--summary-csv", type=Path, default=ROOT / "documents" / "chars_summary.csv")
    p.add_argument("--summary-xlsx", type=Path, default=ROOT / "documents" / "chars60_summary_all.xlsx")
    p.add_argument("--out-xlsx", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--skip-corr", action="store_true")
    p.add_argument("--skip-plots", action="store_true", help="Skip correlation plot generation")
    p.add_argument("--skip-outlier-sheet", action="store_true", help="Do not write outlier sample sheets to Excel (faster, use parquet export instead)")
    p.add_argument("--outlier-top-k", type=int, default=0)
    p.add_argument("--outlier-all", action="store_true")
    p.add_argument("--outlier-full-parquet", action="store_true", help="Export full outlier samples to a parquet file")
    p.add_argument("--outlier-parquet", type=Path, default=None, help="Path for full outlier parquet output (used with --outlier-full-parquet)")
    return p.parse_args()


if __name__ == "__main__":
    run_audit(parse_args())
