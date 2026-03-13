"""
Impute, rank, and output final characteristic datasets.

Polars rewrite of chars_siz/impute_rank_output_bchmk.py.

Logic:
1. Load chars_a_raw.parquet and chars_q_raw.parquet (from merge_chars.py).
2. For overlapping accounting variables (available at both annual and quarterly
   frequency), pick the value from the *more recent* datadate; fall back to
   whichever is non-null.
3. Shift return one period forward (predict t+1 return with t characteristics).
4. Produce four output files:
   - chars_raw_no_impute.parquet  (merged, not imputed)
   - chars_raw_imputed.parquet    (industry-median then cross-sectional-median imputed)
   - chars_rank_no_impute.parquet (cross-sectionally ranked, not imputed → null→0)
   - chars_rank_imputed.parquet   (ranked version with null→0)

Inputs  (from OUTPUT_PATH = ../data/processed/):
    - chars_a_raw.parquet
    - chars_q_raw.parquet

Outputs (to OUTPUT_PATH):
    - chars_raw_no_impute.parquet
    - chars_raw_imputed.parquet
    - chars_rank_no_impute.parquet
    - chars_rank_imputed.parquet
"""

import polars as pl
from tqdm import tqdm
from functions import (
    OUTPUT_PATH, ffi49, fillna_ind, fillna_all, standardize
)


def _unique_keep_order(cols: list[str]) -> list[str]:
    """Deduplicate a list while preserving first-seen order."""
    return list(dict.fromkeys(cols))

# =====================================================================
#  Variable lists
# =====================================================================
# Identification / observation columns (kept as-is, never imputed)
OBS_VARS = [
    'gvkey', 'permno', 'jdate', 'ticker', 'conm', 'comnam',
    'sic', 'ret', 'retx', 'prc', 'shrout',
    # new CRSP v2 identifiers (replacing exchcd / shrcd)
    'primaryexch', 'conditionaltype', 'tradingstatusflg',
    'sharetype', 'securitytype', 'securitysubtype',
    'usincflg', 'issuertype',
]

# Accounting variables available at BOTH annual & quarterly frequency
# The first element is datadate (used to determine recency).
ACCOUNTING_VARS = [
    'datadate',
    'acc', 'bm', 'agr', 'alm', 'ato', 'cash', 'cashdebt', 'cfp', 'chcsho',
    'chtx', 'depr', 'ep', 'gma', 'grltnoa', 'lev', 'lgr', 'ni', 'noa', 'op',
    'pctacc', 'pm', 'rd_sale', 'rdm', 'rna', 'roa', 'roe', 'rsup', 'sgr', 'sp',
    'me_ia', 'bm_ia',
    'cashpr', 'cfp_ia', 'chatoia', 'egr', 'invest', 'chmom', 'rd',
    # quarterly also has chato, chpm — keep them in overlap list
    'chato', 'chpm',
]

# Annual-only characteristics
A_ONLY_VARS = [
    'adm', 'herf', 'hire',
    'absacc', 'age', 'chempia', 'chinv', 'convind', 'currat', 'divi', 'divo',
    'grcapx', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick',
    'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv',
    'quick', 'realestate', 'roic', 'salecash', 'salerec', 'saleinv',
    'secured', 'securedind', 'sin', 'tang', 'tb', 'chpmia',
    'pchcapx', 'chadv', 'grGW', 'obklg', 'chobklg', 'conv',
    'chdrc', 'rdbias', 'operprof', 'capxint', 'xadint',
    'm1', 'm2', 'm3', 'm4', 'm5', 'm6',
]

# Quarterly-only characteristics
Q_ONLY_VARS = [
    'abr', 'sue', 'cinvest', 'nincr', 'pscore',
    'roavol', 'stdacc', 'stdcf', 'scf', 'sgrvol',
    'm7', 'm8',
]

# Monthly-frequency characteristics (come from rolling_chars + accounting momentum)
M_VARS = [
    'beta', 'baspread', 'ill', 'maxret',
    'mom12m', 'mom1m', 'mom36m', 'mom60m', 'mom6m',
    're', 'rvar_capm', 'rvar_ff3', 'rvar_mean',
    'seas1a', 'std_dolvol', 'std_turn', 'zerotrade',
    'me', 'dy', 'turn', 'dolvol', 'indmom',
]


def _safe_select(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Select columns that exist in df, silently skip missing ones."""
    available = [c for c in cols if c in df.columns]
    return df.select(available)


def _reconcile_annual_quarterly(
    df_a: pl.DataFrame,
    df_q: pl.DataFrame,
    obs_vars: list[str],
    accounting_vars: list[str],
    a_only_vars: list[str],
    q_only_vars: list[str],
    m_vars: list[str],
) -> pl.DataFrame:
    """
    Merge annual and quarterly chars.

    For overlapping accounting variables, pick the most-recently-reported value:
      - If both annual and quarterly are available, use the one with later datadate.
      - If only one is available, use that.
    """
    # prefix accounting vars
    a_var_list = ['a_' + v for v in accounting_vars]
    q_var_list = ['q_' + v for v in accounting_vars]

    # --- annual side: obs + accounting(a_) + a_only + monthly ---
    a_cols = _unique_keep_order(obs_vars + accounting_vars + a_only_vars + m_vars)
    a_cols_avail = [c for c in a_cols if c in df_a.columns]
    df_a_sel = df_a.select(a_cols_avail)
    # rename accounting vars → a_ prefix
    rename_a = {v: f'a_{v}' for v in accounting_vars if v in df_a_sel.columns}
    df_a_sel = df_a_sel.rename(rename_a)

    # --- quarterly side: obs + accounting(q_) + q_only ---
    q_cols = _unique_keep_order(obs_vars + accounting_vars + q_only_vars)
    q_cols_avail = [c for c in q_cols if c in df_q.columns]
    df_q_sel = df_q.select(q_cols_avail)
    rename_q = {v: f'q_{v}' for v in accounting_vars if v in df_q_sel.columns}
    df_q_sel = df_q_sel.rename(rename_q)
    # drop obs columns that duplicate with annual side (keep keys)
    q_drop = [c for c in df_q_sel.columns
              if c in obs_vars and c not in ('gvkey', 'permno', 'jdate')]
    df_q_sel = df_q_sel.drop(q_drop)

    # merge
    df = df_a_sel.join(df_q_sel, on=['gvkey', 'permno', 'jdate'], how='left')

    # reconcile overlapping vars (skip datadate)
    for var in tqdm(accounting_vars[1:], desc='Reconciling A/Q'):
        a_col = f'a_{var}'
        q_col = f'q_{var}'
        if a_col not in df.columns and q_col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(var))
            continue
        if a_col not in df.columns:
            df = df.with_columns(pl.col(q_col).alias(var)).drop(q_col)
            continue
        if q_col not in df.columns:
            df = df.with_columns(pl.col(a_col).alias(var)).drop(a_col)
            continue

        # both exist → pick most recent, or whichever is available
        a_avail = pl.col(a_col).is_not_null()
        q_avail = pl.col(q_col).is_not_null()
        both = a_avail & q_avail

        # latest: if q_datadate < a_datadate → use annual, else quarterly
        latest = (
            pl.when(pl.col('q_datadate') < pl.col('a_datadate'))
              .then(pl.col(a_col))
              .otherwise(pl.col(q_col))
        )
        # available: prefer annual if present
        available = (
            pl.when(a_avail).then(pl.col(a_col)).otherwise(pl.col(q_col))
        )
        # final: if both available → use latest; otherwise use available
        df = df.with_columns(
            pl.when(both).then(latest).otherwise(available).alias(var)
        )
        df = df.drop([a_col, q_col])

    # drop a_datadate / q_datadate
    for c in ['a_datadate', 'q_datadate']:
        if c in df.columns:
            df = df.drop(c)

    return df


def _shift_return(df: pl.DataFrame) -> pl.DataFrame:
    """
    Shift return one month forward: t characteristics predict t+1 return.
    Rename retadj → ret for the final output.
    """
    df = df.sort(['permno', 'jdate'])

    # shift return and date forward
    df = df.with_columns([
        pl.col('ret').shift(-1).over('permno').alias('ret_lead'),
        pl.col('jdate').shift(-1).over('permno').alias('date'),
    ])
    df = df.drop('ret').rename({'ret_lead': 'ret'})

    # drop rows where future return is unknown
    df = df.filter(pl.col('ret').is_not_null())
    df = df.drop('jdate')  # date is now the return date

    # replace ±inf with null
    float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    ])

    return df


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    print("Loading raw chars...", flush=True)
    chars_a = pl.read_parquet(OUTPUT_PATH + 'chars_a_raw.parquet')
    chars_q = pl.read_parquet(OUTPUT_PATH + 'chars_q_raw.parquet')

    for label, df in [('chars_a', chars_a), ('chars_q', chars_q)]:
        print(f"  {label}: {df.shape}")

    # ------------------------------------------------------------------
    # Reconcile annual / quarterly
    # ------------------------------------------------------------------
    print("Reconciling annual & quarterly...", flush=True)
    df = _reconcile_annual_quarterly(
        chars_a, chars_q,
        OBS_VARS, ACCOUNTING_VARS, A_ONLY_VARS, Q_ONLY_VARS, M_VARS,
    )
    print(f"  Merged shape: {df.shape}")

    # ------------------------------------------------------------------
    # Shift return forward
    # ------------------------------------------------------------------
    print("Shifting return...", flush=True)
    df = _shift_return(df)
    print(f"  After shift: {df.shape}")

    # fill industry
    df = df.with_columns([
        pl.col('sic').forward_fill().over('permno'),
    ])
    df = df.with_columns([
        pl.col('sic').fill_null(0).cast(pl.Int64),
    ])

    # ------------------------------------------------------------------
    # Output 1: raw (no imputation)
    # ------------------------------------------------------------------
    print("Saving chars_raw_no_impute.parquet ...", flush=True)
    df.write_parquet(OUTPUT_PATH + 'chars_raw_no_impute.parquet')

    # ------------------------------------------------------------------
    # Output 2: imputed
    # ------------------------------------------------------------------
    print("Imputing...", flush=True)
    df_impute = df.clone()
    df_impute = df_impute.with_columns(pl.col('date').cast(pl.Date))

    # add ffi49
    df_impute = df_impute.with_columns([
        pl.col('sic').cast(pl.Int64),
    ])
    df_impute = df_impute.with_columns(ffi49().alias('ffi49'))
    df_impute = df_impute.with_columns([
        pl.col('ffi49').fill_null(49).cast(pl.Int64),
    ])

    # replace ±inf with null before imputation
    float_cols = [c for c in df_impute.columns if df_impute[c].dtype in (pl.Float32, pl.Float64)]
    df_impute = df_impute.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    ])

    # industry-median, then cross-sectional-median
    df_impute = fillna_ind(df_impute, method='median', ffi=49, not_fill_col=OBS_VARS)
    df_impute = fillna_all(df_impute, method='median', not_fill_col=OBS_VARS)

    # re is from IBES with lots of missing → fill remaining with 0
    if 're' in df_impute.columns:
        df_impute = df_impute.with_columns(pl.col('re').fill_null(0))

    print("Saving chars_raw_imputed.parquet ...", flush=True)
    df_impute.write_parquet(OUTPUT_PATH + 'chars_raw_imputed.parquet')

    # ------------------------------------------------------------------
    # Output 3: ranked (no imputation)
    # ------------------------------------------------------------------
    print("Ranking (no impute)...", flush=True)
    df_rank = df.clone()
    df_rank = df_rank.with_columns(pl.col('me').alias('lag_me'))
    # bm < 0 → null before ranking
    if 'bm' in df_rank.columns:
        df_rank = df_rank.with_columns(
            pl.when(pl.col('bm') < 0).then(None).otherwise(pl.col('bm')).alias('bm')
        )
    df_rank = standardize(df_rank)

    # log_me
    df_rank = df_rank.with_columns([
        pl.col('lag_me').log().alias('log_me'),
    ])
    # replace ±inf with 0
    float_cols = [c for c in df_rank.columns if df_rank[c].dtype in (pl.Float32, pl.Float64)]
    df_rank = df_rank.with_columns([
        pl.when(pl.col(c).is_infinite()).then(pl.lit(0)).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    ])

    # fill rank_ columns with 0
    rank_cols = [c for c in df_rank.columns if c.startswith('rank_')]
    df_rank = df_rank.with_columns([pl.col(c).fill_null(0) for c in rank_cols])

    print("Saving chars_rank_no_impute.parquet ...", flush=True)
    df_rank.write_parquet(OUTPUT_PATH + 'chars_rank_no_impute.parquet')

    # ------------------------------------------------------------------
    # Output 4: ranked (imputed)
    # ------------------------------------------------------------------
    # Re-use the same rank from no-impute but fill rank_ nulls with 0
    # (they should already be 0 from standardize, but be safe)
    print("Saving chars_rank_imputed.parquet ...", flush=True)
    df_rank.write_parquet(OUTPUT_PATH + 'chars_rank_imputed.parquet')

    print("Done.", flush=True)
    print(f"  Final columns ({len(df.columns)}): {df.columns}")
