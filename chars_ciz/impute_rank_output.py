"""
Impute, rank, and output final characteristic datasets.

Logic:
1. Load chars_a_raw.parquet and chars_q_raw.parquet.
2. For overlapping accounting variables (both annual & quarterly), pick the
   value with the more recent datadate; fall back to whichever is non-null.
3. Shift return one period forward (t chars predict t+1 return).
4. Compute ffi49 industry codes for ALL outputs (before branching).
5. Produce four output files:
   - chars_raw_no_impute.parquet
   - chars_raw_imputed.parquet
   - chars_rank_no_impute.parquet
   - chars_rank_imputed.parquet
"""

import polars as pl
from tqdm import tqdm

from functions import ffi49, fillna_ind, fillna_all, standardize, INPUT_PATH, OUTPUT_PATH

# =====================================================================
#  Variable lists
# =====================================================================
OBS_VARS = [
    'gvkey', 'permno', 'jdate', 'ticker', 'conm', 'comnam',
    'sic', 'ret', 'retx', 'retadj',
    'exchcd', 'shrcd', 'prc', 'shrout',
]

ACCOUNTING_VARS = [
    'datadate',  # must be here (not OBS_VARS) so it gets a_/q_ prefix for recency comparison
    'acc', 'bm', 'agr', 'alm', 'ato', 'cash', 'cashdebt', 'cfp', 'chcsho',
    'chtx', 'depr', 'ep', 'gma', 'grltnoa', 'lev', 'lgr', 'ni', 'noa', 'op',
    'pctacc', 'pm', 'rd_sale', 'rdm', 'rna', 'roa', 'roe', 'rsup', 'sgr', 'sp',
    'me_ia', 'bm_ia',
    'cashpr', 'cfp_ia', 'chatoia', 'egr', 'invest', 'chmom', 'rd',
]

A_ONLY_VARS = [
    'adm', 'herf', 'hire',
    'absacc', 'age', 'chempia', 'chinv', 'convind', 'currat', 'divi', 'divo',
    'grcapx', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick',
    'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv',
    'quick', 'realestate', 'roic', 'salecash', 'salerec', 'saleinv',
    'secured', 'securedind', 'sin', 'tang', 'tb', 'chpmia',
]

Q_ONLY_VARS = [
    'abr', 'sue', 'cinvest', 'nincr', 'pscore',
    'roavol',
]

M_VARS = [
    'baspread', 'beta', 'ill', 'maxret',
    'mom12m', 'mom1m', 'mom36m', 'mom60m', 'mom6m',
    're', 'rvar_capm', 'rvar_ff3', 'rvar_mean',
    'seas1a', 'std_dolvol', 'std_turn', 'zerotrade',
    'me', 'dy', 'turn', 'dolvol', 'indmom',
]


# =====================================================================
#  Helpers
# =====================================================================
def _available_cols(df, cols):
    """Return only columns that exist in df, preserving order."""
    return [c for c in cols if c in df.columns]


def _replace_inf(df):
    """Replace ±inf and NaN with null in all float columns."""
    float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    return df.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in float_cols
    ])


def _reconcile(df_a, df_q):
    """
    Merge annual and quarterly characteristics.

    For variables in ACCOUNTING_VARS: use the more recent datadate when both
    frequencies are available, otherwise use whichever is non-null.
    """
    a_prefix = {v: f'a_{v}' for v in ACCOUNTING_VARS}
    q_prefix = {v: f'q_{v}' for v in ACCOUNTING_VARS}

    # --- annual side: obs + accounting + a_only + monthly ---
    a_cols = _available_cols(df_a, OBS_VARS + ACCOUNTING_VARS + A_ONLY_VARS + M_VARS)
    df_a_sel = df_a.select(a_cols).rename(
        {v: a_prefix[v] for v in ACCOUNTING_VARS if v in a_cols}
    )

    # --- quarterly side: obs + accounting + q_only + quarterly-only monthly vars ---
    q_cols = _available_cols(df_q, OBS_VARS + ACCOUNTING_VARS + Q_ONLY_VARS)
    q_drop = [c for c in OBS_VARS if c not in ('gvkey', 'permno', 'jdate')]
    df_q_sel = (
        df_q.select(q_cols)
        .rename({v: q_prefix[v] for v in ACCOUNTING_VARS if v in q_cols})
        .drop([c for c in q_drop if c in q_cols])
    )

    # merge
    df = df_a_sel.join(df_q_sel, on=['gvkey', 'permno', 'jdate'], how='left')

    # reconcile each overlapping variable (skip datadate)
    for var in tqdm(ACCOUNTING_VARS[1:], desc='Reconciling A/Q'):
        a_col, q_col = f'a_{var}', f'q_{var}'
        has_a = a_col in df.columns
        has_q = q_col in df.columns

        if not has_a and not has_q:
            df = df.with_columns(pl.lit(None).alias(var))
        elif not has_q:
            df = df.with_columns(pl.col(a_col).alias(var)).drop(a_col)
        elif not has_a:
            df = df.with_columns(pl.col(q_col).alias(var)).drop(q_col)
        else:
            # Both exist: pick by recency, fall back to whichever is available
            a_avail = pl.col(a_col).is_not_null()
            q_avail = pl.col(q_col).is_not_null()
            latest = (
                pl.when(pl.col('q_datadate') < pl.col('a_datadate'))
                .then(pl.col(a_col))
                .otherwise(pl.col(q_col))
            )
            available = pl.when(a_avail).then(pl.col(a_col)).otherwise(pl.col(q_col))
            df = df.with_columns(
                pl.when(a_avail & q_avail).then(latest).otherwise(available).alias(var)
            ).drop([a_col, q_col])

    # drop frequency-specific datadates
    df = df.drop([c for c in ['a_datadate', 'q_datadate'] if c in df.columns])
    return df


def _shift_return(df):
    """Shift return forward one month: t characteristics predict t+1 return."""
    df = df.sort(['permno', 'jdate'])
    df = df.with_columns([
        pl.col('ret').shift(-1).over('permno').alias('ret_lead'),
        pl.col('jdate').shift(-1).over('permno').alias('date'),
    ])
    df = df.drop('ret').rename({'ret_lead': 'ret'})

    df = df.filter(pl.col('ret').is_not_null()).drop('jdate')
    return _replace_inf(df)


def _rank_df(df):
    """Rank characteristics cross-sectionally, add log_me, fill nulls with 0."""
    out = df.clone()
    out = out.with_columns(pl.col('me').alias('lag_me'))
    # bm < 0 → null before ranking (GHZ convention)
    if 'bm' in out.columns:
        out = out.with_columns(
            pl.when(pl.col('bm') < 0).then(None).otherwise(pl.col('bm')).alias('bm')
        )
    out = standardize(out)
    out = out.with_columns(pl.col('lag_me').log().alias('log_me'))
    # (fixed-20260325) convert both NaN and ±inf to 0 after standardization so
    # rank-stage float cleanup matches _replace_inf().
    float_cols = [c for c in out.columns if out[c].dtype in (pl.Float32, pl.Float64)]
    out = out.with_columns([
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan())
        .then(pl.lit(0))
        .otherwise(pl.col(c))
        .alias(c)
        for c in float_cols
    ])
    rank_cols = [c for c in out.columns if c.startswith('rank_')]
    out = out.with_columns([pl.col(c).fill_null(0) for c in rank_cols])
    return out


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    print("Loading raw chars...", flush=True)
    chars_a = pl.read_parquet(OUTPUT_PATH + 'chars_a_raw.parquet')
    chars_q = pl.read_parquet(OUTPUT_PATH + 'chars_q_raw.parquet')

    chars_a = (
        chars_a.drop_nulls(subset=['permno'])
        .with_columns(pl.col('permno').cast(pl.Int64),
                       pl.col('jdate').cast(pl.Date))
        .unique(subset=['permno', 'jdate'])
    )
    chars_q = (
        chars_q.drop_nulls(subset=['permno'])
        .with_columns(pl.col('permno').cast(pl.Int64),
                       pl.col('jdate').cast(pl.Date))
        .unique(subset=['permno', 'jdate'])
    )

    print(f"  chars_a: {chars_a.shape},  chars_q: {chars_q.shape}", flush=True)

    # ------------------------------------------------------------------
    # Reconcile annual / quarterly
    # ------------------------------------------------------------------
    print("Reconciling annual & quarterly...", flush=True)
    df = _reconcile(chars_a, chars_q)

    # ------------------------------------------------------------------
    # Shift return forward
    # ------------------------------------------------------------------
    print("Shifting return...", flush=True)
    if 'retx' in df.columns:
        df = df.drop('retx')
    df = _shift_return(df)
    df = df.sort(['permno', 'date'])
    print(f"  After shift: {df.shape}", flush=True)

    # ------------------------------------------------------------------
    # Fill SIC + compute ffi49 (shared by ALL outputs — Bug 5 fix)
    # ------------------------------------------------------------------
    df = df.with_columns(pl.col('sic').forward_fill().over('permno'))
    df = df.with_columns(pl.col('sic').fill_null(0).cast(pl.Int64))
    df = df.with_columns(ffi49().alias('ffi49'))

    # ------------------------------------------------------------------
    # Output 1: raw (no imputation)
    # ------------------------------------------------------------------
    print("Saving chars_raw_no_impute.parquet ...", flush=True)
    df.write_parquet(OUTPUT_PATH + 'chars_raw_no_impute.parquet')

    # ------------------------------------------------------------------
    # Output 2: imputed (industry-median → cross-sectional-median)
    # ------------------------------------------------------------------
    print("Imputing...", flush=True)
    df_impute = df.clone()
    df_impute = df_impute.with_columns(pl.col('date').cast(pl.Date))
    df_impute = _replace_inf(df_impute)

    df_impute = fillna_ind(df_impute, method='median', ffi=49, not_fill_col=OBS_VARS)
    df_impute = fillna_all(df_impute, method='median', not_fill_col=OBS_VARS)

    # IBES-based `re` has sparse coverage → fill remaining with 0
    if 're' in df_impute.columns:
        df_impute = df_impute.with_columns(pl.col('re').fill_null(0))

    print("Saving chars_raw_imputed.parquet ...", flush=True)
    df_impute.write_parquet(OUTPUT_PATH + 'chars_raw_imputed.parquet')

    # ------------------------------------------------------------------
    # Output 3: ranked (no imputation)
    # ------------------------------------------------------------------
    print("Ranking (no impute)...", flush=True)
    df_rank = _rank_df(df)
    print("Saving chars_rank_no_impute.parquet ...", flush=True)
    df_rank.write_parquet(OUTPUT_PATH + 'chars_rank_no_impute.parquet')
    del df_rank

    # ------------------------------------------------------------------
    # Output 4: ranked (imputed)  — Bug 1 fix: rank the IMPUTED data
    # ------------------------------------------------------------------
    print("Ranking (imputed)...", flush=True)
    df_rank_imp = _rank_df(df_impute)
    print("Saving chars_rank_imputed.parquet ...", flush=True)
    df_rank_imp.write_parquet(OUTPUT_PATH + 'chars_rank_imputed.parquet')
