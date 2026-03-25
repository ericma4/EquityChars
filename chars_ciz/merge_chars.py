"""
Merge accounting characteristics with satellite characteristics (rolling_chars, sue, abr, myre)
and CRSP return data.

Inputs (all from OUTPUT_PATH = ../data/processed/):
    - chars_a_accounting.parquet   (from accounting.py)
    - chars_q_accounting.parquet   (from accounting.py)
    - rolling_chars.parquet        (from rolling_chars.py: beta, baspread, ill, maxret, rvar_capm,
                                    rvar_ff3, rvar_mean, std_dolvol, std_turn, zerotrade)
    - sue.parquet                  (from sue.py)
    - myre.parquet                 (from myre.py)
    - abr.parquet                  (from abr.py)
    - crsp_msf.parquet             (raw CRSP monthly, for delisting-adjusted returns / backfill)

Outputs:
    - chars_a_raw.parquet
    - chars_q_raw.parquet
"""

import polars as pl
from functions import INPUT_PATH, OUTPUT_PATH

# =====================================================================
#  Satellite characteristic files to merge
# =====================================================================
# Each entry: (filename, columns to keep besides permno/date, date_col)
_SATELLITE_FILES = [
    ('sue.parquet',     ['sue'],        'date'),
    ('myre.parquet',    ['re'],         'date'),
    ('abr.parquet',     ['abr'],        'date'),
]

# rolling_chars.parquet already has permno + date + many columns
_ROLLING_CHARS_FILE = 'rolling_chars.parquet'
_ROLLING_CHARS_COLS = [
    'beta', 'baspread', 'ill', 'maxret',
    'rvar_capm', 'rvar_ff3', 'rvar_mean',
    'std_dolvol', 'std_turn', 'zerotrade',
]

# CRSP CIZ -> accounting naming convention
_CRSP_SYNONYMS = {
    'mthcaldt': 'date',
    'mthprc': 'prc',
    'mthret': 'ret',
    'mthretx': 'retx',
}


def _load_satellite(filename, keep_cols, date_col):
    """Load a satellite parquet, align date to month-end, deduplicate."""
    df = pl.read_parquet(OUTPUT_PATH + filename)
    df = df.with_columns([
        pl.col('permno').cast(pl.Int64),
        pl.col(date_col).cast(pl.Date).dt.month_end().alias('jdate'),
    ])
    df = df.select(['permno', 'jdate'] + keep_cols)
    df = df.unique(subset=['permno', 'jdate'], keep='last')
    return df

def _clean_float_cols(df):
    """Replace NaN and ±inf with null in all float columns."""
    float_cols = [c for c in df.columns if df[c].dtype in (pl.Float32, pl.Float64)]
    if not float_cols:
        return df
    return df.with_columns([
        pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
          .then(None)
          .otherwise(pl.col(c))
          .alias(c)
        for c in float_cols
    ])


def _load_rolling_chars():
    """Load rolling_chars parquet."""
    df = pl.read_parquet(OUTPUT_PATH + _ROLLING_CHARS_FILE)
    df = df.with_columns([
        pl.col('permno').cast(pl.Int64),
        pl.col('date').cast(pl.Date).dt.month_end().alias('jdate'),
    ])
    # (fixed-20260325) clean rolling characteristic float columns at merge stage
    # so embedded NaN / inf from upstream do not flow directly into chars_a_raw / chars_q_raw.
    df = _clean_float_cols(df)
    available = [c for c in _ROLLING_CHARS_COLS if c in df.columns]
    df = df.select(['permno', 'jdate'] + available)
    df = df.unique(subset=['permno', 'jdate'], keep='last')
    return df


def _build_crsp_backfill():
    """
    Build a CRSP backfill table for missing ret/retx/me.
    CIZ (v2) mthret already includes delisting returns — no separate adjustment needed.
    """
    crsp = pl.read_parquet(INPUT_PATH + 'crsp_msf.parquet')
    crsp = crsp.rename(_CRSP_SYNONYMS, strict=False)

    # cast Decimal → Float64
    crsp = crsp.with_columns([
        pl.col(c).cast(pl.Float64)
        for c in crsp.columns
        if str(crsp[c].dtype).startswith('Decimal')
    ])

    crsp = crsp.filter(
        pl.col('primaryexch').is_in(['N', 'A', 'Q']) &
        (pl.col('conditionaltype') == 'RW') &
        (pl.col('tradingstatusflg') == 'A')
    )
    crsp = crsp.filter(
        (pl.col('sharetype') == 'NS') &
        (pl.col('securitytype') == 'EQTY') &
        (pl.col('securitysubtype') == 'COM') &
        (pl.col('usincflg') == 'Y') &
        (pl.col('issuertype').is_in(['ACOR', 'CORP']))
    )

    crsp = crsp.with_columns([
        pl.col('date').cast(pl.Date),
        pl.col('permno').cast(pl.Int64),
        pl.col('permco').cast(pl.Int64),
    ])
    crsp = crsp.with_columns([
        pl.col('date').dt.month_end().alias('jdate'),
    ])
    crsp = crsp.filter(pl.col('prc').is_not_null())
    crsp = crsp.with_columns([
        (pl.col('prc').abs() * pl.col('shrout')).alias('me'),
        pl.col('ret').cast(pl.Float64).fill_null(0),
        pl.col('retx').cast(pl.Float64).fill_null(0),
    ])

    # aggregate me: assign sum-of-permco-me to the permno with the largest me
    crsp_summe = crsp.group_by(['jdate', 'permco']).agg(pl.col('me').sum())
    crsp_maxme = crsp.group_by(['jdate', 'permco']).agg(pl.col('me').max())
    crsp = crsp.join(crsp_maxme, on=['jdate', 'permco', 'me'], how='inner')
    crsp = (crsp
        .drop('me')
        .join(crsp_summe, on=['jdate', 'permco'], how='inner')
        .sort(['permno', 'jdate'])
        .unique(subset=['permno', 'jdate'], keep='last')
    )

    crsp = crsp.select([
        'permno', 'jdate',
        pl.col('ret').alias('ret_fill'),
        pl.col('retx').alias('retx_fill'),
        pl.col('me').alias('me_fill'),
    ])
    return crsp


def _merge_satellites(chars):
    """Merge all satellite characteristics into chars."""
    # rolling chars
    rolling = _load_rolling_chars()
    chars = chars.join(rolling, on=['permno', 'jdate'], how='left')

    # other satellites
    for filename, cols, date_col in _SATELLITE_FILES:
        sat = _load_satellite(filename, cols, date_col)
        chars = chars.join(sat, on=['permno', 'jdate'], how='left')

    return chars


def _backfill_crsp(chars, crsp_fill):
    """Fill missing ret/retx/retadj/me from CRSP backfill table."""
    chars = chars.join(crsp_fill, on=['permno', 'jdate'], how='left')
    for col_name in ['ret', 'retx', 'me']:
        fill_col = f'{col_name}_fill'
        if fill_col in chars.columns and col_name in chars.columns:
            chars = chars.with_columns([
                pl.coalesce([pl.col(col_name), pl.col(fill_col)]).alias(col_name)
            ]).drop(fill_col)
        elif fill_col in chars.columns:
            chars = chars.rename({fill_col: col_name})
    # drop rows without return
    chars = chars.filter(
        pl.col('ret').is_not_null() &
        pl.col('retx').is_not_null()
    )
    return chars


# =====================================================================
#  Main
# =====================================================================
if __name__ == '__main__':
    print("Loading accounting characteristics...", flush=True)
    chars_a = pl.read_parquet(OUTPUT_PATH + 'chars_a_accounting.parquet')
    chars_q = pl.read_parquet(OUTPUT_PATH + 'chars_q_accounting.parquet')

    # ensure types
    for label, df in [('chars_a', chars_a), ('chars_q', chars_q)]:
        df = df.with_columns([
            pl.col('permno').cast(pl.Int64),
            pl.col('jdate').cast(pl.Date),
        ])
        df = df.unique(subset=['permno', 'jdate'], keep='last')
        if label == 'chars_a':
            chars_a = df
        else:
            chars_q = df

    print("Loading satellite characteristics...", flush=True)
    chars_a = _merge_satellites(chars_a)
    chars_q = _merge_satellites(chars_q)

    print("Building CRSP backfill...", flush=True)
    crsp_fill = _build_crsp_backfill()

    print("Backfilling CRSP data...", flush=True)
    chars_a = _backfill_crsp(chars_a, crsp_fill)
    chars_q = _backfill_crsp(chars_q, crsp_fill)

    # save
    print(f"Saving chars_a_raw.parquet  shape={chars_a.shape}", flush=True)
    chars_a.write_parquet(OUTPUT_PATH + 'chars_a_raw.parquet')

    print(f"Saving chars_q_raw.parquet  shape={chars_q.shape}", flush=True)
    chars_q.write_parquet(OUTPUT_PATH + 'chars_q_raw.parquet')

    print("Done.", flush=True)
