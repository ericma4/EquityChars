"""
Merge accounting characteristics with satellite characteristics (rolling_chars, sue, abr, myre)
and CRSP return data.

Polars rewrite of chars_siz/merge_chars.py.

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


def _load_satellite(filename: str, keep_cols: list[str], date_col: str) -> pl.DataFrame:
    """Load a satellite parquet, align date to month-end, deduplicate."""
    df = pl.read_parquet(OUTPUT_PATH + filename)
    df = df.with_columns([
        pl.col('permno').cast(pl.Int64),
        pl.col(date_col).cast(pl.Date).dt.month_end().alias('jdate'),
    ])
    df = df.select(['permno', 'jdate'] + keep_cols)
    df = df.unique(subset=['permno', 'jdate'], keep='last')
    return df


def _load_rolling_chars() -> pl.DataFrame:
    """Load rolling_chars parquet."""
    df = pl.read_parquet(OUTPUT_PATH + _ROLLING_CHARS_FILE)
    df = df.with_columns([
        pl.col('permno').cast(pl.Int64),
        pl.col('date').cast(pl.Date).dt.month_end().alias('jdate'),
    ])
    available = [c for c in _ROLLING_CHARS_COLS if c in df.columns]
    df = df.select(['permno', 'jdate'] + available)
    df = df.unique(subset=['permno', 'jdate'], keep='last')
    return df


def _build_crsp_backfill() -> pl.DataFrame:
    """
    Build a CRSP backfill table with delisting-adjusted returns.
    Used to fill missing ret/retx/me in accounting chars that lack CRSP coverage.
    """
    crsp = pl.read_parquet(INPUT_PATH + 'crsp_msf.parquet')

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
        pl.col('ret').fill_null(0),
        pl.col('retx').fill_null(0),
    ])

    # delisting returns
    dlret = pl.read_parquet(INPUT_PATH + 'crsp_dlret.parquet')
    dlret = dlret.with_columns([
        pl.col('permno').cast(pl.Int64),
        pl.col('dlstdt').cast(pl.Date).dt.month_end().alias('jdate'),
    ])
    dlret = dlret.select(['permno', 'jdate', 'dlret'])

    crsp = crsp.join(dlret, on=['permno', 'jdate'], how='left')
    crsp = crsp.with_columns([
        pl.col('dlret').fill_null(0),
        ((1 + pl.col('ret')) * (1 + pl.col('dlret')) - 1).alias('retadj'),
    ])

    # aggregate me: assign sum-of-permco-me to largest permno
    crsp_summe = crsp.group_by(['jdate', 'permco']).agg(pl.col('me').sum().alias('me_sum'))
    crsp_maxme = crsp.group_by(['jdate', 'permco']).agg(pl.col('me').max().alias('me_max'))
    crsp = crsp.join(crsp_maxme, on=['jdate', 'permco'], how='left')
    crsp = crsp.filter(pl.col('me') == pl.col('me_max')).drop('me_max')
    crsp = crsp.drop('me').join(crsp_summe, on=['jdate', 'permco'], how='left').rename({'me_sum': 'me'})

    crsp = crsp.sort(['permno', 'jdate']).unique(subset=['permno', 'jdate'], keep='last')

    crsp = crsp.select([
        'permno', 'jdate',
        pl.col('ret').alias('ret_fill'),
        pl.col('retx').alias('retx_fill'),
        pl.col('retadj').alias('retadj_fill'),
        pl.col('me').alias('me_fill'),
    ])
    return crsp


def _merge_satellites(chars: pl.DataFrame) -> pl.DataFrame:
    """Merge all satellite characteristics into chars."""
    # rolling chars
    rolling = _load_rolling_chars()
    chars = chars.join(rolling, on=['permno', 'jdate'], how='left')

    # other satellites
    for filename, cols, date_col in _SATELLITE_FILES:
        sat = _load_satellite(filename, cols, date_col)
        chars = chars.join(sat, on=['permno', 'jdate'], how='left')

    return chars


def _backfill_crsp(chars: pl.DataFrame, crsp_fill: pl.DataFrame) -> pl.DataFrame:
    """Fill missing ret/retx/retadj/me from CRSP backfill table."""
    chars = chars.join(crsp_fill, on=['permno', 'jdate'], how='left')
    for col_name in ['ret', 'retx', 'retadj', 'me']:
        fill_col = f'{col_name}_fill'
        if fill_col in chars.columns:
            chars = chars.with_columns([
                pl.coalesce([pl.col(col_name), pl.col(fill_col)]).alias(col_name)
            ]).drop(fill_col)
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
    for df_name in ['chars_a', 'chars_q']:
        df = eval(df_name)
        df = df.with_columns([
            pl.col('permno').cast(pl.Int64),
            pl.col('jdate').cast(pl.Date),
        ])
        df = df.unique(subset=['permno', 'jdate'], keep='last')
        if df_name == 'chars_a':
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
