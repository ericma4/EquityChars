### Contact

- Jianxin Ma
- jianxin.ma@warwick.ac.uk

### Version

- All in Python
- Two pipeline variants live in this repository:
  - **`chars_ciz/`** — current CIZ (CRSP CIZ / v2) pipeline. Use this.
  - **`chars_siz/`** — legacy SIZ (CRSP SIZ / v1) pipeline. Kept for reference only.
- The SAS version is here [EquityCharacteristicsSAS](https://feng-cityuhk.github.io/EquityCharacteristicsSAS/)
- Extension to [China A Share Market](https://github.com/Quantactix/ChinaAShareEquityCharacteristics)
- Extension to [Factors and Portfolios in China Market](https://github.com/mlfina/China-A-Sort)

## Prerequisite

- Read the listed papers
- [WRDS](https://wrds-web.wharton.upenn.edu) account with subscription to CRSP, Compustat and IBES.
- Python (with `polars`, `pandas`, `numpy`, `tqdm`, `wrds`, `duckdb`, `pyarrow`)

## Files

- [Characteristics list](documents/chars_summary.csv)

### Main Files (`chars_ciz/`)
- `download_data.py` — pull raw CRSP CIZ monthly/daily, Compustat funda/fundq, IBES, FF factors from WRDS (via DuckDB) into `data/raw/`
- `functions.py` — shared helpers: `ttm`, `ttm4`, `ttm12`, `chars_std`, industry classifications (`ffi49`), imputation and standardization utilities, `INPUT_PATH` / `OUTPUT_PATH` constants
- `accounting.py` — builds annual + quarterly accounting characteristics and merges with monthly CRSP (`crsp_mom`) for monthly-grain chars (`me`, `turn`, `dolvol`, `dy`, `mom*`, `seas1a`, `indmom`, …). Outputs `chars_a_accounting.parquet`, `chars_q_accounting.parquet`
- `rolling_chars.py` — rolling daily-window characteristics from CRSP daily: `beta`, `baspread`, `ill`, `maxret`, `rvar_capm`, `rvar_ff3`, `rvar_mean`, `std_dolvol`, `std_turn`, `zerotrade`. Outputs `rolling_chars.parquet`
- `sue.py` — unexpected quarterly earnings (SUE)
- `abr.py` — cumulative abnormal returns around earnings announcement dates
- `myre.py` — revisions in analysts' earnings forecasts (uses IBES)
- `merge_chars.py` — merges accounting + rolling + satellite characteristics (`sue`, `abr`, `myre`) with CRSP backfill. Outputs `chars_a_raw.parquet`, `chars_q_raw.parquet`
- `impute_rank_output.py` — reconciles annual/quarterly accounting variables by most-recent `datadate`, computes `ffi49`, lags returns one period, and writes the four final outputs (raw / imputed × no-rank / rank)
- `iclink_ciz.sas` — IBES ↔ CRSP CIZ link table (run on WRDS SAS Studio); output saved to `data/raw/iclink_ciz.csv`

### Documents
- `documents/chars_summary.csv` — current characteristic acronym, description, author, year, category

## How to use (CIZ pipeline)

All commands are run from `chars_ciz/`. Paths are configured in `functions.py` (`INPUT_PATH`, `OUTPUT_PATH`).

1. `python download_data.py` — pull raw WRDS tables into `data/raw/` (also run `iclink_ciz.sas` on WRDS to produce `iclink_ciz.csv`)
2. `python accounting.py` — build annual/quarterly accounting + monthly chars
3. `python rolling_chars.py` — daily-window rolling chars
4. `python sue.py`, `python abr.py`, `python myre.py` — satellite chars (can run in parallel; `myre.py` requires IBES + the iclink table)
5. `python merge_chars.py` — merge everything into `chars_a_raw.parquet` and `chars_q_raw.parquet`
6. `python impute_rank_output.py` — produce the four final outputs

## Outputs

### Data

The stock universe is the top three U.S. exchanges (NYSE / AMEX / NASDAQ), filtered to common equity via CRSP CIZ flags (`sharetype='NS'`, `securitytype='EQTY'`, `securitysubtype='COM'`, `usincflg='Y'`, `issuertype∈{ACOR,CORP}`, `conditionaltype='RW'`, `tradingstatusflg='A'`). The date range follows the available WRDS coverage.

Returns are shifted one period forward so that characteristics at time $t$ predict the return at $t+1$ (i.e. $ret_{t+1}$ is aligned with $chars_t$).

The four final files (all parquet) are:

1. `chars_raw_no_impute.parquet` — raw characteristic levels, missing values preserved
2. `chars_raw_imputed.parquet` — same as above with industry-median / industry-mean imputation
3. `chars_rank_no_impute.parquet` — cross-sectional rank-standardized characteristics (no imputation)
4. `chars_rank_imputed.parquet` — cross-sectional rank-standardized characteristics (imputed)

### Information Variables

- stock identifier: `gvkey`, `permno`, `ticker`, `conm`, `comnam`
- time: `datadate` (accounting), `date` (return date)
- industry: `sic`, `ffi49`
- price / size: `prc`, `shrout`, `me`, `log_me`, `lag_me`
- return: `ret` (delisting-adjusted via CRSP CIZ `mthret`)

<!-- ## Method

### Equity Characteristics

This topic is summaried by **Green Hand Zhang** and **Hou Xue Zhang**.

### Portfolio Characteristics

Portfolio charactaristics is the equal-weighted / value-weighted averge of the characteristics for all equities in the portfolio.

The portfolios includes and not limited to:

- Characteristics-sorted Portfolio, see the listed papers and also [Deep Learning in Characteristics-Sorted Factor Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3243683)
- DGTW Benchmark, see [DGTW 1997 JF](https://doi.org/10.1111/j.1540-6261.1997.tb02724.x)
- Industry portfolio -->

## Reference

### Papers

- **Dissecting Anomalies with a Five-Factor Model** by [Fama and French 2015 RFS](https://doi.org/10.1093/rfs/hhv043)
  - Define the characteristics of a portfolio as the value-weight averages (market-cap weights) of the variables for the firms in the portfolio
  - [French's Data Library](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

- **The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns** by [Green Hand Zhang 2017 RFS](https://doi.org/10.1093/rfs/hhx019)
  - [sas code from Green's website](https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view)
- **Replicating Anormalies** by [Hou Xue Zhang 2018 RFS](https://doi.org/10.1093/rfs/hhy131)
  - [Anormaly Portfolios by Zhang's website](http://global-q.org/index.html)

### Codes

- Calculate equity characteristics with SAS code, mainly refering to [SAS code by Green Hand Zhang](https://drive.google.com/file/d/0BwwEXkCgXEdRQWZreUpKOHBXOUU/view).
- Portfolio characteristics, mainly refering to [WRDS Financial Ratios Suite](https://wrds-www.wharton.upenn.edu/pages/support/research-wrds/sample-programs/wrds-sample-programs/wrds-financial-ratios-suite/) and [Variable Definition](https://wrds-www.wharton.upenn.edu/documents/793/WRDS_Industry_Financial_Ratio_Manual.pdf)

**All comments are welcome.**
