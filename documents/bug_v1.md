# Data Quality and Bug Report: `accounting.py`

**Prepared by:** Code Review Analysis
**Date:** 2026-03-04
**Files Reviewed:** `chars_ciz/accounting.py`, `chars_ciz/functions.py`
**Outputs Examined:** `data/processed/chars_a_accounting.parquet`, `data/processed/chars_q_accounting.parquet`

---

## Executive Summary

This report documents bugs, formula errors, and data quality issues found in `accounting.py`. The script implements a quantitative finance factor construction pipeline producing annual and quarterly accounting characteristics. While the pipeline runs to completion (confirmed by `log_accounting.txt`), **ten critical bugs** affecting numerical results were identified, along with multiple moderate-severity issues and general data quality concerns. Several bugs will silently produce wrong numbers without raising errors.

---

## 1. Critical Bugs (Incorrect Numerical Results)

### Bug 1: `invest` — Mixing Gross and Net PP&E Across Time

**Location:** Lines 637–648
**Severity:** Critical

When `ppegt` (gross PP&E) is available, the formula computes:
```python
((pl.col('ppegt') - pl.col('ppent_l1')) + (pl.col('invt') - pl.col('invt_l1'))) / at_l1
```
This subtracts **lagged net PP&E** (`ppent_l1`) from **current gross PP&E** (`ppegt`). The change in gross PP&E should be `ppegt − ppegt_l1`, not `ppegt − ppent_l1`. This mixes gross and net values across time periods, producing economically meaningless investment growth rates for any firm where `ppegt` is available.

**Correct formula:**
```python
.when(pl.col('ppegt').is_null())
  .then(((ppent - ppent_l1) + (invt - invt_l1)) / at_l1)
  .otherwise(((ppegt - ppegt_l1) + (invt - invt_l1)) / at_l1)
```

---

### Bug 2: `pchgm_pchsale` — Wrong Denominator for Sales Growth

**Location:** Lines 733–738
**Severity:** Critical

```python
((sale - sale_l1) / pl.col('sale').replace(0, None))
```
The denominator for the percentage change in sales uses **current** `sale`, not **lagged** `sale_l1`. Percentage change should be `(sale − sale_l1) / sale_l1`. Using current sales deflates the growth rate for growing firms and inflates it for shrinking firms, systematically biasing the characteristic.

---

### Bug 3: `pchdepr` — Wrong Lag in Denominator

**Location:** Lines 756–761
**Severity:** Critical

```python
(dp_l1 / pl.col('ppent').replace(0, None)).replace(0, None)  # DENOMINATOR
```
The denominator normalizes the **lagged** depreciation rate by **current** `ppent` instead of **lagged** `ppent_l1`. The correct expression for the prior-year depreciation rate is `dp_l1 / ppent_l1`. Using current ppent makes the denominator endogenously correlated with the numerator.

---

### Bug 4: `grGW` — Wrong Scaling Denominator

**Location:** Lines 797–800
**Severity:** Critical

```python
((pl.col('gdwl') - pl.col('gdwl_l1')) / pl.col('gdwl').replace(0, None)).alias('grGW')
```
Goodwill growth is scaled by **current** `gdwl` instead of **lagged** `gdwl_l1`. Using the current value as denominator is non-standard and biases the growth rate downward for firms with increasing goodwill. The standard definition is `(gdwl − gdwl_l1) / gdwl_l1`.

---

### Bug 5: `nincr` — Quarterly-over-Quarterly Instead of Year-over-Year

**Location:** Lines 1915–1947
**Severity:** Critical — fundamental methodological error

The code computes:
```python
pl.when(pl.col('ibq') > pl.col('ibq_l1'))...  # sequential quarter comparison
```
The `nincr` characteristic (Barth, Elliott, Finn 1999) counts consecutive **year-over-year** earnings increases, comparing Q_t with Q_{t−4}, Q_{t−4} with Q_{t−8}, etc. The code instead compares consecutive quarters (ibq[t] vs ibq[t−1], ibq[t−1] vs ibq[t−2], ...), which captures seasonal patterns and is neither what the paper defines nor what is typically implemented in replication studies.

**Correct implementation:**
```python
pl.when(pl.col('ibq') > pl.col('ibq_l4'))...  # year-over-year
pl.when(pl.col('ibq_l4') > pl.col('ibq_l8'))...
```

---

### Bug 6: `acc` (Quarterly) — Annual Cash Flow vs Quarterly Income Mismatch

**Location:** Lines 1466–1478
**Severity:** Critical

```python
.otherwise(
    (pl.col('ibq') - pl.col('oancfy')) / avg_atq
)
```
`ibq` is **quarterly** net income, but `oancfy` is **annual** operating cash flow (`y` suffix = annual Compustat variable). Their difference is economically meaningless — subtracting a full year's cash flow from one quarter's income. The quarterly acc formula should use `oancfq` (quarterly operating cash flow) if available, or fall back to the balance-sheet accruals formula. This same mismatch propagates into quarterly `pctacc` (lines 1580–1596).

---

### Bug 7: `chmom` — Wrong Window Length (11 months instead of 6)

**Location:** Lines 2040–2064
**Severity:** Critical

```python
chmom(1, 12, crsp_mom)  # start=1, end=12
```
The function `chmom(start, end)` uses:
- First half: `range(1, 12)` → lags 1–11 → **11 months**
- Second half: `range(7, 18)` → lags 7–17 → **11 months**

The standard definition (Gutierrez & Kelley 2008; Hou, Xue, Zhang 2015) is:
- `chmom = momentum(t−1, t−6) − momentum(t−7, t−12)` — each window is **6 months**

The code should call `chmom(1, 7, ...)` to yield `range(1,7)` (lags 1–6) and `range(7,13)` (lags 7–12).

---

### Bug 8: Systematic Off-by-One in All Momentum Characteristics

**Location:** Lines 2063–2069
**Severity:** Critical

The `mom(start, end)` function iterates `range(start, end)` (exclusive end). Therefore:

| Variable | Call | Actual Lags | Expected |
|---|---|---|---|
| `mom12m` | `mom(1, 12)` | 1–11 (11 months) | 1–12 (12 months) |
| `mom6m` | `mom(1, 6)` | 1–5 (5 months) | 1–6 (6 months) |
| `mom36m` | `mom(12, 36)` | 12–35 (24 months) | 13–36 (24 months, skip 12) |
| `mom60m` | `mom(12, 60)` | 12–59 (48 months) | 13–60 (48 months, skip 12) |

All momentum windows are one period short. For `mom12m`, this produces an 11-month return mislabeled as 12-month. For `mom36m` and `mom60m`, the skip-month convention also appears to be off by one.

---

### Bug 9: `mve_f` Calculated Before `csho` Zero-Replacement

**Location:** Lines 37–46
**Severity:** Moderate-Critical

In Polars, all expressions within a single `with_columns([...])` call are evaluated using the **original** column values before any mutation. The code:
```python
comp = comp.with_columns([
    pl.when(pl.col('csho') == 0).then(None).otherwise(pl.col('csho')).alias('csho'),
    (pl.col('csho') * pl.col('prcc_f')).alias('mve_f')  # uses OLD csho
])
```
`mve_f` is computed using the **original** `csho` (including zeros) rather than the zero-replaced version. Firms with `csho=0` will have `mve_f=0` instead of `null`. Since `mve_f` is used in some analyses, this propagates zeros where nulls are expected.

---

### Bug 10: `rd` Variable — Incorrect Denominator Label

**Location:** Lines 666–679
**Severity:** Moderate

The intermediate variable is named `xrd/at_l1` but is computed as `xrd / at` (current assets, not lagged):
```python
(pl.col('xrd') / pl.col('at').replace(0, None)).alias('xrd/at_l1')  # misnaming
```
The rd condition then also uses `xrd / at` (current). If the intent was to use `at_l1` as the denominator (standard for scaling), both instances are wrong. This also means `rd` is computed using inconsistent denominators relative to `xrdint` (which correctly uses average assets).

---

## 2. Moderate Bugs (Methodological Concerns)

### Bug 11: `sgrvol` Uses 15 Lags; `stdacc`/`roavol` Use 16

**Location:** Line 2374 vs Lines 1846–1852
**Severity:** Moderate

```python
chars_std(0, 16, data_rawq, 'sacc')   # stdacc: lags 0–15 (16 quarters)
chars_std(0, 16, data_rawq, 'roa')    # roavol: lags 0–15 (16 quarters)
chars_std(0, 15, data_rawq, 'rsup')   # sgrvol: lags 0–14 (15 quarters) ← inconsistent
```
`sgrvol` uses one fewer quarter than the related measures. If 16 quarters is the intended window (as for the Mohanram m7/m8 signals), this is a bug.

---

### Bug 12: `seas1a` Uses Lag 11 Instead of Lag 12

**Location:** Line 2070
**Severity:** Moderate

```python
pl.col('ret').shift(11).over('permno').alias('seas1a')
```
The one-year seasonal return characteristic is typically defined as the return **12 months ago** (lag 12), not 11 months. Lag 11 introduces a 1-month timing error. In combination with the momentum off-by-one bug (Bug 8), the seasonal timing is further misaligned.

---

### Bug 13: Mohanram m2/m3 Inconsistency When `oancf` is Null

**Location:** Lines 1127–1163
**Severity:** Moderate

`m2` is based on `cfroa`, which falls back to `(ib + dp) / avg_at` when `oancf` is null. `m3` is based on raw `oancf`, which is null for many pre-1988 firm-years. Thus, for those firms, `m2` is computed using the proxy while `m3` is automatically 0 (null comparison evaluates to False). This means the Mohanram G-score systematically undercounts for early sample firms where the cash flow statement is unavailable.

---

### Bug 14: `beq` Uses Different Preferred Stock Variable Than Annual `be`

**Location:** Lines 1405–1413
**Severity:** Moderate

Annual `be` uses the preferred stock cascade: `pstkrv → pstkl → pstk` (with fallback logic). Quarterly `beq` uses only `pstkq` directly. For firms where `pstkq` differs materially from the annual preferred stock hierarchy, this creates inconsistency between annual and quarterly book equity measures.

---

### Bug 15: `beq` Sets Null When `seqq ≤ 0`

**Location:** Lines 1405–1408
**Severity:** Moderate

```python
pl.when(pl.col('seqq') > 0)
  .then(pl.col('seqq') + pl.col('txditcq') - pl.col('pstkq'))
  .otherwise(None)
```
The annual version computes `be = seq + txditc - ps` first and then checks if the result is positive. The quarterly version instead sets `beq = None` whenever `seqq ≤ 0`, even though `seqq + txditcq - pstkq` might still be positive. This over-suppresses observations.

---

## 3. Data Quality Issues

### DQ-1: No Winsorization

**Severity:** High
**Impact:** All output characteristics

None of the ~80+ characteristics are winsorized before writing to parquet. Extreme outliers in financial ratios (e.g., `lev`, `ep`, `sp`, `bm`) will dominate any analysis. Standard academic practice is to winsorize at the 1%/99% level cross-sectionally or set extreme values to null. The absence of winsorization makes the output unsuitable for direct use in factor research.

---

### DQ-2: `unique()` Without Specifying Subset Columns

**Location:** Lines 33–34, 178, 1291
**Severity:** High

```python
comp = comp.sort(['gvkey', 'datadate']).unique()
crsp = crsp.sort(['permno', 'date']).unique()
```
Calling `.unique()` without arguments removes only **exact duplicate rows** (all columns identical). If two rows share the same primary key (gvkey+datadate or permno+date) but differ in any other column, both are retained. The intended deduplication should specify the key columns: `.unique(subset=['gvkey', 'datadate'])`. This is particularly concerning for Compustat where data restatements create multiple rows per firm-period.

---

### DQ-3: Financial Firms Not Excluded

**Severity:** High
**Impact:** All accounting-based ratios

Financial firms (SIC 6000–6999) are included in all calculations. Accounting ratios such as `lev` (total liabilities / market equity), `noa`, `ato`, `depr`, `currat`, `quick`, `salecash` are either undefined or misleading for banks, insurance companies, and investment firms. Most factor replication studies exclude financial firms (SIC 6000–6999). Including them distorts cross-sectional industry comparisons and can create spurious factor loadings.

---

### DQ-4: No Data Quality Validation or Summary Statistics

**Severity:** High

The log file contains only:
```
Finish Annual Variables Calculation!
Finish Quarterly Variables Calculation!
```
There are no checks for: observation counts by year, null rates per characteristic, plausible value ranges, year-over-year consistency, or coverage rates by exchange. A script of this complexity should validate output at key stages.

---

### DQ-5: `sin` Stock Classification Uses Mixed NAICS/SIC Codes

**Location:** Lines 1192–1206
**Severity:** Moderate

Tobacco and alcohol are identified by **SIC codes** (`sic >= 2100`, `sic <= 2199`, etc.), while gambling is identified by **NAICS codes** (`naics == '7132'`, etc.). If `naics` is null (common for historical observations), all gambling firms will be misclassified as non-sin. Additionally:
- SIC 7993 (coin-operated amusement devices, including gambling) is not included
- SIC 7999 (amusement and recreation) is not included
- Defense/weapons (SIC 3489, 3760–3769) are excluded from `sin`, consistent with some but not all papers

---

### DQ-6: `bm_ia` and `me_ia` Use `datadate` Grouping with Monthly `me`

**Location:** Lines 2213–2225
**Severity:** Moderate

After monthly expansion, `bm = be / me` varies month-to-month (as `me` changes) but the industry adjustment groups by `datadate`:
```python
df_temp = data_rawa.group_by(['datadate', 'ffi49']).agg(pl.col('bm').mean())
```
Firms with different fiscal year ends (e.g., March vs. December) are grouped together under different `datadate` values, meaning the industry mean `bm` is computed from heterogeneous months. The `indmom` characteristic correctly uses `date` (monthly date) instead of `datadate`. The same convention should apply to `bm_ia` and `me_ia`.

---

### DQ-7: Forward-Fill Applied to All Columns Including Monthly Variables

**Location:** Lines 2169–2171
**Severity:** Moderate

```python
data_rawa = data_rawa.with_columns([
    pl.all().forward_fill().over(['permno', 'datadate'])
])
```
`pl.all()` includes monthly CRSP variables (`ret`, `me`, `prc`, `vol`, `mom12m`, etc.) in addition to accounting variables. While forward-filling within (permno, datadate) is unlikely to cause issues for most monthly CRSP series (since each month has a unique jdate), any gap in monthly data would cause the prior month's return/price/volume to be incorrectly propagated. A more conservative approach would specify only the accounting columns for forward-fill.

---

### DQ-8: `sic` Cast to Different Integer Types

**Location:** Lines 347, 1392
**Severity:** Low-Moderate

```python
# Annual
pl.col('sic').cast(pl.Int64)  # line 347
# Quarterly
pl.col('sic').cast(pl.Int32)  # line 1392
```
Inconsistent integer types for the same variable across the two datasets. While operationally benign for SIC codes (which fit in Int16), this inconsistency could cause type mismatch errors in downstream code that joins or compares annual and quarterly output.

---

### DQ-9: No Delisting Return Verification

**Location:** Lines 2018–2019
**Severity:** Moderate

The comment states: `# No need to add delisting return in the new CIZ CRSP format`. It is unclear whether the new CIZ CRSP monthly returns (`mthret`) already embed delisting returns for firms removed from the exchange. Shumway (1997) and Beaver et al. (2007) document that ignoring delisting returns introduces substantial upward bias in measured returns, particularly for value/distressed stocks. This assumption should be formally verified.

---

### DQ-10: `unique()` on CRSP After Sort Does Not Guarantee Deduplication by Key

**Location:** Lines 177–178
**Severity:** Moderate

```python
crsp = crsp.sort(['permno', 'date']).unique()
```
If a permno-date combination has two records with different prices (e.g., bid and ask prices stored separately), both are retained after `unique()`. The ME aggregation logic later handles permco-level aggregation but does not handle permno-level duplicates at this stage.

---

### DQ-11: `dy` (Dividend Yield) Computed Twice with Different Methods

**Location:** Lines 691–692 (commented out annual), Lines 2274–2277
**Severity:** Low-Moderate

The annual `dy` at line 2276:
```python
(pl.col('dvt') / pl.col('me').replace(0, None)).alias('dy')
```
uses total annual dividends from Compustat divided by current market equity. The momentum section computes a TTM dividend yield via:
```python
ttm12('mdivpay', crsp_mom) / pl.col('me')
```
These two approaches will produce different values. The annual output `chars_a` uses `dvt/me` (Compustat-based), which is simpler but may not match the return-implied dividend yield used in the momentum approach. The quarterly dataset does not include `dy` at all. There is no `dy` in the final `chars_q.select(...)` list, creating asymmetry between the two outputs.

---

### DQ-12: `sacc` (Quarterly Standardized Accruals) Uses 1-Quarter Balance Sheet Lag

**Location:** Lines 1803–1819
**Severity:** Moderate

```python
data_rawq.with_columns([
    pl.col('actq').shift(1).over('permno').alias('actq_l1'),
    ...
])
sacc_temp = (ΔCA - ΔCash) - (ΔCL - ΔDLC)  # 1-quarter differences
sacc = sacc_temp / saleq
```
`sacc` computes balance-sheet accruals using **1-quarter** lags. This means it captures quarter-to-quarter changes, which include substantial seasonal variation for firms with seasonal business models. The `acc` variable uses 4-quarter lags (`actq_l4`, etc.), which controls for seasonality. Using 1-quarter lags for `sacc` makes it incomparable to `acc` and introduces noise. The `stdacc` (standard deviation over 16 quarters of `sacc`) will therefore reflect seasonal accrual swings rather than accrual quality.

---

### DQ-13: `scf` (Standardized Cash Flow) Uses Quarterly Income, Not TTM

**Location:** Lines 1856–1868
**Severity:** Moderate

```python
((pl.col('ibq') / pl.col('saleq').replace(0, None)) - pl.col('sacc')).alias('scf')
```
`scf` mixes `sacc` (which uses 1-quarter balance sheet changes) with `ibq/saleq` (a single-quarter ratio). For `stdcf` (standard deviation of `scf` over 16 quarters) to be meaningful, the cash flow measure should be on the same time scale as the accrual measure.

---

## 4. Summary Table

| # | Variable | Issue | Severity |
|---|---|---|---|
| 1 | `invest` | `ppegt - ppent_l1` should be `ppegt - ppegt_l1` | Critical |
| 2 | `pchgm_pchsale` | Sales growth denominator is current `sale`, not `sale_l1` | Critical |
| 3 | `pchdepr` | Depreciation rate denominator uses current `ppent`, not `ppent_l1` | Critical |
| 4 | `grGW` | Goodwill growth denominator is current `gdwl`, not `gdwl_l1` | Critical |
| 5 | `nincr` | Sequential quarterly comparison instead of year-over-year | Critical |
| 6 | `acc` (quarterly) | `oancfy` (annual) vs `ibq` (quarterly) frequency mismatch | Critical |
| 7 | `chmom` | 11-month windows instead of 6-month per standard definition | Critical |
| 8 | `mom12m`/`6m`/`36m`/`60m` | Systematic off-by-one in all momentum windows | Critical |
| 9 | `mve_f` | Calculated before `csho` zero-replacement in same `with_columns` | Moderate |
| 10 | `rd` | Variable named `xrd/at_l1` but uses current `at`, not `at_l1` | Moderate |
| 11 | `sgrvol` | 15 lags vs 16 lags for `stdacc`/`roavol` | Moderate |
| 12 | `seas1a` | Uses lag 11 instead of lag 12 | Moderate |
| 13 | `m2`/`m3` | Inconsistency when `oancf` is null | Moderate |
| 14 | `beq` | Uses `pstkq` only; annual `be` uses fallback hierarchy | Moderate |
| 15 | `beq` | Sets null when `seqq ≤ 0`, not when result ≤ 0 | Moderate |
| 16 | No winsorization | All characteristics may contain extreme outliers | High DQ |
| 17 | `unique()` | No subset specified; may retain key duplicates | High DQ |
| 18 | Financial firms | Not excluded from any calculations | High DQ |
| 19 | No validation | No coverage stats or sanity checks in output | High DQ |
| 20 | `sin` stocks | Mixed NAICS/SIC; incomplete gambling classification | Moderate DQ |
| 21 | `bm_ia`/`me_ia` | `datadate` grouping inconsistent with monthly `me` | Moderate DQ |
| 22 | Forward-fill | `pl.all()` includes monthly price/return variables | Moderate DQ |
| 23 | `sic` cast | Int64 annual vs Int32 quarterly | Low DQ |
| 24 | Delisting returns | CIZ format assumption not verified | Moderate DQ |
| 25 | `dy` | Defined twice with different methods; absent from quarterly | Low DQ |
| 26 | `sacc` | 1-quarter accrual lag vs 4-quarter for `acc` | Moderate DQ |

---

## 5. Recommended Priority for Fixes

**Immediate (affects factor validity):**
1. Fix `nincr` to use year-over-year comparisons (Bug 5)
2. Fix `chmom` window lengths (Bug 7)
3. Fix momentum off-by-one errors (Bug 8)
4. Fix quarterly `acc`/`pctacc` `oancfy` vs `oancfq` mismatch (Bug 6)
5. Fix `invest` to use `ppegt - ppegt_l1` (Bug 1)

**Short-term (formula corrections):**
6. Fix `pchgm_pchsale` denominator (Bug 2)
7. Fix `pchdepr` denominator (Bug 3)
8. Fix `grGW` denominator (Bug 4)
9. Fix `sgrvol` to use 16 lags (Bug 11)
10. Add winsorization (DQ-1)

**Medium-term (data quality):**
11. Add `unique(subset=...)` throughout (DQ-2)
12. Exclude or flag financial firms (DQ-3)
13. Add output validation and summary statistics (DQ-4)
14. Verify delisting return treatment in CIZ format (DQ-9)

---

*End of Report*
