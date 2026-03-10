# Bug Report: accounting.py vs Paper Definitions

**Date**: 2026-03-09

## References
- **GHZ**: Chen & Zimmermann (2020), "Open Source Cross-Sectional Asset Pricing", Online Appendix Table 1
- **HXZ**: Hou, Xue & Zhang (2020), "Replicating Anomalies", Review of Financial Studies
- **FF2015**: Fama & French (2015), "A five-factor model"

---

## BUG 1 (CRITICAL): `ppegt_l1` undefined — `invest` calculation will crash

**Location**: Line 646-658

**Code**:
```python
# Line 654
((pl.col('ppegt') - pl.col('ppegt_l1')) +
 (pl.col('invt') - pl.col('invt_l1'))) / pl.col('at_l1').replace(0, None)
```

**Problem**: `ppegt_l1` (1-year lag of ppegt) is **never defined** — no `ppegt.shift(1).over('permno')` exists before this usage. Only `ppent_l1` is defined (line 643). The comment says "(fixed)2026-03-06: replace ppent_l1 with ppegt_l1 for consistency" but the lag column was never created.

**Paper (HXZ A.3.4, GHZ InvestPPEInv)**: "annual change in **gross property, plant, and equipment (PPEGT)** plus annual change in inventory (INVT) scaled by 1-year-lagged total assets."

**Fix**: Add `pl.col('ppegt').shift(1).over('permno').alias('ppegt_l1')` to the lag definitions at line 642-645, e.g.:
```python
data_rawa = data_rawa.with_columns([
    pl.col('ppent').shift(1).over('permno').alias('ppent_l1'),
    pl.col('invt').shift(1).over('permno').alias('invt_l1'),
    pl.col('ppegt').shift(1).over('permno').alias('ppegt_l1'),  # ADD THIS
])
```

---

## BUG 2 (MEDIUM): `chinv` uses wrong denominator `at_l2` instead of avg(at, at_l1)

**Location**: Line 714-718

**Code**:
```python
((pl.col('invt') - pl.col('invt_l1')) /
 ((pl.col('at') + pl.col('at_l2')) / 2).replace(0, None)
).alias('chinv')
```

**Problem**: Uses `at_l2` (2-year lagged total assets) in the denominator. Should be `at_l1` (1-year lag).

**Paper (GHZ, Thomas & Zhang 2002 ChInv)**: "12 month change in inventory (invt) divided by **average total assets**." Average total assets = `(at + at_l1) / 2`.

**Fix**:
```python
((pl.col('invt') - pl.col('invt_l1')) /
 ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)
).alias('chinv')
```

---

## BUG 3 (MEDIUM): `chato` (annual) uses wrong lagged denominator

**Location**: Line 590-593

**Code**:
```python
((pl.col('sale') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)) -
 (pl.col('sale_l1') / ((pl.col('at') + pl.col('at_l2')) / 2).replace(0, None))).alias('chato')
```

**Problem**: The second term computes lagged ATO as `sale_l1 / avg(at_t, at_{t-2})`. This incorrectly uses **current** at in the lagged term's average. The lagged ATO's denominator should be `avg(at_{t-1}, at_{t-2})`.

**Paper (GHZ, Soliman 2008 ChAssetTurnover)**: Change in asset turnover, where ATO = sale / avg(at). The lagged ATO should use lagged averages.

**Fix**:
```python
((pl.col('sale') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)) -
 (pl.col('sale_l1') / ((pl.col('at_l1') + pl.col('at_l2')) / 2).replace(0, None))).alias('chato')
```

---

## BUG 4 (LOW): `chadv` uses `log(xad + 1)` instead of `log(xad)`

**Location**: Line 778-780

**Code**:
```python
((pl.col('xad') + 1).log() - (pl.col('xad_l1') + 1).log()).alias('chadv')
```

**Paper (GHZ, Chemmanur & Yan 2009)**: "**Log of advertising expense (xad) minus log of advertising expense last year.**"

**Issue**: The paper specifies `log(xad) - log(xad_l1)`, not `log(xad + 1) - log(xad_l1 + 1)`. Adding 1 is a common workaround for zero values but distorts the variable for small xad values. If xad=0, the paper intends the observation to be excluded (null), not set to 0 via `log(0+1)=0`.

**Fix**:
```python
(pl.col('xad').log() - pl.col('xad_l1').log()).alias('chadv')
```

---

## BUG 5 (MEDIUM): Quarterly `acc` and `pctacc` — full audit

### 5A. Quarterly `acc` — time horizon mismatch and `dpq` error

**Location**: Line 1516-1527

**Current code**:
```python
pl.when(pl.col('oancfq').is_null())
  .then(
      ((actq - actq_l4) - (cheq - cheq_l4) - (lctq - lctq_l4) + (dlcq - dlcq_l4) + (txpq - txpq_l4) - dpq) /
      ((atq + atq_l4) / 2)
  )
  .otherwise(
      (ibq - oancfq) / ((atq + atq_l4) / 2)
  )
  .alias('acc')
```

**Problem 1 — `dpq` is quarterly, everything else is 4-quarter**: The balance-sheet branch uses 4-quarter changes for all working capital items (`actq_l4`, `cheq_l4`, `lctq_l4`, `dlcq_l4`, `txpq_l4` — annual deltas), but subtracts just `dpq` (one quarter's depreciation). This is wrong — depreciation should also cover 4 quarters. The TTM depreciation `dpq4 = ttm4('dpq')` is already computed at line 1540 and should be used instead.

**Problem 2 — time horizon mismatch between branches**:
- **Balance-sheet branch**: 4-quarter changes (≈ annual accruals) / avg(atq, atq_l4)
- **Cash-flow branch**: `ibq - oancfq` (single quarter accruals) / avg(atq, atq_l4)

These produce different magnitudes — the balance-sheet branch gives ~4× larger values than the cash-flow branch.

**Paper (Sloan 1996, GHZ Accruals)**: Both branches should use the **same time horizon**. Two valid approaches:

**Option A — Quarterly `acc`** (one quarter of accruals):
```python
# Need 1-quarter lags for balance sheet approach
data_rawq = data_rawq.with_columns([
    pl.col('actq').shift(1).over('permno').alias('actq_l1'),
    pl.col('cheq').shift(1).over('permno').alias('cheq_l1'),
    pl.col('lctq').shift(1).over('permno').alias('lctq_l1'),
    pl.col('dlcq').shift(1).over('permno').alias('dlcq_l1'),
    pl.col('txpq').shift(1).over('permno').alias('txpq_l1'),
])
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_null())
      .then(
          ((pl.col('actq') - pl.col('actq_l1')) - (pl.col('cheq') - pl.col('cheq_l1')) -
           (pl.col('lctq') - pl.col('lctq_l1')) + (pl.col('dlcq') - pl.col('dlcq_l1')) +
           (pl.col('txpq') - pl.col('txpq_l1')) - pl.col('dpq')) /
          ((pl.col('atq') + pl.col('atq').shift(1).over('permno')) / 2).replace(0, None)
      )
      .otherwise(
          (pl.col('ibq') - pl.col('oancfq')) /
          ((pl.col('atq') + pl.col('atq').shift(1).over('permno')) / 2).replace(0, None)
      )
      .alias('acc')
])
```

**Option B — Annual `acc` from quarterly data** (4 quarters of accruals, comparable to annual `acc`):
```python
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_null())
      .then(
          ((pl.col('actq') - pl.col('actq_l4')) - (pl.col('cheq') - pl.col('cheq_l4')) -
           (pl.col('lctq') - pl.col('lctq_l4')) + (pl.col('dlcq') - pl.col('dlcq_l4')) +
           (pl.col('txpq') - pl.col('txpq_l4')) - pl.col('dpq4')) /       # dpq4, not dpq
          ((pl.col('atq') + pl.col('atq_l4')) / 2).replace(0, None)
      )
      .otherwise(
          (ttm4('ibq', data_rawq) - ttm4('oancfq', data_rawq)) /          # TTM, not single quarter
          ((pl.col('atq') + pl.col('atq_l4')) / 2).replace(0, None)
      )
      .alias('acc')
])
```

**Recommendation**: Option A (quarterly) is cleaner and more natural for quarterly data. Option B requires `dpq4` and `ttm4()` calls, and is redundant with the annual `acc` already computed.

---

### 5B. Quarterly `pctacc` — redundant fallback, dead branches, and same `dpq` issue

**Location**: Line 1627-1654

**Current code**:
```python
# Line 1627-1635: REDUNDANT oancfq fallback (already done at lines 1506-1512)
data_rawq = data_rawq.with_columns([
    pl.col('wcaptq').fill_null(0).alias('wcaptq')
])
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_not_null())
      .then(pl.col('oancfq'))
      .otherwise(pl.col('ibq') + pl.col('dpq') - pl.col('wcaptq'))
      .alias('oancfq')
])

# Line 1637-1654: pctacc with 4 branches
pl.when((pl.col('oancfq').is_null()) & (pl.col('ibq') == 0))       # Branch 1: BS / 0.01
  .then(BS_formula / 0.01)
.when(pl.col('oancfq').is_null())                                    # Branch 2: BS / |ibq|
  .then(BS_formula / abs(ibq))
.when(pl.col('ibq') == 0)                                           # Branch 3: CF / 0.01
  .then((ibq - oancfq) / 0.01)
.otherwise((ibq - oancfq) / abs(ibq))                               # Branch 4: CF / |ibq|
.alias('pctacc')
```

**Problem 1 — Redundant `oancfq` fallback**: Lines 1627-1635 repeat the Level 3 fallback (`ibq + dpq - wcaptq`) that was already applied at lines 1506-1512. This overwrites `oancfq` a second time and also mutates `wcaptq` by filling nulls, which could affect later code.

**Problem 2 — Branches 1-2 are near-dead code**: After the three-level fallback (lines 1498-1512) AND the redundant second fallback (1630-1635), `oancfq` is null **only when `ibq` or `dpq` is also null** (since Level 3 = `ibq + dpq - wcaptq`). If `ibq` is null:
- Branch 1 (`oancfq.is_null() & ibq == 0`): **false** (null ≠ 0)
- Branch 2 (`oancfq.is_null()`): **true** → computes `BS / abs(ibq)` → **null** (since ibq is null)
- Result: pctacc = null regardless

So the balance-sheet branches never produce meaningful values. They are dead code after the oancfq fallback.

**Problem 3 — Same `dpq` issue in balance-sheet formula**: The BS formula in branches 1-2 uses `dpq` (1-quarter) with 4-quarter changes — same mismatch as in `acc`.

**Correct specification for `pctacc`** (Hafzalla, Lundholm & Van Winkle 2011; GHZ):
- `pctacc = (ibq - oancfq) / |ibq|`, with `0.01` floor when `ibq = 0`
- `oancfq` comes from the three-level fallback in BUG 14
- No balance-sheet branches needed

**Fix** — Remove redundant fallback and dead branches, simplify to:
```python
# pctacc — oancfq already filled by three-level fallback (BUG 14, lines 1498-1512)
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('ibq') == 0)
      .then((pl.col('ibq') - pl.col('oancfq')) / 0.01)
      .otherwise((pl.col('ibq') - pl.col('oancfq')) / pl.col('ibq').abs().replace(0, None))
      .alias('pctacc')
])
```

---

### 5C. Annual `acc` and `pctacc` — verified correct

**Annual `acc` (line 435-447)**: ✅ Correct
```
BS: (dACT - dCHE - dLCT + dDLC + dTXP - DP) / avg(AT)     — all annual
CF: (ib - oancf) / avg(AT)                                   — all annual
```

**Annual `pctacc` (line 560-579)**: ✅ Correct
```
CF: (ib - oancf) / |ib|          — with 0.01 floor when ib = 0
BS: same Sloan formula / |ib|    — fallback when oancf is null
```
Both use consistent annual time horizons. The 0.01 convention and branch ordering are correct per GHZ.

**Note on the `/ 0.01` convention when `ib == 0`**:

The `pctacc` formula is `(ib - oancf) / |ib|`. When `ib = 0`, the denominator is zero and the ratio is undefined. The code substitutes `0.01` as the denominator in this case. This is **not** from the original Hafzalla, Lundholm & Van Winkle (2011) paper — the original paper defines percent accruals as `(NI - FCF) / |NI|` without specifying a treatment for zero-income observations (such observations would naturally be excluded).

The `0.01` convention is an **implementation choice from the GHZ replication code** ([OpenSourceAP/CrossSection, `Signals/pyCode/Predictors/PctAcc.py`](https://github.com/OpenSourceAP/CrossSection/blob/master/Signals/pyCode/Predictors/PctAcc.py)):
```python
# GHZ code:
df["PctAcc"] = (df["ib"] - df["oancf"]) / np.abs(df["ib"])
df.loc[df["ib"] == 0, "PctAcc"] = (df["ib"] - df["oancf"]) / 0.01
```

**Why 0.01?** When `ib = 0`, the numerator simplifies to `(0 - oancf) = -oancf`. Dividing by `0.01` is equivalent to multiplying by 100, producing `pctacc = -oancf × 100`. The effect is:
- It avoids division by zero so zero-income observations are **retained** rather than dropped.
- It amplifies the accrual signal for these firms: a firm with zero earnings but nonzero cash flow gets a very large (positive or negative) `pctacc` value, reflecting that 100% of cash flow is unexplained by earnings.
- The choice of `0.01` (rather than, say, `0.001` or `1`) is a pragmatic convention — it produces values large enough to push these observations into extreme deciles during portfolio sorts, which is conceptually appropriate since zero-income firms with nonzero cash flow have extreme percent accruals by construction.
- After winsorization (typically at 1%/99%), these extreme values are capped, so the exact constant matters less in practice.

**References**:
- Hafzalla, N., Lundholm, R., & Van Winkle, E. M. (2011). "Percent Accruals." *The Accounting Review*, 86(1), 209–236. — Original definition: `PctAcc = (NI - FCF) / |NI|`.
- Chen, A. Y. & Zimmermann, T. (2022). "Open Source Cross-Sectional Asset Pricing." *Critical Finance Review*, 11(2), 207–264. — GHZ replication code implements the `0.01` floor for `|ib| = 0`.

---

## BUG 6 (MEDIUM): Quarterly `op` uses 4-quarter lag `beq_l4` as denominator

**Location**: Line 1549-1554

**Code**:
```python
data_rawq = data_rawq.with_columns([
    pl.col('beq').shift(4).over('permno').alias('beq_l4')
])
data_rawq = data_rawq.with_columns([
    ((ttm4('revtq', data_rawq) - ttm4('cogsq', data_rawq) - ttm4('xsgaq', data_rawq) - ttm4('xintq', data_rawq))
     / pl.col('beq_l4').replace(0, None)).alias('op')
])
```

**Problem**: Uses `beq_l4` (4-quarter-lagged book equity) as the denominator.

**Paper**:
- **HXZ A.4.12 (Ope, Fama-French 2015)**: "scaled by book equity (the denominator is **current, not lagged**, book equity)."
- **HXZ A.4.13 (Ole)**: uses **1-period-lagged** book equity.
- **HXZ A.4.14 (Oleq)**: "quarterly operating profits... scaled by **1-quarter-lagged** book equity."

None of the paper variants use 4-quarter lag.

**Fix**: Use `beq` (current) for FF2015 Ope, or `beq_l1` (1-quarter lag) for HXZ Oleq:
```python
# For Ope (FF2015):
... / pl.col('beq').replace(0, None)).alias('op')
# For Oleq (HXZ):
... / pl.col('beq').shift(1).over('permno').replace(0, None)).alias('op')
```

---

## BUG 7 (LOW): Misleading comment on `pchdepr` (line 765)

**Location**: Line 765-771

**Comment**: `# (fixed)2026-03-06: replace ppent_l1 with ppegt_l1 for consistency.`

**Actual code**: Still uses `ppent` and `ppent_l1` (which is **correct** per the paper).

**Paper (GHZ, Holthausen & Larcker 1992)**: "Annual percentage change in the ratio of depreciation (dp) to **property, plant and equipment (ppent)**."

**Fix**: Remove or correct the misleading comment. The code correctly uses `ppent`.

---

## BUG 8 (LOW): Misleading comment on `grGW` (line 806)

**Location**: Line 806

**Comment**: `# (fixed)2026-03-06: replace gdwl with gdwl_l1 for consistency.`

**Actual code** (line 808): `(pl.col('gdwl') - pl.col('gdwl_l1')) / pl.col('gdwl_l1')` — Uses both `gdwl` and `gdwl_l1`, which is the standard growth rate formula. The comment is confusing.

**Fix**: Remove or clarify the misleading comment.

---

## Verified Correct Formulas

| Variable | Line | Formula in Code | Paper Reference | Status |
|----------|------|-----------------|-----------------|--------|
| `acc` (annual) | 435-447 | `(dCA-dCHE-dCL+dDLC+dTXP-DP) / avg(at)` or `(ib-oancf) / avg(at)` | GHZ (Sloan 1996): avg total assets | ✅ Correct |
| `agr` | 455-457 | `(at - at_l1) / at_l1` | HXZ A.3.2: `at_{t-1}/at_{t-2} - 1` | ✅ Correct |
| `op` (annual) | 496-505 | `(revt - cogs - xsga - xint) / be` | HXZ A.4.12 (Ope, FF2015): current BE | ✅ Correct |
| `operprof` | 957-962 | `(revt - cogs - xsga - xint) / ceq_l1` | HXZ A.4.13 (Ole): lagged BE | ✅ Correct |
| `noa` | 607-614 | `(OA - OL) / at_l1` | HXZ A.3.5: NOA scaled by lagged at | ✅ Correct |
| `rna` | 618-623 | `oiadp / noa_raw_l1` | HXZ A.4.5: OIADP / lagged NOA | ✅ Correct |
| `pm` | 626-628 | `oiadp / sale` | HXZ A.4.5: OIADP / sale | ✅ Correct |
| `ato` | 632-634 | `sale / noa_raw_l1` | HXZ A.4.5: sale / lagged NOA | ✅ Correct |
| `gma` | 536-538 | `(revt - cogs) / at_l1` | GHZ (Novy-Marx 2013): lagged at | ✅ Correct |
| `ni` | 480-492 | `log(csho*ajex) - log(csho_l1*ajex_l1)` | HXZ A.3.10: log split-adj shares ratio | ✅ Correct |
| `pctacc` (annual) | 560-579 | Matches GHZ with 0.01 floor | GHZ (Hafzalla et al. 2011) | ✅ Correct |
| `invest` formula | 647-658 | `(Δppegt + Δinvt) / at_l1` | HXZ A.3.4 (dPia): Δppegt + Δinvt / lagged at | ✅ Correct (formula, but ppegt_l1 undefined) |
| `grltnoa` | 906-916 | `(Δppent+Δintan+Δao-Δlo+dp) / avg(at)` | HXZ A.3.6 (dLno) | ✅ Correct |
| `tang` | 1221-1225 | `(che + .715*rect + .547*invt + .535*ppent) / at` | GHZ (Hahn & Lee 2009) | ✅ Correct |
| `be` | 392-401 | `seq + txditc - ps` (with ps hierarchy) | HXZ book equity definition | ✅ Correct |

---

## BUG 9 (MEDIUM): `chmom` computes 11-month change instead of 6-month change

**Location**: Line 2067-2091

**Code**:
```python
def chmom(start, end, df):
    result_first_half = prod(1 + ret.shift(i)) - 1  for i in range(start, end)
    result_second_half = prod(1 + ret.shift(i)) - 1  for i in range(start+6, end+6)
    return result_first_half - result_second_half

chmom(1, 12, crsp_mom).alias('chmom')
```

**Analysis**: `chmom(1, 12)` computes:
- First half: `range(1, 12)` = lags [1,...,11] → **11-month** cumulative return
- Second half: `range(7, 18)` = lags [7,...,17] → another **11-month** cumulative return, shifted by 6

So `chmom = mom(1,12) - mom(7,18)`. Both windows are 11 months wide, not 6.

**Paper (Green et al. 2017)**: "Change in **six-month** momentum" = current 6-month momentum minus lagged 6-month momentum.

**Correct formula**: `chmom = mom6m_t - mom6m_{t-6}` where:
- Current 6-month momentum: cumulative return from t-1 to t-6 = `mom(1, 7)` (lags 1..6)
- Lagged 6-month momentum: cumulative return from t-7 to t-12 = `mom(7, 13)` (lags 7..12)

**Fix**: Change the call to `chmom(1, 7, crsp_mom)` and update the function, or equivalently:
```python
crsp_mom = crsp_mom.with_columns([
    (mom(1, 7, crsp_mom) - mom(7, 13, crsp_mom)).alias('chmom')
])
```

---

## BUG 10 (MEDIUM): `seas1a` uses `shift(11)` instead of `shift(12)`

**Location**: Line 2097

**Code**:
```python
pl.col('ret').shift(11).over('permno').alias('seas1a')
```

**Problem**: Uses lag 11 (return 11 months ago) instead of lag 12 (return 12 months ago = same calendar month last year).

**Paper (HXZ A.5.51, Heston & Sadka 2008)**: "Ra1 = returns in month **t−12**" — the return in the same calendar month one year ago.

**Paper (GHZ MomSeasAlt1a)**: "Average return in the **same month** in the previous year."

At monthly frequency, the return 12 months ago (same calendar month) is `ret.shift(12)`, not `ret.shift(11)`.

**Fix**:
```python
pl.col('ret').shift(12).over('permno').alias('seas1a')
```

---

## BUG 11 (MEDIUM): `indmom` uses wrong return window and weighting

**Location**: Line 2311-2312

**Code**:
```python
df_temp = data_rawa.group_by(['date', 'ffi49']).agg(pl.col('mom12m').mean().alias('indmom'))
```

**Problems**:
1. Uses `mom12m` (12-month return, t-12 to t-1) — paper says 6-month return
2. Uses `mean` (equal-weighted) — paper says market-value-weighted
3. Uses FF49 industries — paper says 2-digit SIC

**Paper (GHZ IndMom, Grinblatt & Moskowitz 1999)**: "Weighted average of firm-level **6 month buy-and-hold return**. Average is taken over **two digit industries** each month and weights are based on **market value of equity**."

**Fix**:
```python
# Compute 2-digit SIC
data_rawa = data_rawa.with_columns([
    (pl.col('sic') // 100).alias('sic2')
])
# Value-weighted 6-month industry return
df_temp = (data_rawa
    .with_columns([(pl.col('mom6m') * pl.col('me')).alias('vw_mom6m')])
    .group_by(['date', 'sic2'])
    .agg([
        pl.col('vw_mom6m').sum().alias('sum_vw'),
        pl.col('me').sum().alias('sum_me')
    ])
    .with_columns([(pl.col('sum_vw') / pl.col('sum_me')).alias('indmom')])
)
data_rawa = data_rawa.join(df_temp.select(['date', 'sic2', 'indmom']), on=['date', 'sic2'], how='left')
```

---

## BUG 12 (LOW): `mom36m` window may be incorrect

**Location**: Line 2096

**Code**:
```python
mom(12, 36, crsp_mom).alias('mom36m')
```

**Analysis**: `mom(12, 36)` uses lags [12,...,35] = 24 returns, giving cumulative return from t-12 to t-35. This covers months t-35 to t-12 (24 months).

**Paper (GHZ)**: "Stock return between months **t-36 and t-13**." This is the cumulative return from end-of-t-37 to end-of-t-13, covering 24 monthly returns at lags 13,...,36 = `mom(13, 37)`.

**Issue**: The code uses lags 12-35, while the paper specifies lags 13-36. The code includes the t-12 month and excludes the t-36 month; the paper excludes the t-12 month and includes the t-36 month.

**Fix**:
```python
mom(13, 37, crsp_mom).alias('mom36m')
```

---

## BUG 13 (LOW): `mom60m` window may be incorrect

**Location**: Line 2092

**Code**:
```python
mom(12, 60, crsp_mom).alias('mom60m')
```

**Analysis**: `mom(12, 60)` uses lags [12,...,59] = 48 returns from t-12 to t-59.

**Paper (HXZ A.2.8, De Bondt & Thaler 1985 reversal)**: "prior returns from month **t−60 to t−13**" = lags [13,...,60] = 48 returns = `mom(13, 61)`.

**Issue**: Same off-by-one shift as mom36m. Includes lag-12 (which is part of 12-month momentum) and excludes lag-60.

**Fix**:
```python
mom(13, 61, crsp_mom).alias('mom60m')
```

---

## Verified Correct Momentum/CRSP Formulas

| Variable | Line | Formula in Code | Paper Reference | Status |
|----------|------|-----------------|-----------------|--------|
| `mom12m` | 2093 | `mom(1, 12)`: lags 1..11 = t-1 to t-11 | GHZ: "return between t-12 and t-1" = 11 returns | ✅ Correct |
| `mom6m` | 2095 | `mom(1, 6)`: lags 1..5 = t-1 to t-5 | GHZ: "return between t-6 and t-1" = 5 returns | ✅ Correct (GHZ convention, not skipping t-1) |
| `mom1m` | 2094 | `ret` (current month return) | GHZ: "Stock return over the previous month" | ✅ Correct |
| `dolvol` | 2109-2110 | `log(vol_l2 * prc_l2)` | GHZ: "Log of two-month lagged vol × two-month lagged prc" | ✅ Correct |
| `turn` | 2111 | `avg(vol_l1, vol_l2, vol_l3) / (1000 * shrout)` | GHZ: "Sum of vol over 3 months / (3 × shrout)" + CIZ unit conversion | ✅ Correct |
| `dy` (monthly) | 2121 | `ttm12('mdivpay') / me` | GHZ: TTM dividends / ME | ✅ Correct |
| `mom` function | 2050-2064 | Cumulative product of (1+ret_lag) | Standard compound return | ✅ Correct |

---

---

## BUG 14 (CRITICAL): `oancfq` not in download_data.py — quarterly `acc` will crash

**Location**: Line 1489-1503 (acc), download_data.py line 143-144

**Code** (accounting.py line 1492):
```python
pl.when(pl.col('oancfq').is_null())
```

**Problem**: `oancfq` is referenced in the quarterly `acc` calculation but was **never downloaded** in `download_data.py`. Only `oancfy` (year-to-date) was downloaded (line 144). This will cause a column-not-found runtime error.

**Background**: Compustat Xpressfeed data item list does not always contain `OANCFQ` (net cash flow from operating activities, quarterly). However, some Compustat vintages do populate it. The robust approach is to download both `oancfq` and `oancfy`, then use a multi-level fallback.

**Fix (download_data.py)** — ✅ FIXED: Added `f.oancfq` and `f.wcaptq` to the `comp_fundq` query alongside `f.oancfy`:
```sql
f.oancfy, f.oancfq, f.wcaptq, f.dlttq, ...
```

**Fix (accounting.py)** — Construct a robust `oancfq` with three-level fallback:
```python
# Level 1: Use oancfq directly if available from Compustat
# Level 2: Derive from oancfy (year-to-date) — oancfq = oancfy(t) - oancfy(t-1)
data_rawq = data_rawq.with_columns([
    pl.col('oancfy').shift(1).over('permno').alias('oancfy_l1')
])
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_not_null())
      .then(pl.col('oancfq'))
      .when(pl.col('fqtr') == 1)
      .then(pl.col('oancfy'))
      .otherwise(pl.col('oancfy') - pl.col('oancfy_l1'))
      .alias('oancfq')
])
# Level 3: If still null, use ibq + dpq - wcaptq (wcaptq defaults to 0 if null)
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_not_null())
      .then(pl.col('oancfq'))
      .otherwise(pl.col('ibq') + pl.col('dpq') - pl.col('wcaptq').fill_null(0))
      .alias('oancfq')
])
```

This unified `oancfq` column can then be used directly by `acc`, `pctacc`, `pscore`, and any other variable that needs quarterly operating cash flow.

---

## BUG 15 (MEDIUM): Quarterly `pctacc` and `pscore` use `oancfy` (annual YTD) instead of quarterly `oancfq`

**Location**: Line 1604-1621 (pctacc), Line 1988/1993 (pscore p_temp2/p_temp4)

**`pctacc` code** (line 1605-1619):
```python
pl.when((pl.col('oancfy').is_null()) & (pl.col('ibq') == 0))  # oancfy = annual YTD
  ...
.when(pl.col('ibq') == 0)
  .then((pl.col('ibq') - pl.col('oancfy')) / 0.01)            # ibq is quarterly, oancfy is annual
.otherwise((pl.col('ibq') - pl.col('oancfy')) / ...)
```

**`pscore` code** (line 1988/1993):
```python
pl.when(pl.col('oancfy') > 0).then(1).otherwise(0).alias('p_temp2'),        # should use TTM oancf
pl.when(pl.col('oancfy') > pl.col('niq4')).then(1).otherwise(0).alias('p_temp4'),  # time horizon mismatch
```

**Problems**:
1. **pctacc**: Mixes quarterly `ibq` with annual YTD `oancfy` — different time horizons
2. **pscore p_temp2**: `oancfy` is YTD, not TTM — inconsistent across fiscal quarters (Q1 oancfy ≈ 1 quarter of cash flow vs Q4 oancfy ≈ 4 quarters)
3. **pscore p_temp4**: Compares `oancfy` (YTD) with `niq4` (TTM net income) — time horizon mismatch

**Note on OANCFQ**: `oancfq` and `wcaptq` are now downloaded in `download_data.py` (see BUG 14 fix). After the three-level fallback in BUG 14 (`oancfq` → derived from `oancfy` → `ibq + dpq - wcaptq`), a unified `oancfq` column is available for all downstream variables.

**Fix**: Derive `oancfq` from `oancfy` (see BUG 14), build `oancfq_filled` with fallback (see BUG 5), then:
- **pctacc**: Use `oancfq_filled` (which is `oancfq` if available, else `ibq + dpq - wcaptq`):
```python
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('ibq') == 0)
      .then((pl.col('ibq') - pl.col('oancfq_filled')) / 0.01)
      .otherwise((pl.col('ibq') - pl.col('oancfq_filled')) / pl.col('ibq').abs().replace(0, None))
      .alias('pctacc')
])
```
- **pscore**: Use `ttm4('oancfq')` (TTM quarterly operating cash flow) for both p_temp2 and p_temp4:
```python
data_rawq = data_rawq.with_columns([ttm4('oancfq', data_rawq).alias('oancfq4')])
# p_temp2:
pl.when(pl.col('oancfq4') > 0).then(1).otherwise(0).alias('p_temp2'),
# p_temp4:
pl.when(pl.col('oancfq4') > pl.col('niq4')).then(1).otherwise(0).alias('p_temp4'),
```

---

## BUG 16 (MEDIUM): `nincr` uses quarter-over-quarter comparison instead of year-over-year

**Location**: Line 1938-1972

**Code**:
```python
pl.when(pl.col('ibq') > pl.col('ibq_l1')).then(1).otherwise(0).alias('nincr_temp1'),
pl.when(pl.col('ibq_l1') > pl.col('ibq_l2')).then(1).otherwise(0).alias('nincr_temp2'),
...
```

**Problem**: The code compares `ibq` with `ibq_l1` (previous quarter), i.e., **quarter-over-quarter** increases. The paper specifies **4-quarter** (year-over-year) increases.

**Paper (GHZ)**: "Number of **4-quarter** net income (ibq) increases over the previous 2 years."

This means: count consecutive instances where `ibq_t > ibq_{t-4}`, `ibq_{t-1} > ibq_{t-5}`, etc. (each comparison is year-over-year). Over 2 years = 8 quarters, the maximum consecutive count is 8.

**Fix**:
```python
data_rawq = data_rawq.with_columns([
    pl.col('ibq').shift(4).over('permno').alias('ibq_l4'),
    pl.col('ibq').shift(5).over('permno').alias('ibq_l5'),
    ...
    pl.col('ibq').shift(11).over('permno').alias('ibq_l11'),
    pl.col('ibq').shift(12).over('permno').alias('ibq_l12'),
])
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('ibq') > pl.col('ibq_l4')).then(1).otherwise(0).alias('nincr_temp1'),
    pl.when(pl.col('ibq_l1') > pl.col('ibq_l5')).then(1).otherwise(0).alias('nincr_temp2'),
    ...  # each ibq_{t-k} > ibq_{t-k-4}
])
# Then apply same consecutive product logic
```

---

## BUG 17 (LOW): `be` calculation missing fallback when `seq` is null

**Location**: Line 390-401

**Code**:
```python
(pl.col('seq') + pl.col('txditc') - pl.col('ps')).alias('be')
```

**Problem**: Only uses `seq` (stockholders' equity). When `seq` is null, `be` becomes null even if `ceq + pstk` or `at - lt` are available.

**Paper (HXZ)**: "Stockholders' equity is the value reported by Compustat (item SEQ), if it is available. If not, we measure stockholders' equity as the book value of common equity (item CEQ) plus the par value of preferred stock (item PSTK), or the book value of assets (item AT) minus total liabilities (item LT)."

**Fix**:
```python
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('seq').is_not_null()).then(pl.col('seq'))
      .when(pl.col('ceq').is_not_null() & pl.col('pstk').is_not_null())
      .then(pl.col('ceq') + pl.col('pstk'))
      .otherwise(pl.col('at') - pl.col('lt'))
      .alias('seq_filled')
])
data_rawa = data_rawa.with_columns([
    (pl.col('seq_filled') + pl.col('txditc') - pl.col('ps')).alias('be')
])
```

---

## TODO Items Reviewed

| Location | TODO Text | Verdict |
|----------|-----------|---------|
| L135 | `@TODO: add primary_sec = [a,b,c,adr]` | Enhancement — not a bug, but could improve coverage of ADR/dual-listed stocks |
| L285 | `@Todo: check数据比例` | Data quality check — compare with/without dedup |
| L390 | `@TODO: be/beq calculation` | **BUG 17** — missing fallback for seq |
| L1406 | `@TODO: 统一Int64/Int32` | Minor type consistency — not a calculation bug |
| L1433 | `@TODO: dy_a dy_q的计算方式不同` | See **NOTE: `dy` annual vs quarterly methods** below — two different data sources and formulas, both valid per GHZ |
| L1490 | `@TODO: if oancfq not available, use oancfy` | **BUG 14** — download_data.py ✅ FIXED (added `oancfq`, `wcaptq`); accounting.py still needs three-level fallback |
| L1939 | `@TODO: check equation` | **BUG 16** — nincr uses quarter-over-quarter instead of year-over-year |
| L2048 | `@TODO: check windows 11 or 12` | Resolved: `mom(1, 12)` = 11 lags = correct for GHZ convention |
| L2089 | `@TODO: check windows 6 or 12` | **BUG 9** — chmom uses 11-month windows, should be 6-month |
| L2188 | `@TODO: double check forward_fill` | Correct — forward-fills annual accounting data to monthly frequency within each permno/datadate group |
| L2398 | `@TODO: check 0-15/0-16` | `chars_std(0, 15)` = 15 values ≈ 3.75 years. GHZ says roavol uses "4 years" = 16 quarters = `chars_std(0, 16)`. If sgrvol follows same convention, should also be `chars_std(0, 16)`. Minor discrepancy. |

---

## NOTE: `dy` (dividend yield) — annual vs quarterly/monthly methods

**TODO reference**: Line 1433 — `@TODO: dy_a dy_q的计算方式不同`

The codebase computes dividend yield (`dy`) using **two different methods** depending on the data frequency. Both are valid but use different data sources and formulas.

### Method 1: Annual `dy` (Compustat-based)

**Location**: Line 2338-2339 (in the annual merge section)

**Code**:
```python
(pl.col('dvt') / pl.col('me').replace(0, None)).alias('dy')
```

**Formula**: `dy = dvt / me`
- `dvt`: Total dividends paid (Compustat annual, cash dividends on common stock)
- `me`: Market equity (price × shares outstanding from CRSP)

**Data source**: Compustat `funda` → `dvt` field

**Characteristics**:
- Simple ratio of annual dividends to current market cap
- Available only at annual frequency (once per fiscal year)
- Directly from the accounting statement (cash flow statement or footnote)
- **Paper (GHZ DivYield, Litzenberger & Ramaswamy 1982)**: "Dividends per share (dvt / csho) divided by price (prcc_f)" — equivalent to `dvt / me`

### Method 2: Monthly `dy` (CRSP return-based, TTM)

**Location**: Line 2149-2158 (in the CRSP monthly section)

**Code**:
```python
# Step 1: Implied monthly dividend payment from return spread
(pl.col('ret') - pl.col('retx')).alias('retdy')

# Step 2: Dollar dividend = return spread × lagged market equity
(pl.col('retdy') * pl.col('me_l1')).alias('mdivpay')

# Step 3: TTM dividend yield = sum of 12 monthly dividends / current ME
(ttm12('mdivpay', crsp_mom) / pl.col('me').replace(0, None)).alias('dy')
```

**Formula**: `dy = Σ(t-12 to t-1) mdivpay_t / me_t`  where `mdivpay = (ret - retx) × me_{t-1}`
- `ret`: Total return including dividends (CRSP)
- `retx`: Return excluding dividends (CRSP, price return only)
- `ret - retx`: Implied dividend yield component for the month
- `me_l1`: Lagged market equity (to convert yield back to dollar amount)
- `ttm12()`: Trailing twelve months sum

**Data source**: CRSP monthly stock file → `mthret`, `mthretx`

**Characteristics**:
- Updated monthly with a rolling 12-month window
- Captures all distributions (regular dividends, special dividends, return of capital) as reflected in CRSP returns
- More timely than the annual Compustat measure
- **Paper (GHZ DivYield)**: The monthly version uses "sum of dividends over the past 12 months divided by market equity," where dividends are imputed from `ret - retx`

### Comparison

| Aspect | Annual (`dvt / me`) | Monthly (TTM from CRSP returns) |
|--------|---------------------|----------------------------------|
| Source | Compustat `dvt` | CRSP `ret - retx` |
| Frequency | Annual | Monthly (rolling 12-month) |
| Timeliness | Stale (up to 12 months old) | Current (updates monthly) |
| Coverage | Only firms with Compustat data | All CRSP firms with returns |
| Dividend types | Common stock cash dividends only | All distributions in CRSP returns |
| Used in | Annual cross-section (`data_rawa`) | Monthly cross-section (`crsp_mom`) |

### Quarterly `dy` (commented out)

**Location**: Line 1436-1442

The quarterly version is **commented out**. It would have used the same CRSP return-based method (`ttm12` of `mdivpay`), applied at quarterly frequency. Since the monthly version in `crsp_mom` already covers this and gets merged into the final dataset, the quarterly version in `data_rawq` is redundant and correctly left disabled.

**Verdict**: Not a bug. Both methods are valid implementations of dividend yield per GHZ. The annual version is used when constructing annual characteristics from Compustat; the monthly version is used when constructing monthly characteristics from CRSP. They serve different update frequencies in the final merged dataset.

---

## Summary

| # | Variable | Severity | Issue |
|---|----------|----------|-------|
| 1 | `invest` (annual) | **CRITICAL** | `ppegt_l1` undefined — runtime crash |
| 14 | `acc` (quarterly) | **CRITICAL** | `oancfq` not downloaded — runtime crash (download_data.py ✅ FIXED: added `oancfq`, `wcaptq`) |
| 2 | `chinv` | **MEDIUM** | Denominator uses `at_l2`, should be `at_l1` |
| 3 | `chato` (annual) | **MEDIUM** | Lagged ATO uses wrong avg assets denominator |
| 5A | `acc` (quarterly) | **MEDIUM** | Balance-sheet uses `dpq` (1-qtr) with 4-qtr changes; CF branch (1-qtr) vs BS branch (4-qtr) time mismatch |
| 5B | `pctacc` (quarterly) | **MEDIUM** | Redundant oancfq fallback, dead BS branches, same `dpq` issue; should simplify to `(ibq - oancfq) / \|ibq\|` |
| 6 | `op` (quarterly) | **MEDIUM** | Uses 4-quarter lag BE instead of current or 1-quarter lag |
| 9 | `chmom` | **MEDIUM** | Uses 11-month windows instead of 6-month windows |
| 10 | `seas1a` | **MEDIUM** | Uses `shift(11)` instead of `shift(12)` for same-month-last-year |
| 11 | `indmom` | **MEDIUM** | Wrong return window (12m vs 6m), wrong weighting (EW vs VW), wrong industry grouping |
| 15 | `pctacc`/`pscore` (quarterly) | **MEDIUM** | `oancfy` (YTD) misused — should derive `oancfq` and use TTM |
| 16 | `nincr` | **MEDIUM** | Quarter-over-quarter comparison instead of year-over-year (4-quarter) |
| 17 | `be` | LOW | Missing fallback when `seq` is null (ceq+pstk or at-lt) |
| 4 | `chadv` | LOW | `log(xad+1)` vs paper's `log(xad)` |
| 7 | `pchdepr` comment | LOW | Misleading comment (code is correct) |
| 8 | `grGW` comment | LOW | Misleading comment |
| 12 | `mom36m` | LOW | Lag window off by 1: uses lags 12-35 instead of 13-36 |
| 13 | `mom60m` | LOW | Lag window off by 1: uses lags 12-59 instead of 13-60 |
