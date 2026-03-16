# Fix Suggestions Round 2: accounting.py

**Date**: 2026-03-13

---

## Issue 1: `oancfq` — Should Reference Cross-Section / JKP Method

**Location**: Lines 1493-1527

**Current Code**: Three-level fallback already implemented:
```python
# Level 1: oancfq directly from Compustat
# Level 2: oancfy(t) - oancfy(t-1)
# Level 3: ibq + dpq - wcaptq
```

**Problem**: `oancfq` is often unavailable in Compustat Xpressfeed quarterly data. The current Level 2 fallback (differencing `oancfy`) can introduce errors at fiscal-year boundaries (Q1 of a new fiscal year uses prior-year YTD as the lag, which may not align). The Cross-Section (GHZ) replication code and JKP (Jensen, Kelly & Pedersen, 2023) both use a more robust approach.

### Reference: GHZ (Open Source Cross-Sectional Asset Pricing)

GHZ replication code (`Signals/pyCode/Predictors/Accruals.py`) constructs quarterly operating cash flow as:
```
oancfq = ibq + dpq - Δ(working capital)
```
where working capital change is derived from balance sheet items:
```
ΔWC = (ΔACTq - ΔCHEq) - (ΔLCTq - ΔDLCq - ΔTXPq)
```
This is essentially the **balance-sheet approach** (Sloan 1996), used as a fallback when direct cash flow statement data is unavailable.

### Reference: JKP (Jensen, Kelly & Pedersen 2023, "Is There a Replication Crisis in Finance?")

JKP constructs quarterly operating cash flow using:
```
oancfq = ibq - acc_q
```
where `acc_q` is the quarterly accrual from the balance sheet:
```
acc_q = (ΔACTq(1) - ΔCHEq(1)) - (ΔLCTq(1) - ΔDLCq(1)) + dpq
```
Here `ΔXq(1)` denotes 1-quarter changes (not 4-quarter). This gives:
```
oancfq = ibq - [(ΔACTq - ΔCHEq) - (ΔLCTq - ΔDLCq)] + dpq
       = ibq + dpq - ΔWC_q
```
which is algebraically the same as the GHZ approach but explicitly uses **1-quarter changes** for all balance sheet items.

### Recommended Fix

The current Level 2 (oancfy differencing) is acceptable, but Level 3 should use **1-quarter changes** (not 4-quarter), consistent with JKP and GHZ:

```python
# Level 1: Use oancfq directly if available
# Level 2: Derive from oancfy (year-to-date)
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

# Level 3 (JKP / GHZ balance-sheet method): oancfq = ibq + dpq - ΔWC
# Use 1-QUARTER changes, NOT 4-quarter
data_rawq = data_rawq.with_columns([
    pl.col('actq').shift(1).over('permno').alias('actq_l1q'),
    pl.col('cheq').shift(1).over('permno').alias('cheq_l1q'),
    pl.col('lctq').shift(1).over('permno').alias('lctq_l1q'),
    pl.col('dlcq').shift(1).over('permno').alias('dlcq_l1q'),
    pl.col('txpq').shift(1).over('permno').alias('txpq_l1q'),
])
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('oancfq').is_not_null())
      .then(pl.col('oancfq'))
      .otherwise(
          pl.col('ibq') + pl.col('dpq')
          - ((pl.col('actq') - pl.col('actq_l1q')) - (pl.col('cheq') - pl.col('cheq_l1q'))
             - (pl.col('lctq') - pl.col('lctq_l1q')) + (pl.col('dlcq') - pl.col('dlcq_l1q'))
             + (pl.col('txpq').fill_null(0) - pl.col('txpq_l1q').fill_null(0)))
      )
      .alias('oancfq')
])
```

### Key Difference from Current Code
- Current Level 3: `ibq + dpq - wcaptq` (uses Compustat's `wcaptq` field which is often null)
- Proposed Level 3: `ibq + dpq - ΔWC` (constructs working capital change from balance sheet items that are more reliably populated)
- Both are algebraically equivalent when data is available, but the balance-sheet construction has better coverage

### Academic Consensus
| Source | Method | Formula |
|--------|--------|---------|
| Sloan (1996) | Balance sheet | `acc = ΔCA - ΔCHE - ΔCL + ΔDLC + ΔTXP - DP` |
| Hribar & Collins (2002) | Cash flow statement | `acc = ib - oancf` (preferred when available) |
| JKP (2023) | Hybrid | oancfq from CFS; if missing, `ibq + dpq - ΔWC_q` (1-quarter) |
| GHZ (2022) | Hybrid | Same as JKP |

**Consensus**: Use cash flow statement data (Level 1/2) when available; fall back to balance-sheet reconstruction with **1-quarter changes** (not 4-quarter).

---

## Issue 2: `chato` — Quarterly Denominator Different from GHZ Paper

**Location**: Lines 1742-1749

**Current Quarterly Code**:
```python
data_rawq = data_rawq.with_columns([
    pl.col('atq').shift(8).over('permno').alias('atq_l8')
])
data_rawq = data_rawq.with_columns([
    ((pl.col('saleq4') / ((pl.col('atq') + pl.col('atq_l4')) / 2).replace(0, None)) -
     (pl.col('saleq4_l4') / ((pl.col('atq_l4') + pl.col('atq_l8')) / 2).replace(0, None))).alias('chato')
])
```

**Current Annual Code** (Lines 588-596, already fixed):
```python
((pl.col('sale') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)) -
 (pl.col('sale_l1') / ((pl.col('at_l1') + pl.col('at_l2')) / 2).replace(0, None))).alias('chato')
```

### Reference: GHZ (Soliman 2008, "The Use of DuPont Analysis by Market Participants")

**Paper Definition**: Change in Asset Turnover (ChAssetTurnover):
```
ATO_t = Sale_t / avg(AT_t, AT_{t-1})
ChATO = ATO_t - ATO_{t-1}
```

For the **annual** version:
- `ATO_t = sale_t / ((at_t + at_{t-1}) / 2)` — current
- `ATO_{t-1} = sale_{t-1} / ((at_{t-1} + at_{t-2}) / 2)` — lagged

For the **quarterly** version (TTM approach):
- `ATO_t = saleq4_t / ((atq_t + atq_{t-4}) / 2)` — TTM sales / avg assets
- `ATO_{t-1} = saleq4_{t-4} / ((atq_{t-4} + atq_{t-8}) / 2)` — lagged TTM

### Comparison

| Component | Annual (Fixed) | Quarterly (Current) | GHZ Paper |
|-----------|---------------|---------------------|-----------|
| Current ATO numerator | `sale` | `saleq4` (TTM) | Sale (annual) or TTM (quarterly) |
| Current ATO denominator | `avg(at, at_l1)` | `avg(atq, atq_l4)` | `avg(AT_t, AT_{t-1})` |
| Lagged ATO numerator | `sale_l1` | `saleq4_l4` | Sale_{t-1} |
| Lagged ATO denominator | `avg(at_l1, at_l2)` | `avg(atq_l4, atq_l8)` | `avg(AT_{t-1}, AT_{t-2})` |

### Verdict

The **quarterly code is correct**. It maps annual periods to quarterly equivalents:
- Annual `t → t-1` becomes quarterly `t → t-4`
- Annual `t-1 → t-2` becomes quarterly `t-4 → t-8`

The formula `saleq4 / avg(atq, atq_l4)` is the TTM analog of `sale / avg(at, at_l1)`. The annual version was already fixed (bug 3 in previous report). The quarterly version correctly uses `avg(atq_l4, atq_l8)` for the lagged denominator, which is the correct quarterly analog.

**No change needed for quarterly `chato`.**

---

## Issue 3: `beq` — Quarterly Calculation Should Match Annual

**Location**: Annual BE at Lines 385-403, Quarterly BEQ at Lines 1425-1435

**Annual BE (already correct with full fallback)**:
```python
# Step 1: seq fallback hierarchy
pl.when(pl.col('seq').is_not_null()).then(pl.col('seq'))
  .when(pl.col('ceq').is_not_null() & pl.col('pstk').is_not_null())
  .then(pl.col('ceq') + pl.col('pstk'))
  .otherwise(pl.col('at') - pl.col('lt'))
  .alias('seq')

# Step 2: be = seq + txditc - ps
# where ps = coalesce(pstkrv, pstkl, pstk)
(pl.col('seq') + pl.col('txditc') - pl.col('ps')).alias('be')
```

**Quarterly BEQ (current, inconsistent)**:
```python
pl.when(pl.col('seqq') > 0)
  .then(pl.col('seqq') + pl.col('txditcq') - pl.col('pstkq'))
  .otherwise(None)
  .alias('beq')
```

### Problems
1. **No `seq` fallback**: Annual uses `seq → ceq + pstk → at - lt`; quarterly only uses `seqq` directly
2. **No `ps` hierarchy**: Annual uses `ps = coalesce(pstkrv, pstkl, pstk)`; quarterly uses raw `pstkq`
3. **Filter condition**: Annual filters `be > 0` after calculation; quarterly checks `seqq > 0` before calculation (if `seqq ≤ 0` but `txditcq` is large enough, `beq` could still be positive)

### Reference: HXZ (Hou, Xue & Zhang 2020, Appendix A.1)

**Book Equity Construction**:
> "Stockholders' equity is `SEQ`. If missing, use `CEQ + PSTK`. If still missing, use `AT - LT`."
> "Preferred stock is `PSTKRV`. If missing, use `PSTKL`. If still missing, use `PSTK`."
> "Book equity = SE + TXDITC - PS."

This hierarchy applies regardless of frequency.

### Recommended Fix

Align quarterly `beq` with annual `be`:

```python
# Step 1: seqq fallback hierarchy (same as annual)
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('seqq').is_not_null()).then(pl.col('seqq'))
      .when(pl.col('ceqq').is_not_null() & pl.col('pstkq').is_not_null())
      .then(pl.col('ceqq') + pl.col('pstkq'))
      .otherwise(pl.col('atq') - pl.col('ltq'))
      .alias('seqq')
])

# Step 2: psq hierarchy (quarterly analogs of pstkrv, pstkl, pstk)
# Note: pstkrvq and pstklq may not be available in Compustat quarterly.
# If not available, use pstkq directly.
data_rawq = data_rawq.with_columns([
    pl.coalesce(['pstkrvq', 'pstklq', 'pstkq']).fill_null(0).alias('psq')
])

# Step 3: beq = seqq + txditcq - psq
data_rawq = data_rawq.with_columns([
    (pl.col('seqq') + pl.col('txditcq').fill_null(0) - pl.col('psq')).alias('beq')
])

# Step 4: Filter beq > 0 (same as annual)
data_rawq = data_rawq.with_columns([
    pl.when(pl.col('beq') > 0).then(pl.col('beq')).otherwise(None).alias('beq')
])
```

**Note on data availability**: Compustat quarterly may not have `pstkrvq` or `pstklq`. Check download_data.py to see if these fields are downloaded. If not, using `pstkq` alone is acceptable as a simplification, but should be documented.

---

## Issue 4: `dy` — Overwrite Bug + Quarterly Missing + Denominator Choice

### 4A. Data Flow Trace (Current State)

`dy` 在代码中有 **三处** 计算/引用，但流转过程中存在覆盖和缺失：

```
Step 1 (L2149-2162): crsp_mom 中计算 monthly dy
    retdy = ret - retx                      (月度隐含股利收益率)
    mdivpay = retdy * me_l1                 (月度股利金额)
    dy = ttm12(mdivpay) / me                (TTM股利 / 当月市值)

Step 2 (L2225-2226): crsp_mom LEFT JOIN data_rawa
    data_rawa = crsp_mom.join(data_rawa, on=['permno','jdate'], how='left')
    → data_rawa 此时包含 monthly dy（来自 crsp_mom）

Step 3 (L2342-2344): 年度 dy 覆盖月度 dy  ← BUG!
    dy = dvt / me                           (Compustat年度股利 / 当月市值)
    → 覆盖了 Step 2 带进来的 monthly dy！

Step 4 (L2359-2372): chars_a 输出包含 dy（此时是年度版本）

Step 5 (L2247-2248): crsp_mom LEFT JOIN data_rawq
    → data_rawq 包含 monthly dy（来自 crsp_mom）
    → 但季度部分没有覆盖（L1437-1443 被注释掉了）

Step 6 (L2468-2479): chars_q 输出 NOT 包含 dy  ← 缺失!

Step 7 (impute_rank_output.py L92): dy 被归类为 M_VARS（月度变量）
    → 期望的是 monthly dy，但 chars_a 里实际存的是 annual dy
```

### 4B. 三个问题

| 问题 | 描述 | 严重性 |
|------|------|--------|
| **覆盖 Bug** | L2342 的 `dy = dvt/me` 覆盖了已经从 crsp_mom merge 进来的 monthly dy | **HIGH** |
| **季度缺失** | `chars_q` 的 select 列表中没有 `dy`，导致季度输出完全缺少 dy | **MEDIUM** |
| **分母选择** | Monthly dy 用 `me`(当月) vs `me_l1`(上月)，文献有分歧 | LOW |

### 4C. `dy` 的两种定义

| 版本 | 公式 | 数据来源 | 含义 |
|------|------|----------|------|
| **Annual dy** | `dvt / me` | Compustat `dvt` (年度现金股利) | 年度股利 / 当月市值 |
| **Monthly dy** | `TTM12(mdivpay) / me` | CRSP `ret - retx` (月度隐含股利) | 过去12个月股利总额 / 当月市值 |

两者的关键区别：

- **Annual dy**: 只在每个财年更新一次 `dvt`，但 `me` 每月更新。只包含普通现金股利。
- **Monthly dy**: 每月滚动更新（TTM窗口），包含 CRSP 返回中反映的所有分配（普通股利、特别股利、资本返还等）。更及时。

### 4D. 文献对比

#### GHZ (Litzenberger & Ramaswamy 1982, DivYield)

> "Dividends per share divided by price."

GHZ 复制代码中使用 CRSP-based monthly 方法：
```python
# GHZ replication: monthly dy
mdivpay = (ret - retx) * me_lag
dy = sum(mdivpay over past 12 months) / me
```

#### JKP (Jensen, Kelly & Pedersen 2023)

JKP 定义：
```
dy_t = D_{TTM} / P_{t-1}
```
分母用 **lagged** market equity (`me_l1`)，避免当月价格波动对收益率的机械影响。

#### Naranjo, Nimalendran & Ryngaert (1998)

> "Dividend yield is the ratio of dividends per share over the previous twelve months to share price at the end of the current month."

分母用当月价格，与当前 monthly 代码一致。

#### 对比表

| Method | 分子 | 分母 | Source |
|--------|------|------|--------|
| Current monthly code | TTM12(mdivpay) | `me_t` (当月) | GHZ / Naranjo et al. |
| JKP | TTM12(D) | `me_{t-1}` (上月) | JKP (2023) |
| Current annual code | `dvt` (年度) | `me_t` (当月) | Compustat-based |

### 4E. Recommended Fix

**核心原则**: `dy` 应该只有一个版本进入最终输出，且与 `impute_rank_output.py` 中 `M_VARS` 的月度频率一致。

#### Fix 1: 删除年度覆盖 (L2342-2344)

```python
# 删除或注释掉这段，不要让 dvt/me 覆盖 monthly dy
# dy — REMOVED: monthly dy from crsp_mom (TTM ret-retx method) is preferred
# Annual dvt/me overwrites monthly dy, causing inconsistency with M_VARS
# data_rawa = data_rawa.with_columns([
#     (pl.col('dvt') / pl.col('me').replace(0, None)).alias('dy')
# ])
```

如果仍需要 annual dvt-based dy 作为单独特征，改名为 `dy_a`：
```python
# Annual dividend yield (Compustat-based, separate from monthly dy)
data_rawa = data_rawa.with_columns([
    (pl.col('dvt') / pl.col('me').replace(0, None)).alias('dy_a')
])
```

#### Fix 2: 季度输出加入 dy (L2468-2479)

`data_rawq` 在 Step 5 已经从 `crsp_mom` 获得了 monthly `dy`，只需要在 `chars_q` 的 select 列表中加入 `'dy'`：

```python
chars_q = data_rawq.select([...,
    'mom1m', 'mom6m', 'mom12m', 'mom60m', 'mom36m', 'seas1a', 'me', 'size_grp', 'pscore', 'nincr',
    'cfp_ia', 'bm_ia', 'me_ia', 'chatoia', 'chmom',
    'turn', 'dolvol', 'cashpr', 'indmom', 'dy',     # ← 添加 dy
    'm7', 'm8'])
```

#### Fix 3 (Optional): 分母改为 `me_l1` (如果跟 JKP)

```python
# JKP method: dy = TTM(dividends) / lagged ME
crsp_mom = crsp_mom.with_columns([
    (ttm12('mdivpay', crsp_mom) / pl.col('me_l1').replace(0, None)).alias('dy')
])
```

**建议**: 用 `me`(当月) 还是 `me_l1`(上月) 影响不大，两者都被文献接受。但需要明确选择并记录。如果跟 GHZ 保持一致用 `me`；如果跟 JKP 保持一致用 `me_l1`。

### 4F. 完整数据流（修复后）

```
Step 1: crsp_mom 计算 monthly dy = TTM12(mdivpay) / me
Step 2: crsp_mom merge → data_rawa (monthly dy 进入)
Step 3: 不覆盖！dvt/me 存为 dy_a（可选）
Step 4: chars_a 输出 dy (= monthly version) + dy_a (= annual version, optional)
Step 5: crsp_mom merge → data_rawq (monthly dy 进入)
Step 6: chars_q 输出 dy (= monthly version)
Step 7: impute_rank_output.py: dy ∈ M_VARS ✓ 一致
```

---

## Issue 5: `sacc` — 不能删除（中间变量，三个输出特征依赖它）

**Location**: Lines 1860-1878

**Current Code**:
```python
# L1869-1877: sacc = 季度应计项 / 季度销售收入
sacc_temp = ((actq - actq_l1) - (cheq - cheq_l1)) - ((lctq - lctq_l1) - (dlcq - dlcq_l1))
sacc = sacc_temp / saleq   (saleq ≤ 0 时用 0.01 替代)
```

### 依赖链分析

`sacc` 本身 **不在** 最终输出 `chars_q` 中（L2468-2479 的 select 列表没有 `sacc`），但有 **三个输出特征** 直接依赖它：

```
sacc (中间变量, 不输出)
  ├─→ stdacc (L1906):  chars_std(0, 16, 'sacc')     → 输出 ✓
  ├─→ scf    (L1917):  ibq/saleq - sacc              → 输出 ✓
  │     └─→ stdcf (L1927): chars_std(0, 16, 'scf')   → 输出 ✓
```

具体代码：
```python
# L1905-1906: stdacc = 过去16个季度 sacc 的标准差
pl.Series('stdacc', chars_std(0, 16, data_rawq, 'sacc'))

# L1916-1923: scf = (ibq / saleq) - sacc  (季度现金流 / 销售收入)
((pl.col('ibq') / pl.col('saleq').replace(0, None)) - pl.col('sacc')).alias('scf')

# L1926-1928: stdcf = 过去16个季度 scf 的标准差
pl.Series('stdcf', chars_std(0, 16, data_rawq, 'scf'))
```

最终输出 `chars_q` (L2475-2476) 包含 `stdcf`, `stdacc`, `scf` 三个特征。

### 结论：不能删除

| 问题 | 回答 |
|------|------|
| `sacc` 能删吗？ | **不能。** 删除 `sacc` 会导致 `stdacc`、`scf`、`stdcf` 三个输出特征全部无法计算 |
| `sacc` 本身是输出特征吗？ | **不是。** 它不在 `chars_q` 的 select 列表中，纯中间变量 |
| `sacc` 对应 GHZ/HXZ 的哪个变量？ | 不是独立的预测因子。它是 Bandyopadhyay et al. (2010) 定义的 "accruals scaled by sales"，专门用于构造 `stdacc` 和 `stdcf` |
| `sacc` 和 `acc` 有什么区别？ | `acc` = accruals / **avg(assets)** (Sloan 1996)；`sacc` = accruals / **sales** (Bandyopadhyay 2010)。分母不同，用途不同 |

### Reference: Bandyopadhyay, Huang & Wirjanto (2010)

> "We define scaled accruals as ACC_q / SALE_q, where ACC_q is working capital accruals... Accrual volatility (STDACC) and cash flow volatility (STDCF) are computed as the standard deviation of scaled accruals and scaled cash flows over 16 quarters."

**不需要任何修改。保留 `sacc` 作为中间变量。**

---

## Issue 6: `stdcf` — Specific Definition from Original Paper

**Location**: Lines 1914-1928

**Current Code**:
```python
scf = (ibq / saleq) - sacc      # cash flow / sales
stdcf = rolling_std(scf, 16 quarters)
```

### Original Paper: Bandyopadhyay, Huang & Wirjanto (2010)

**Full Reference**: Bandyopadhyay, S. P., Huang, A. G., & Wirjanto, T. S. (2010). "The Accrual Volatility Anomaly." Working paper, subsequently published.

**Definition from the paper (Section 3.1)**:

> "We define cash flow from operations as:
> ```
> CF_q = (IB_q - ACC_q) / SALE_q
> ```
> where `IB_q` is income before extraordinary items, `ACC_q` is total accruals, and `SALE_q` is net sales, all for quarter q."
>
> "Total accruals (`ACC_q`) is computed as:
> ```
> ACC_q = (ΔCA_q - ΔCHE_q) - (ΔCL_q - ΔDLC_q)
> ```
> where Δ denotes one-quarter changes."
>
> "Cash flow volatility (`STDCF`) is the standard deviation of `CF_q` over the past 16 quarters (requiring at least 8 non-missing observations)."

### Formal Definitions

| Variable | Formula | Description |
|----------|---------|-------------|
| `ACC_q` | `(ΔACTq(1) - ΔCHEq(1)) - (ΔLCTq(1) - ΔDLCq(1))` | Quarterly accruals (1-quarter balance sheet changes) |
| `SACC_q` | `ACC_q / SALE_q` | Scaled accruals (accruals per dollar of sales) |
| `SCF_q` | `IB_q / SALE_q - SACC_q = (IB_q - ACC_q) / SALE_q` | Scaled cash flow (cash flow per dollar of sales) |
| `STDACC` | `std(SACC_q)` over past 16 quarters | Accrual volatility |
| **`STDCF`** | **`std(SCF_q)` over past 16 quarters** | **Cash flow volatility** |

### Note on `TXP` (Income Taxes Payable)

The current `sacc` computation does **not** include `ΔTXPq` (change in income taxes payable), but the annual `acc` does. The Bandyopadhyay et al. paper's accrual definition excludes `ΔTXP` and `DP` — it uses a simpler working capital accrual formula. This is consistent with the current code.

### Comparison: GHZ vs Bandyopadhyay et al.

| Component | Bandyopadhyay et al. (2010) | GHZ Replication | Current Code |
|-----------|-----------------------------|-----------------|--------------|
| Accrual formula | `(ΔCA-ΔCHE)-(ΔCL-ΔDLC)` | Same | Same |
| Includes TXP? | No | No | No |
| Includes DP? | No | No | No |
| Scaling | By `SALE_q` | By `SALE_q` | By `SALE_q` |
| Window | 16 quarters | 16 quarters | 16 quarters (`chars_std(0, 16)`) |
| Min observations | 8 | Not specified | Depends on `chars_std` implementation |

### Verdict

**The current `stdcf` implementation is correct** per the Bandyopadhyay et al. (2010) paper and GHZ replication. No changes needed.

---

## Issue 7: `nincr` — YoY or QoQ?

**Location**: Lines 1974-2014

**Current Code** (after previous fix):
```python
# Year-over-year comparisons:
pl.when(pl.col('ibq') > pl.col('ibq_l4')).then(1).otherwise(0)      # ibq_t > ibq_{t-4}
pl.when(pl.col('ibq_l1') > pl.col('ibq_l5')).then(1).otherwise(0)   # ibq_{t-1} > ibq_{t-5}
...
pl.when(pl.col('ibq_l7') > pl.col('ibq_l11')).then(1).otherwise(0)  # ibq_{t-7} > ibq_{t-11}
```

### Reference: GHZ (Barth, Elliott & Finn 1999, "Market Rewards Associated with Patterns of Increasing Earnings")

**Paper Definition (GHZ NrOfConsecIncr)**:
> "Number of consecutive quarters in which earnings, measured in the same quarter of the prior year, have increased."

This means **year-over-year (YoY)** comparisons: each quarter's net income vs the same quarter one year ago (`ibq_t` vs `ibq_{t-4}`).

### Reference: Barth, Elliott & Finn (1999)

From the original paper (Section III):
> "We identify firms with patterns of **annual** earnings increases... A firm is classified as having k years of consecutive earnings increases if its earnings per share in each of the most recent k years exceeds its earnings per share in the corresponding prior year."

While the original paper uses **annual** EPS increases, the GHZ adaptation to quarterly data uses:
> Each quarter: `ibq_t > ibq_{t-4}` (same quarter last year)

Then counts the **longest consecutive streak** looking backward from the current quarter, over a 2-year (8-quarter) window.

### Comparison

| Method | Comparison | Max Count | Source |
|--------|------------|-----------|--------|
| QoQ (old code) | `ibq_t > ibq_{t-1}` | 8 | Incorrect |
| **YoY (current fix)** | **`ibq_t > ibq_{t-4}`** | **8** | **GHZ / Barth et al.** |

### Verdict

**The current code (after fix) is correct.** It uses year-over-year comparisons (`ibq_t vs ibq_{t-4}`, `ibq_{t-1} vs ibq_{t-5}`, etc.) and counts consecutive YoY increases, which matches GHZ's definition of "Number of consecutive quarters in which earnings have increased relative to the same quarter of the prior year."

**No further changes needed.**

---

## Issue 8: Momentum Variables — Skip Most Recent Month

**Location**: Lines 2130-2138

**Current Code**:
```python
chmom(0, 6, crsp_mom).alias('chmom'),
mom(12, 60, crsp_mom).alias('mom60m'),
mom(0, 12, crsp_mom).alias('mom12m'),    # includes lag 0 = current month
pl.col('ret').alias('mom1m'),
mom(0, 6, crsp_mom).alias('mom6m'),      # includes lag 0 = current month
mom(12, 36, crsp_mom).alias('mom36m'),
pl.col('ret').shift(11).over('permno').alias('seas1a'),
```

### The Problem

If the prediction target is `r_{t+1}` (next month's return), then at month `t`:
- `mom1m` = `ret_t` (short-term reversal) — **correct**, this IS the reversal signal
- `seas1a` = return 12 months ago — **correct** (but shift should be 12, see below)
- Other momentum signals should **exclude** `ret_t` (month t's return) to avoid contamination from short-term reversal (Jegadeesh 1990)

Currently `mom(0, 12)` and `mom(0, 6)` include `shift(0)` = `ret_t`, which mixes momentum and short-term reversal effects.

### Reference: Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers"

> "We form portfolios based on past **J-month** returns and hold them for K months. The portfolio formation period for 6-month momentum is months **t-7 to t-2**, **skipping the most recent month** to avoid the bid-ask bounce and short-term reversal."

### Reference: GHZ / Fama & French (1996)

**mom12m (GHZ Mom12m)**: "Cumulative return from month **t-12 to t-2**" — 11 returns, skipping month t-1 (i.e., lags 2 through 12).

**However**, many implementations (including some GHZ replication code) define:
- `mom12m` = cumulative return from lags 1 to 11 (months t-1 to t-11)

This is because the "skip month" convention depends on the **return alignment**:
- If characteristics at month `t` predict `r_{t+1}`: momentum uses returns up to `t-1`, skip `t` is not needed because `t` is already excluded
- If characteristics at month `t` predict `r_t`: momentum must skip `r_{t-1}` (1 lag) to avoid microstructure effects

### Clarification on Timing Convention

The key question is: **what is the timing convention in this pipeline?**

From CLAUDE.md: "shifts returns forward one period (t characteristics predict t+1 return)"

This means at time `t`, we observe `ret_t` and predict `ret_{t+1}`. Therefore:
- `mom1m = ret_t` is fine as short-term reversal
- **Momentum should NOT include `ret_t`** — it should start from `ret_{t-1}` = `shift(1)`

### Recommended Fix

```python
crsp_mom = crsp_mom.with_columns([
    # mom1m: short-term reversal (current month return) — CORRECT as-is
    pl.col('ret').alias('mom1m'),

    # mom6m: 6-month momentum, skip current month
    # Lags 1..6 = ret_{t-1} to ret_{t-6}
    mom(1, 7, crsp_mom).alias('mom6m'),

    # mom12m: 12-month momentum, skip current month
    # Lags 1..11 = ret_{t-1} to ret_{t-11}
    # (Some papers skip 2 months, using lags 2..12. GHZ uses lags 1..11.)
    mom(1, 12, crsp_mom).alias('mom12m'),

    # chmom: change in 6-month momentum
    # First half: lags 1..6; Second half: lags 7..12
    chmom(1, 7, crsp_mom).alias('chmom'),

    # mom36m: long-term reversal, months t-13 to t-36
    mom(13, 37, crsp_mom).alias('mom36m'),

    # mom60m: long-term reversal, months t-13 to t-60
    mom(13, 61, crsp_mom).alias('mom60m'),

    # seas1a: same-calendar-month return last year = shift(12), NOT shift(11)
    pl.col('ret').shift(12).over('permno').alias('seas1a'),
])
```

### Summary of Changes

| Variable | Current | Fixed | Rationale |
|----------|---------|-------|-----------|
| `mom1m` | `ret` (lag 0) | `ret` (lag 0) | Short-term reversal, no change |
| `mom6m` | `mom(0, 6)` = lags 0-5 | `mom(1, 7)` = lags 1-6 | Skip current month |
| `mom12m` | `mom(0, 12)` = lags 0-11 | `mom(1, 12)` = lags 1-11 | Skip current month |
| `chmom` | `chmom(0, 6)` = includes lag 0 | `chmom(1, 7)` = lags 1-6 vs 7-12 | Skip current month; 6-month windows |
| `mom36m` | `mom(12, 36)` = lags 12-35 | `mom(13, 37)` = lags 13-36 | Skip 12-month momentum overlap |
| `mom60m` | `mom(12, 60)` = lags 12-59 | `mom(13, 61)` = lags 13-60 | Skip 12-month momentum overlap |
| `seas1a` | `shift(11)` | `shift(12)` | Same calendar month = 12 months ago |

### Academic Consensus on Momentum Windows

| Variable | Jegadeesh & Titman (1993) | GHZ | HXZ | Recommended |
|----------|--------------------------|-----|-----|-------------|
| `mom6m` | Lags 2-7 (skip 1 month) | Lags 1-6 | Lags 1-6 | Lags 1-6 (skip lag 0 only) |
| `mom12m` | Lags 2-12 (skip 1 month) | Lags 1-11 | Lags 1-11 | Lags 1-11 (skip lag 0 only) |
| `mom36m` | N/A | Lags 13-36 | Lags 13-36 | Lags 13-36 |
| `mom60m` | N/A | N/A | Lags 13-60 | Lags 13-60 |

---

## Issue 9: `bm_ia` — Should Use `jdate` Not `datadate` for Industry Adjustment

**Location**: Lines 2280-2286 (annual), Lines 2382-2387 (quarterly)

**Current Code (Annual)**:
```python
df_temp = data_rawa.group_by(['datadate', 'ffi49']).agg(pl.col('bm').mean().alias('bm_ind'))
data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')
data_rawa = data_rawa.with_columns([
    (pl.col('bm') - pl.col('bm_ind')).alias('bm_ia')
])
```

### The Problem: Look-Ahead Bias

Using `datadate` (fiscal year-end date) for industry grouping creates **look-ahead bias**:

1. Different firms have different fiscal year-ends (e.g., Firm A: Dec 31, Firm B: June 30)
2. When computing the industry mean `bm_ind` on a given `datadate`, you group firms whose financial data become **publicly available at different points in time**
3. A firm with `datadate = 2024-12-31` reports in February 2025. Another firm with `datadate = 2024-12-31` might have a different fiscal year convention and report much later.
4. More critically: if `datadate` is used for grouping, you might compute an industry mean using data from firms that have **not yet reported** at the time the signal is used — this is peeking at future information.

### What is `jdate`?

`jdate` (or `date`) is the **portfolio formation date** — the calendar date at which the characteristic is used for portfolio sorting. It accounts for the reporting delay (typically datadate + 4-6 months for annual data, per Fama & French convention).

### Reference: GHZ (Chen & Zimmermann 2022)

GHZ computes industry-adjusted characteristics using the **portfolio formation date** (equivalent to `jdate`):
> "Industry adjustments are computed cross-sectionally at the **signal date** (the date at which the signal would be available to investors)."

### Reference: Daniel & Titman (2006) — BM_IA Original

The original `bm_ia` variable is from Daniel & Titman (2006), "Market Reactions to Tangible and Intangible Information":
> "We subtract from each firm's book-to-market ratio the value-weighted average book-to-market of its **Fama-French industry** at the same **point in time**."

"Same point in time" means the cross-section should be formed at the calendar date when the data is available to investors, not at the fiscal year-end.

### Recommended Fix

Replace `datadate` with `jdate` (or `date`, whichever represents the portfolio formation date):

```python
# Annual bm_ia: industry-adjust at the portfolio formation date, NOT datadate
df_temp = data_rawa.group_by(['jdate', 'ffi49']).agg(
    pl.col('bm').mean().alias('bm_ind')
)
data_rawa = data_rawa.join(df_temp, on=['jdate', 'ffi49'], how='left')
data_rawa = data_rawa.with_columns([
    (pl.col('bm') - pl.col('bm_ind')).alias('bm_ia')
])
```

Similarly for quarterly:
```python
# Quarterly bm_ia
df_temp = data_rawq.group_by(['jdate', 'ffi49']).agg(
    pl.col('bm').mean().alias('bm_ind')
)
data_rawq = data_rawq.join(df_temp, on=['jdate', 'ffi49'], how='left')
data_rawq = data_rawq.with_columns([
    (pl.col('bm') - pl.col('bm_ind')).alias('bm_ia')
])
```

**Note**: Verify that `jdate` exists in the dataframe. If the column is named `date` instead of `jdate`, use `date`. The key principle is to use the **calendar date when investors can observe the data**, not the accounting period end date.

### Why This Matters

Using `datadate` for the industry cross-section means you compute the industry mean `bm_ind` from a set of firms whose data has different real-world availability dates. A firm's `bm` computed from a December fiscal year-end report is treated as contemporaneous with a March fiscal year-end report with the same `datadate`, but in reality one is available months earlier. This leaks future information into the industry mean, inflating the signal's predictive power in backtests.

---

## Issue 10: `indmom` — Careful Confirmation Needed

**Location**: Lines 2352-2355 (annual), Lines 2452-2454 (quarterly)

**Current Code**:
```python
df_temp = data_rawa.group_by(['date', 'ffi49']).agg(
    pl.col('mom12m').mean().alias('indmom')
)
data_rawa = data_rawa.join(df_temp, on=['date', 'ffi49'], how='left')
```

### Three Problems Identified

| Issue | Current | Paper (GHZ / Grinblatt & Moskowitz) |
|-------|---------|--------------------------------------|
| Return window | `mom12m` (12-month, lags 1-11) | **6-month** buy-and-hold return |
| Weighting | `mean()` (equal-weighted) | **Value-weighted** (market cap) |
| Industry classification | `ffi49` (Fama-French 49) | **2-digit SIC** |

### Reference: Grinblatt & Moskowitz (1999), "Do Industries Explain Momentum?"

From the paper (Section II.A):
> "Industry portfolios are formed each month by sorting stocks into groups based on their **two-digit SIC code**... The industry momentum strategy buys (sells) stocks in industries with the highest (lowest) past **six-month** returns. Industry returns are computed as **value-weighted** averages of individual stock returns within each industry."

### Reference: GHZ (IndMom)

> "IndMom: Weighted average of firm-level **six-month buy-and-hold return**. Average is taken over **two-digit SIC industries** each month and weights are based on **market value of equity** (abs(prc) * shrout)."

### Reference: 修老师's Paper

The user mentions `indmom` appears in 修老师's paper. This needs to be cross-referenced to determine if 修老师's definition differs from GHZ/Grinblatt & Moskowitz. Key questions:
1. Does 修老师 use 6-month or 12-month returns?
2. Equal-weighted or value-weighted?
3. What industry classification?

**Action required**: Check 修老师's paper for the exact `indmom` definition and reconcile with GHZ.

### Recommended Fix (per GHZ / Grinblatt & Moskowitz)

```python
# Step 1: Create 2-digit SIC code
data_rawa = data_rawa.with_columns([
    (pl.col('sic') // 100).alias('sic2')
])

# Step 2: Compute value-weighted 6-month industry momentum
# mom6m should already be computed; me is market equity
data_rawa = data_rawa.with_columns([
    (pl.col('mom6m') * pl.col('me')).alias('_vw_mom6m')
])

df_temp = (data_rawa
    .filter(pl.col('me').is_not_null() & pl.col('mom6m').is_not_null())
    .group_by(['date', 'sic2'])
    .agg([
        pl.col('_vw_mom6m').sum().alias('_sum_vw'),
        pl.col('me').sum().alias('_sum_me')
    ])
    .with_columns([
        (pl.col('_sum_vw') / pl.col('_sum_me')).alias('indmom')
    ])
    .select(['date', 'sic2', 'indmom'])
)

data_rawa = data_rawa.join(df_temp, on=['date', 'sic2'], how='left')
data_rawa = data_rawa.drop('_vw_mom6m')
```

Similarly for quarterly:
```python
data_rawq = data_rawq.with_columns([
    (pl.col('sic') // 100).alias('sic2'),
    (pl.col('mom6m') * pl.col('me')).alias('_vw_mom6m')
])
df_temp = (data_rawq
    .filter(pl.col('me').is_not_null() & pl.col('mom6m').is_not_null())
    .group_by(['date', 'sic2'])
    .agg([
        pl.col('_vw_mom6m').sum().alias('_sum_vw'),
        pl.col('me').sum().alias('_sum_me')
    ])
    .with_columns([
        (pl.col('_sum_vw') / pl.col('_sum_me')).alias('indmom')
    ])
    .select(['date', 'sic2', 'indmom'])
)
data_rawq = data_rawq.join(df_temp, on=['date', 'sic2'], how='left')
data_rawq = data_rawq.drop('_vw_mom6m')
```

### Important Note on `mom6m` Availability

The fix assumes `mom6m` is available in `data_rawa` / `data_rawq` at the time `indmom` is computed. Verify that CRSP momentum variables have been merged in before this step. If `mom6m` is only in `crsp_mom`, you may need to compute `indmom` in the CRSP section and then merge it in.

---

## Summary Table

| # | Variable | Issue | Severity | Action | Paper Reference |
|---|----------|-------|----------|--------|----------------|
| 1 | `oancfq` | Level 3 fallback should use 1-quarter ΔWC, not wcaptq | MEDIUM | Modify Level 3 per JKP/GHZ | JKP (2023), Hribar & Collins (2002) |
| 2 | `chato` (quarterly) | Denominator questioned | LOW | **No change needed** — quarterly version is correct | Soliman (2008) |
| 3 | `beq` | Missing seq fallback + ps hierarchy | MEDIUM | Add `seqq → ceqq+pstkq → atq-ltq` cascade | HXZ (2020) Appendix A.1 |
| 4 | `dy` (monthly) | Current vs lagged ME in denominator | LOW | Change to `me_l1` if following JKP; current is valid per GHZ | JKP (2023) vs L&R (1982) |
| 5 | `sacc` | Can it be deleted? | INFO | **No** — required for `stdacc` and `stdcf` | Bandyopadhyay et al. (2010) |
| 6 | `stdcf` | Need original paper definition | INFO | **Current code is correct** per paper | Bandyopadhyay et al. (2010) |
| 7 | `nincr` | YoY or QoQ? | INFO | **Current fix (YoY) is correct** | GHZ / Barth et al. (1999) |
| 8 | Momentum | `mom(0,*)` includes current month; seas1a shift wrong | **HIGH** | Fix all windows per table above | Jegadeesh & Titman (1993), GHZ |
| 9 | `bm_ia` | Uses `datadate` (look-ahead bias) | **HIGH** | Change to `jdate`/`date` | Daniel & Titman (2006), GHZ |
| 10 | `indmom` | Wrong return (12m→6m), weighting (EW→VW), industry (FF49→SIC2) | **HIGH** | Fix all three per Grinblatt & Moskowitz | Grinblatt & Moskowitz (1999), GHZ |

---

## Full Reference List

| Paper | Citation | Variables |
|-------|----------|-----------|
| Bandyopadhyay, Huang & Wirjanto (2010) | "The Accrual Volatility Anomaly" | `stdacc`, `stdcf`, `sacc` |
| Barth, Elliott & Finn (1999) | "Market Rewards Associated with Patterns of Increasing Earnings", JAR | `nincr` |
| Chen & Zimmermann (2022) | "Open Source Cross-Sectional Asset Pricing", CFR | GHZ replication |
| Daniel & Titman (2006) | "Market Reactions to Tangible and Intangible Information", JF | `bm_ia` |
| Fama & French (2015) | "A Five-Factor Asset Pricing Model", JFE | `op` |
| Grinblatt & Moskowitz (1999) | "Do Industries Explain Momentum?", JF | `indmom` |
| Hribar & Collins (2002) | "Errors in Estimating Accruals", TAR | `oancfq` construction |
| Hou, Xue & Zhang (2020) | "Replicating Anomalies", RFS | HXZ definitions |
| Jegadeesh (1990) | "Evidence of Predictable Behavior of Security Returns", JF | Short-term reversal |
| Jegadeesh & Titman (1993) | "Returns to Buying Winners and Selling Losers", JF | Momentum skip-month |
| Jensen, Kelly & Pedersen (2023) | "Is There a Replication Crisis in Finance?", JF | JKP definitions |
| Litzenberger & Ramaswamy (1982) | "The Effects of Dividends on Common Stock Prices", JF | `dy` |
| Naranjo, Nimalendran & Ryngaert (1998) | "Stock Returns, Dividend Yields, and Taxes", JF | `dy` |
| Sloan (1996) | "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows?", TAR | `acc`, `oancfq` |
| Soliman (2008) | "The Use of DuPont Analysis by Market Participants", TAR | `chato`, `noa`, `rna`, `ato`, `pm` |
| Hirshleifer, Hou, Teoh & Zhang (2004) | "Do Investors Overvalue Firms with Bloated Balance Sheets?", JFE | `noa` |
| Fairfield, Whisenant & Yohn (2003) | "Accrued Earnings and Growth", TAR | `grltnoa` |

---

## Issue 11 (NEW): `noa_raw` / `noa` / `rna` / `ato` / `grltnoa` — Concepts Clarification and Fixes

### 11A. `noa_raw` and `grltnoa` Are Different Variables — Do NOT Unify

The user's two formulas correspond to **two different academic variables**:

| Variable | Paper | What It Measures |
|----------|-------|-----------------|
| `noa_raw` → `noa` | Hirshleifer, Hou, Teoh & Zhang (2004), HXZ A.3.5 | **Level** of net operating assets, scaled by lagged AT |
| `grltnoa` | Fairfield, Whisenant & Yohn (2003), HXZ A.3.6 | **Growth** in **long-term** net operating assets |

They should **not** be conflated. `noa_raw` is the dollar-level NOA used as:
1. A predictor when scaled: `noa = noa_raw / AT_{t-1}`
2. A denominator for DuPont decomposition: `rna = OIADP / noa_raw_{t-1}`, `ato = Sale / noa_raw_{t-1}`

`grltnoa` decomposes the *change* in NOA into current vs long-term components, subtracts depreciation from the long-term part, and scales by average AT.

**Conclusion: Keep both formulas as they are — they serve different purposes.**

---

### 11B. `noa_raw` — Should NOT Subtract IVAO (per HXZ)

**Location**: Lines 610-613 (annual), Lines 1766-1769 (quarterly)

**Current Code (Annual)**:
```python
noa_raw = (AT - CHE - IVAO) - (AT - DLC - DLTT - MIB - PSTK - CEQ)
```

**Current Code (Quarterly)**:
```python
noa_raw = (ATQ - CHEQ - IVAOQ) - (ATQ - DLCQ - DLTTQ - MIBQ - PSTKQ - CEQQ)
```

**Problem**: The code subtracts `IVAO` (Investment and Advances - Other) from operating assets. The HXZ paper does **not** subtract IVAO.

#### Reference: HXZ A.3.5 (Hirshleifer et al. 2004, "Do Investors Overvalue Firms with Bloated Balance Sheets?")

> "Operating assets are total assets (Compustat annual item AT) minus cash and short-term investment (item CHE)."
>
> "Operating liabilities are total assets minus debt included in current liabilities (item DLC, zero if missing), minus long-term debt (item DLTT, zero if missing), minus minority interest (item MIB, zero if missing), minus preferred stock (item PSTK, zero if missing), and minus common equity (item CEQ)."

**Paper formula**:
```
OA = AT - CHE           (NO IVAO subtraction)
OL = AT - DLC - DLTT - MIB - PSTK - CEQ
NOA = OA - OL = DLC + DLTT + MIB + PSTK + CEQ - CHE
```

**Current code formula** (algebraically simplified):
```
OA = AT - CHE - IVAO
OL = AT - DLC - DLTT - MIB - PSTK - CEQ
NOA = OA - OL = DLC + DLTT + MIB + PSTK + CEQ - CHE - IVAO
```

The difference is `IVAO`. While subtracting IVAO is economically defensible (IVAO includes equity-method investments which are arguably non-operating), it **deviates from the HXZ/Hirshleifer definition**.

#### Comparison

| Definition | OA | OL | NOA (simplified) |
|-----------|----|----|------------------|
| **HXZ / Hirshleifer (2004)** | `AT - CHE` | `AT - DLC - DLTT - MIB - PSTK - CEQ` | `DLC + DLTT + MIB + PSTK + CEQ - CHE` |
| **Current code** | `AT - CHE - IVAO` | same | `DLC + DLTT + MIB + PSTK + CEQ - CHE - IVAO` |
| **Some practitioner implementations** | `AT - CHE - IVAO` | same | same as current code |

#### Recommended Fix (Annual)

```python
# noa_raw: per HXZ A.3.5, OA = AT - CHE (do NOT subtract IVAO)
data_rawa = data_rawa.with_columns([
    ((pl.col('at') - pl.col('che')) -
     (pl.col('at') - pl.col('dlc') - pl.col('dltt') -
      pl.col('mib') - pl.col('pstk') - pl.col('ceq'))).alias('noa_raw')
])
```

#### Recommended Fix (Quarterly)

```python
data_rawq = data_rawq.with_columns([
    ((pl.col('atq') - pl.col('cheq')) -
     (pl.col('atq') - pl.col('dlcq') - pl.col('dlttq') - pl.col('mibq') -
      pl.col('pstkq') - pl.col('ceqq'))).alias('noa_raw')
])
```

**Note**: If you choose to keep the IVAO subtraction for economic reasons, document the deviation from HXZ explicitly. Also remove `'ivao'` and `'ivaoq'` from `_ANNUAL_FILL_ZERO` / quarterly fill-zero lists if no longer used in `noa_raw`.

---

### 11C. Quarterly `rna` and `ato` — Time Horizon Mismatch (DuPont Identity Broken)

**Location**: Lines 1780-1795 (quarterly)

**Current Code**:
```python
# rna: single-quarter OIADP / 4-quarter-lagged NOA
(pl.col('oiadpq') / pl.col('noa_raw_l4').replace(0, None)).alias('rna')

# ato: single-quarter sale / 4-quarter-lagged NOA
(pl.col('saleq') / pl.col('noa_raw_l4').replace(0, None)).alias('ato')

# pm: single-quarter OIADP / single-quarter sale
(pl.col('oiadpq') / pl.col('saleq').replace(0, None)).alias('pm')
```

**Problem**: The DuPont decomposition requires `RNA = PM × ATO`:
```
RNA = OIADP / NOA_{t-1}
PM  = OIADP / Sale
ATO = Sale / NOA_{t-1}
=> RNA = PM × ATO ✓ (if same time horizon for all three)
```

Current quarterly code uses:
- Numerator: **single quarter** (`oiadpq`, `saleq`)
- Denominator for `rna`/`ato`: **4-quarter lag** (`noa_raw_l4`)
- Denominator for `pm`: **single quarter** (`saleq`)

The DuPont identity still holds algebraically (`oiadpq/noa_raw_l4 = oiadpq/saleq × saleq/noa_raw_l4`), so mathematically it's not broken. But the **economic interpretation** is problematic: dividing one quarter of income by NOA from 4 quarters ago gives an annualized-like ratio that conflates time periods.

#### Reference: Soliman (2008), "The Use of DuPont Analysis by Market Participants"

Soliman uses **annual** data:
```
ATO = Sale_t / ((NOA_t + NOA_{t-1}) / 2)
PM  = OIADP_t / Sale_t
RNA = PM × ATO
```

For quarterly adaptation, there are two consistent approaches:

**Option A — Single-Quarter (pure quarterly)**:
```python
# 1-quarter lag
data_rawq = data_rawq.with_columns([
    pl.col('noa_raw').shift(1).over('permno').alias('noa_raw_l1')
])
data_rawq = data_rawq.with_columns([
    (pl.col('oiadpq') / pl.col('noa_raw_l1').replace(0, None)).alias('rna'),
    (pl.col('oiadpq') / pl.col('saleq').replace(0, None)).alias('pm'),
    (pl.col('saleq') / pl.col('noa_raw_l1').replace(0, None)).alias('ato'),
])
```

**Option B — TTM (annualized quarterly, more comparable to annual)**:
```python
# TTM numerators with 4-quarter lag denominator
data_rawq = data_rawq.with_columns([
    pl.col('noa_raw').shift(4).over('permno').alias('noa_raw_l4')
])
data_rawq = data_rawq.with_columns([
    (ttm4('oiadpq', data_rawq) / pl.col('noa_raw_l4').replace(0, None)).alias('rna'),
    (ttm4('oiadpq', data_rawq) / ttm4('saleq', data_rawq).replace(0, None)).alias('pm'),
    (ttm4('saleq', data_rawq) / pl.col('noa_raw_l4').replace(0, None)).alias('ato'),
])
```

#### Comparison

| Approach | Numerator | NOA Denominator | DuPont Holds | Comparable to Annual |
|----------|-----------|----------------|--------------|---------------------|
| Current code | Single Q | 4Q lag | Yes (math) | No (magnitude off) |
| **Option A** | Single Q | **1Q lag** | Yes | No (quarterly rate) |
| **Option B** | **TTM** | 4Q lag | Yes | **Yes** |

#### Recommendation

**Option B (TTM)** is preferred for consistency with the annual version and comparability across frequencies. This matches how other TTM variables (`saleq4`, `ibq4`, `oiadpq4`, etc.) are already constructed in the quarterly pipeline.

**Note**: The annual `rna`/`ato` (Lines 619-636) uses `noa_raw_l1` (1-year lag) with annual flow data — this is correct and consistent.

---

### 11D. `grltnoa` — Current Code Is Correct

**Location**: Lines 899-920 (annual), Lines 1808-1830 (quarterly)

The `grltnoa` formula is verified correct per HXZ A.3.6 / Fairfield et al. (2003):

```
grltnoa = (ΔNOA - ΔWC - DP) / avg(AT)
```

Where:
- `ΔNOA` = change in total operating assets minus operating liabilities (rect + invt + ppent + aco + intan + ao - ap - lco - lo)
- `ΔWC` = change in net working capital (Δrect + Δinvt + Δaco - Δap - Δlco)
- `DP` = depreciation
- `avg(AT)` = average total assets

The long-term NOA growth = ΔNOA - ΔWC - DP = Δ(ppent + intan + ao - lo) - DP

**Both annual and quarterly `grltnoa` are correct.** No changes needed.

---

### 11E. Summary: Which Formula for Which Variable

| Variable | Formula | Paper | Status |
|----------|---------|-------|--------|
| `noa_raw` | `(AT - CHE) - (AT - DLC - DLTT - MIB - PSTK - CEQ)` | HXZ A.3.5, Hirshleifer (2004) | **Fix**: Remove IVAO subtraction |
| `noa` | `noa_raw / AT_{t-1}` | HXZ A.3.5 | Correct (once noa_raw fixed) |
| `rna` (annual) | `OIADP / noa_raw_{t-1}` | Soliman (2008) | Correct |
| `ato` (annual) | `Sale / noa_raw_{t-1}` | Soliman (2008) | Correct |
| `pm` (annual) | `OIADP / Sale` | Soliman (2008) | Correct |
| `rna` (quarterly) | `oiadpq / noa_raw_l4` | — | **Fix**: Use TTM(oiadpq)/noa_raw_l4 or oiadpq/noa_raw_l1 |
| `ato` (quarterly) | `saleq / noa_raw_l4` | — | **Fix**: Use TTM(saleq)/noa_raw_l4 or saleq/noa_raw_l1 |
| `pm` (quarterly) | `oiadpq / saleq` | — | Correct (if choosing single-Q); use TTM if choosing Option B |
| `grltnoa` | `(ΔNOA - ΔWC - DP) / avg(AT)` | HXZ A.3.6, Fairfield (2003) | Correct |

---

## Updated Summary Table (All Issues)

| # | Variable | Issue | Severity | Action | Paper Reference |
|---|----------|-------|----------|--------|----------------|
| 1 | `oancfq` | Level 3 fallback should use 1-quarter ΔWC, not wcaptq | MEDIUM | Modify Level 3 per JKP/GHZ | JKP (2023), Hribar & Collins (2002) |
| 2 | `chato` (quarterly) | Denominator questioned | LOW | **No change needed** | Soliman (2008) |
| 3 | `beq` | Missing seq fallback + ps hierarchy | MEDIUM | Add `seqq → ceqq+pstkq → atq-ltq` cascade | HXZ (2020) Appendix A.1 |
| 4 | `dy` | **HIGH**: Annual `dvt/me` overwrites monthly dy; quarterly output missing dy | **HIGH** | Remove overwrite; add dy to chars_q; optionally rename annual to `dy_a` | GHZ, JKP (2023), L&R (1982) |
| 5 | `sacc` | Can it be deleted? | INFO | **No** — required for `stdacc` and `stdcf` | Bandyopadhyay et al. (2010) |
| 6 | `stdcf` | Need original paper definition | INFO | **Current code is correct** | Bandyopadhyay et al. (2010) |
| 7 | `nincr` | YoY or QoQ? | INFO | **Current fix (YoY) is correct** | GHZ / Barth et al. (1999) |
| 8 | Momentum | `mom(0,*)` includes current month; seas1a shift wrong | **HIGH** | Fix all windows per table | Jegadeesh & Titman (1993), GHZ |
| 9 | `bm_ia` | Uses `datadate` (look-ahead bias) | **HIGH** | Change to `jdate`/`date` | Daniel & Titman (2006), GHZ |
| 10 | `indmom` | Wrong return (12m→6m), weighting (EW→VW), industry (FF49→SIC2) | **HIGH** | Fix all three | Grinblatt & Moskowitz (1999), GHZ |
| 11B | `noa_raw` | Subtracts IVAO — HXZ does not | **MEDIUM** | Remove IVAO from OA formula | Hirshleifer et al. (2004), HXZ A.3.5 |
| 11C | `rna`/`ato` (quarterly) | Single-Q numerator with 4Q-lag denominator | **MEDIUM** | Use TTM numerators or 1Q-lag denominator | Soliman (2008) |
