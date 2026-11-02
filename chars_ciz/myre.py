# Calculate HXZ Replicating Anomalies
# RE: Revisions in analysts' earnings forecasts

import polars as pl
from functions import INPUT_PATH, OUTPUT_PATH

#########################################################################
# Merging IBES and CRSP by using ICLINK table. Merging last month price #
#########################################################################

# Read ICLINK table from local file
iclink = pl.scan_csv(INPUT_PATH + 'iclink_ciz.csv')
# Convert all column names to lowercase first
iclink = iclink.select([pl.col(c).alias(c.lower()) for c in iclink.collect_schema().names()])
# Then rename specific columns as needed
iclink = iclink.rename({'issuernm': 'comnam'})

# Read IBES data
ibes = pl.scan_parquet(INPUT_PATH + 'ibes.parquet')

# Filtering IBES
# 20261030fix-HXZ A.1.3: current fiscal year forecasts only (FPI = 1).
ibes = ibes.filter(
    (pl.col('fpi') == '1') &
    pl.col('medest').is_not_null() &
    pl.col('fpedats').is_not_null()
)

# Add merge_date (end of month for statpers)
ibes = ibes.with_columns([
    pl.col('statpers').dt.month_end().alias('merge_date')
])

# Read CRSP monthly stock file from local
crsp_msf = pl.scan_parquet(INPUT_PATH + 'crsp_msf.parquet')
crsp_msf = crsp_msf.rename({'mthcaldt': 'date', 'mthprc': 'prc', 'mthcumfacpr': 'cfacpr'})

# Add merge_date (next month end)
crsp_msf = crsp_msf.with_columns([
    pl.col('date').dt.month_end().alias('date'),
])
crsp_msf = crsp_msf.with_columns([
    (pl.col('date') + pl.duration(days=1)).dt.month_end().alias('merge_date')
])

# Merge IBES with ICLINK
ibes_iclink = ibes.join(iclink, on='ticker', how='left')

# Merge with CRSP
ibes_crsp = ibes_iclink.join(crsp_msf, on=['permno', 'merge_date'], how='inner')
ibes_crsp = ibes_crsp.sort(['ticker', 'fpedats', 'statpers'])

###############################
# Merging last month forecast #
###############################

# Create last month columns using partitioned shift (no guard needed)
ibes_crsp = ibes_crsp.with_columns([
    pl.col('statpers').shift(1).over(['ticker', 'fpedats']).alias('statpers_last_month'),
    pl.col('meanest').shift(1).over(['ticker', 'fpedats']).alias('meanest_last_month'),
])

# Re-sort
ibes_crsp = ibes_crsp.sort(['ticker', 'permno', 'fpedats', 'statpers'])

###########################
# Drop empty "last month" #
# Calculate HXZ RE        #
###########################

# 20261030fix-Keep months without a prior forecast (e.g. FY1 rollover) with a null revision, so the
# six-month window below still reaches the revisions before the rollover.

# Calculate adjusted price and monthly revision
ibes_crsp = ibes_crsp.with_columns([
    # 20261030fix-CRSP stores the cumulative factor as Decimal(17,12); after many reverse splits it
    # underflows to 0 (250 rows), and Decimal division by zero aborts the script.
    (pl.col('prc').cast(pl.Float64) / pl.col('cfacpr').cast(pl.Float64).replace(0, None)).alias('prc_adj')
])


# 20261030fix-A revision compares consecutive calendar months; a skipped month is not a monthly change.
month_id = pl.col('statpers').dt.year() * 12 + pl.col('statpers').dt.month()
prev_month_id = pl.col('statpers_last_month').dt.year() * 12 + pl.col('statpers_last_month').dt.month()
ibes_crsp = ibes_crsp.with_columns([
    pl.when((month_id - prev_month_id == 1) & (pl.col('prc_adj') > 0))
    .then((pl.col('meanest') - pl.col('meanest_last_month')) / pl.col('prc_adj'))
    .alias('monthly_revision')
])

# 20261030fix-One FY1 row per stock-month; the six-month window spans fiscal-year rollovers.
ibes_crsp = (ibes_crsp.sort(['permno', 'statpers', 'ticker'])
             .unique(subset=['permno', 'statpers'], keep='first', maintain_order=True))

##################
# Calculate RE   #
##################

# 20261030fix-HXZ A.1.3 (Eq. A2): RE at month t sums the revisions of months t-5..t (the row's own
# month is the latest revision before the t+1 return), matched by calendar month; at least four.
# Main used rows l1..l6 (one month stale) and averaged them.
prev = lambda c, j: c.shift(j).over('permno', order_by='statpers')
revisions = [pl.col('monthly_revision')] + [
    # month t-k is at most k rows back (fewer when IBES skips a month)
    pl.coalesce([pl.when(month_id - prev(month_id, j) == k).then(prev(pl.col('monthly_revision'), j))
                 for j in range(1, k + 1)])
    for k in range(1, 6)
]
ibes_crsp = ibes_crsp.with_columns(pl.concat_list(revisions).list.drop_nulls().alias('past_revisions'))
ibes_crsp = ibes_crsp.with_columns(
    pl.when(pl.col('past_revisions').list.len() >= 4).then(pl.col('past_revisions').list.sum()).alias('re'))

# Filter and finalize
ibes_crsp = ibes_crsp.filter(pl.col('re').is_not_null())

# Select final columns and rename
ibes_crsp = ibes_crsp.select(['ticker', 'statpers', 'fpedats', 'anndats_act', 'curr_act', 'permno', 're'])
ibes_crsp = ibes_crsp.rename({'statpers': 'date'})

# Write output (collect LazyFrame to DataFrame first)
ibes_crsp.collect().write_parquet(OUTPUT_PATH + 'myre.parquet')

print("RE calculation completed successfully!")