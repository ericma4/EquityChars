# Calculate HXZ Replicating Anomalies
# RE: Revisions in analysts' earnings forecasts

import polars as pl
import duckdb
from functions import INPUT_PATH, OUTPUT_PATH

###############################################
# Merging IBES and CRSP by using ICLINK table #
###############################################

# Read ICLINK table from local file
iclink = pl.scan_csv(INPUT_PATH + 'iclink_ciz.csv')
# Convert all column names to lowercase first
iclink = iclink.select([pl.col(c).alias(c.lower()) for c in iclink.collect_schema().names()])
# Then rename specific columns as needed
iclink = iclink.rename({'issuernm': 'comnam'})

# Read IBES data
# 20261030fix-HXZ A.1.3 requires unadjusted IBES EPS and PRICE; the old file contains adjusted EPS only.
ibes = pl.scan_parquet(INPUT_PATH + 'ibes_unadjusted.parquet')

# Filtering IBES
# 20261030fix-Exclude FPI=2 forecasts; HXZ Appendix A.1.3 uses current-fiscal-year FPI=1 forecasts.
# The original filter checked medest and fpedats only; retain those conditions and add the FPI filter.
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
# 20261030fix-CRSP retains the original stock-month membership; prices now come from IBES.
crsp_msf = crsp_msf.select('permno', pl.col('mthcaldt').alias('date'))

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

# 20261030fix-Sort and retain the latest observation per stock, forecast period, and month.
ibes_crsp = (
    ibes_crsp
    .select('ticker', 'permno', 'statpers', 'fpedats', 'anndats_act', 'curr_act', 'meanest', 'price', 'prdays')
    .collect()
    .with_columns((pl.col('statpers').dt.year() * 12 + pl.col('statpers').dt.month()).alias('month_id'))
    .sort(['ticker', 'permno', 'fpedats', 'month_id', 'statpers'], descending=[False, False, False, False, True])
    .unique(['ticker', 'permno', 'fpedats', 'month_id'], keep='first', maintain_order=True)
    .with_row_index('event_id')
)

# 20261030fix-WRDS unadjusted-data note, Method 3: align share bases using CRSP daily split factors.
# Map statistical dates to the preceding market trading day before joining stock-specific factors.
calendar = (pl.scan_parquet(INPUT_PATH + 'crsp_dsf.parquet')
    .select(pl.col('dlycaldt').alias('split_date'))
    .unique().sort('split_date').collect())
# 20261030fix-Do not extend split factors beyond the available CRSP history.
ibes_crsp = ibes_crsp.filter(pl.col('statpers') <= calendar['split_date'].max())
snapshot_dates = (pl.concat([ibes_crsp.select('statpers'),
    ibes_crsp.select(pl.col('prdays').alias('statpers'))]).drop_nulls().unique().sort('statpers')
    .join_asof(calendar, left_on='statpers', right_on='split_date', strategy='backward'))
factors = (pl.scan_parquet(INPUT_PATH + 'crsp_dsf.parquet')
    .select(pl.col('permno').cast(pl.Int64), pl.col('dlycaldt').alias('split_date'),
            pl.col('dlycumfacshr').cast(pl.Float64).alias('split_factor'))
    .filter(pl.col('split_date').is_in(snapshot_dates['split_date'].implode()))
    .collect())
ibes_crsp = (ibes_crsp.join(snapshot_dates, on='statpers', how='left', validate='m:1')
    .join(factors, on=['permno', 'split_date'], how='left', validate='m:1').drop('split_date')
    # 20261030fix-IBES PRDAYS is the actual quote date; its share basis can differ from STATPERS.
    .join(snapshot_dates.rename({'statpers': 'prdays'}), on='prdays', how='left', validate='m:1')
    .join(factors.rename({'split_factor': 'price_split_factor'}),
          on=['permno', 'split_date'], how='left', validate='m:1').drop('split_date'))

# 20261030fix-Match the previous calendar month for the same stock and forecast period.

con = duckdb.connect()
con.register('forecasts', ibes_crsp.to_arrow())
ibes_crsp = con.sql("""
    SELECT a.*, b.meanest AS meanest_last_month,
           b.price AS price_last_month, b.split_factor AS split_factor_last_month,
           b.price_split_factor AS price_factor_last_month
    FROM forecasts a LEFT JOIN forecasts b
      ON a.ticker=b.ticker AND a.permno=b.permno AND a.fpedats=b.fpedats
     AND b.month_id=a.month_id-1
""").pl()

# Re-sort
ibes_crsp = ibes_crsp.sort(['ticker', 'permno', 'fpedats', 'statpers'])

###########################
# Calculate HXZ RE        #
###########################

# 20261030fix-Disabled: dropping forecasts without a prior month compresses history; keep them with a null revision.

# Calculate adjusted price and monthly revision
ibes_crsp = ibes_crsp.with_columns([
    # 20261030fix-HXZ A.1.3: express the prior IBES price and forecast in current-snapshot shares.
    (pl.col('price_last_month') * pl.col('split_factor') /
     pl.col('price_factor_last_month').replace(0, None)).alias('prc_adj')
])

# 20261030fix-Keep invalid-price records for history; leave only their revision null.
ibes_crsp = ibes_crsp.with_columns([
    pl.when(pl.col('prc_adj') > 0)
    # 20261030fix-Disabled: unadjusted forecasts on opposite sides of a split cannot be subtracted directly.
    .then((pl.col('meanest') - pl.col('meanest_last_month') *
           pl.col('split_factor') / pl.col('split_factor_last_month').replace(0, None)) / pl.col('prc_adj'))
    .alias('monthly_revision')
])

# Calculate RE   #

# 20261030fix-Disabled: row shifts can skip missing months or reference future observations; match calendar months.

con.register('revisions', ibes_crsp.to_arrow())
# 20261030fix-HXZ A.1.3 uses return month t; the formation month contributes the current and previous five revisions.
for k in range(0, 6):
    lag = con.sql(f"""
        SELECT a.event_id, b.monthly_revision AS monthly_revision_l{k}
        FROM revisions a LEFT JOIN revisions b
          -- 20261030fix-Match historical FY1 revisions by stock and month across fiscal-year changes; monthly differences still match the same target year.
          ON a.ticker=b.ticker AND a.permno=b.permno
         AND b.month_id=a.month_id-{k}
    """).pl()
    ibes_crsp = ibes_crsp.join(lag, on='event_id', how='left')
con.close()

# 20261030fix-Count valid revisions within the six calendar months.

ibes_crsp = ibes_crsp.with_columns([
    # 20261030fix-HXZ A.1.3: retain the window above; range(1, 7) lagged it one extra month.
    pl.concat_list([f'monthly_revision_l{k}' for k in range(0, 6)]).list.drop_nulls().alias('past_revisions')
]).with_columns([
    # 20261030fix-HXZ A.1.3 requires at least four revisions; sum them as in Eq. A2.
    pl.when(pl.col('past_revisions').list.len() >= 4)
    .then(pl.col('past_revisions').list.sum())
    .alias('re')
])

# Filter and finalize
# 20261030fix-Disabled: ticker deduplication does not guarantee one stock-month; select the latest eligible observation.

ibes_crsp = (
    # 20261030fix-FY1 can end before the snapshot while annual earnings are still unannounced.
    ibes_crsp.filter(pl.col('anndats_act').is_null() | (pl.col('statpers') < pl.col('anndats_act')))
    .with_columns(pl.col('statpers').dt.month_end().alias('jdate'))
    .sort(['permno', 'jdate', 'statpers', 'fpedats', 'ticker'], descending=[False, False, True, False, False])
    .unique(['permno', 'jdate'], keep='first', maintain_order=True)
)

# Select final columns and rename
# 20261030fix-Restore anndats_act and curr_act in the original output schema.
ibes_crsp = ibes_crsp.select(['ticker', 'statpers', 'fpedats', 'anndats_act', 'curr_act', 'permno', 're'])
ibes_crsp = ibes_crsp.rename({'statpers': 'date'})

# Write output
ibes_crsp.write_parquet(OUTPUT_PATH + 'myre.parquet')

print("RE calculation completed successfully!")
