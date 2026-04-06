# Calculate HSZ Replicating Anomalies
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
ibes = ibes.filter(
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

ibes_crsp = ibes_crsp.filter(pl.col('statpers_last_month').is_not_null())

# Calculate adjusted price and monthly revision
ibes_crsp = ibes_crsp.with_columns([
    (pl.col('prc') / pl.col('cfacpr')).alias('prc_adj')
])

ibes_crsp = ibes_crsp.filter(pl.col('prc_adj') > 0)

ibes_crsp = ibes_crsp.with_columns([
    ((pl.col('meanest') - pl.col('meanest_last_month')) / pl.col('prc_adj')).alias('monthly_revision')
])

# Create permno_fpedats identifier
ibes_crsp = ibes_crsp.with_columns([
    (pl.col('permno').cast(pl.Utf8) + '-' + pl.col('fpedats').cast(pl.Utf8)).alias('permno_fpedats')
])

# Drop duplicates and add count
ibes_crsp = ibes_crsp.unique(subset=['permno_fpedats', 'statpers'], keep='first')
ibes_crsp = ibes_crsp.with_columns([
    pl.col('permno_fpedats').cum_count().over('permno_fpedats').alias('count')
])

##################
# Calculate RE   #
##################

# Create lagged monthly_revision columns (partition by permno_fpedats to avoid mixing forecast periods)
ibes_crsp = ibes_crsp.with_columns([
    pl.col('monthly_revision').shift(1).over('permno_fpedats').alias('monthly_revision_l1'),
    pl.col('monthly_revision').shift(2).over('permno_fpedats').alias('monthly_revision_l2'),
    pl.col('monthly_revision').shift(3).over('permno_fpedats').alias('monthly_revision_l3'),
    pl.col('monthly_revision').shift(4).over('permno_fpedats').alias('monthly_revision_l4'),
    pl.col('monthly_revision').shift(5).over('permno_fpedats').alias('monthly_revision_l5'),
    pl.col('monthly_revision').shift(6).over('permno_fpedats').alias('monthly_revision_l6')
])

# Calculate RE based on count
ibes_crsp = ibes_crsp.with_columns([
    pl.when(pl.col('count') == 4)
    .then((pl.col('monthly_revision_l1') + pl.col('monthly_revision_l2') + pl.col('monthly_revision_l3')) / 3)
    .when(pl.col('count') == 5)
    .then((pl.col('monthly_revision_l1') + pl.col('monthly_revision_l2') + pl.col('monthly_revision_l3') + pl.col('monthly_revision_l4')) / 4)
    .when(pl.col('count') == 6)
    .then((pl.col('monthly_revision_l1') + pl.col('monthly_revision_l2') + pl.col('monthly_revision_l3') + pl.col('monthly_revision_l4') + pl.col('monthly_revision_l5')) / 5)
    .when(pl.col('count') >= 7)
    .then((pl.col('monthly_revision_l1') + pl.col('monthly_revision_l2') + pl.col('monthly_revision_l3') + pl.col('monthly_revision_l4') + pl.col('monthly_revision_l5') + pl.col('monthly_revision_l6')) / 6)
    .otherwise(None)
    .alias('re')
])

# Filter and finalize
ibes_crsp = ibes_crsp.filter(pl.col('count') >= 4)
ibes_crsp = ibes_crsp.sort(['ticker', 'statpers', 'fpedats'])
ibes_crsp = ibes_crsp.unique(subset=['ticker', 'statpers'], keep='first')

# Select final columns and rename
ibes_crsp = ibes_crsp.select(['ticker', 'statpers', 'fpedats', 'anndats_act', 'curr_act', 'permno', 're'])
ibes_crsp = ibes_crsp.rename({'statpers': 'date'})

# Write output (collect LazyFrame to DataFrame first)
ibes_crsp.collect().write_parquet(OUTPUT_PATH + 'myre.parquet')

print("RE calculation completed successfully!")