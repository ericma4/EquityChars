# Calculate HSZ Replicating Anomalies
# ABR: Cumulative abnormal stock returns around earnings announcements
# Optimized version with Polars + DuckDB

import polars as pl
from polars import col
import duckdb
import os
from functions import INPUT_PATH, OUTPUT_PATH

###################
# Compustat Block #
###################
print("=" * 10, "Loading Compustat data", "=" * 10)

comp = (
    pl.scan_parquet(INPUT_PATH + "comp_fundq.parquet")
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr"])
    .with_columns(
        col("datadate").cast(pl.Date),
        col("rdq").cast(pl.Date),
    )
    .collect()
)

print("=" * 10, "Compustat data ready", "=" * 10)

###################
#    CCM Block    #
###################
print("=" * 10, "Loading CCM links", "=" * 10)

ccm = (
    pl.scan_parquet(INPUT_PATH + "ccm.parquet")
    .select([
        "gvkey",
        col("permno").cast(pl.Int64),
        col("linkdt").cast(pl.Date),
        col("linkenddt").cast(pl.Date),
    ])
    .with_columns(
        col("linkenddt").fill_null(pl.date(2099, 12, 31))
    )
    .collect()
)

# Join comp with CCM and filter by link dates
ccm_merged = (
    comp.join(ccm, on="gvkey", how="left")
    .filter(
        (col("datadate") >= col("linkdt"))
        & (col("datadate") <= col("linkenddt"))
    )
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr", "permno"])
)

print("=" * 10, "CCM data ready", "=" * 10)

###################
#    CRSP Block   #
###################
# Map RDQ to first trading day on or after
print("=" * 10, "Mapping RDQ to trading days", "=" * 10)

# Get trading days from CRSP daily data
trading_days = (
    pl.scan_parquet(INPUT_PATH + "crsp_dsf.parquet")
    .select(col("dlycaldt").cast(pl.Date).alias("date"))
    .unique()
    .sort("date")
    .collect()
)

# OPTIMIZATION 1: Single asof join instead of 6 joins in a loop
ccm3 = (
    ccm_merged
    .sort("rdq")
    .join_asof(
        trading_days,
        left_on="rdq",
        right_on="date",
        strategy="forward",
    )
    .rename({"date": "rdq_trad"})
    .select(["gvkey", "permno", "datadate", "fyearq", "fqtr", "rdq", "rdq_trad"])
)

print("=" * 10, "CRSP block ready", "=" * 10)

#############################
#    CRSP abnormal return   #
#############################
print("=" * 10, "Calculating abnormal returns", "=" * 10)

# Load CRSP daily data
crsp_d = (
    pl.scan_parquet(INPUT_PATH + "crsp_dsf.parquet")
    .select([
        col("permno").cast(pl.Int64),
        col("dlycaldt").cast(pl.Date).alias("date"),
        col("dlyret").alias("ret"),
    ])
    .collect()
)

# Load S&P 500 returns from index data
sp500 = (
    pl.scan_parquet(INPUT_PATH + "crsp_ind.parquet")
    .select([
        col("dlycaldt").cast(pl.Date).alias("date"),
        col("dlyprcret").alias("sprtrn"),
    ])
    .collect()
)

# Join to get abnormal returns
crsp_d = (
    crsp_d.join(sp500, on="date", how="left")
    .with_columns(
        (col("ret") - col("sprtrn")).alias("abrd")
    )
    .select(["date", "permno", "ret", "sprtrn", "abrd"])
)

print("=" * 10, "CRSP abnormal return ready", "=" * 10)

################################
#    Event window range join   #
################################
print("=" * 10, "Calculating event window returns", "=" * 10)

# Add window bounds to CCM data
ccm3 = ccm3.filter(col("rdq_trad").is_not_null()).with_columns([
    (col("rdq_trad") - pl.duration(days=10)).alias("minus10d"),
    (col("rdq_trad") + pl.duration(days=5)).alias("plus5d"),
])

# OPTIMIZATION 2: Use DuckDB with parallel execution
con = duckdb.connect(":memory:")
con.execute("SET threads TO 8;")  # Parallel execution

con.register("ccm_data", ccm3.to_arrow())
con.register("crsp_data", crsp_d.to_arrow())

df = con.execute("""
    SELECT a.gvkey, a.permno, a.datadate, a.fyearq, a.fqtr, 
           a.rdq, a.rdq_trad, b.date, b.abrd
    FROM ccm_data a 
    LEFT JOIN crsp_data b 
    ON a.permno = b.permno 
    AND a.minus10d <= b.date 
    AND b.date <= a.plus5d
    ORDER BY a.permno, a.rdq_trad, b.date
""").pl()

# Filter out missing returns
df = df.filter(col("abrd").is_not_null())

###############################
#    Count trading days       #
###############################
print("=" * 10, "Counting trading days", "=" * 10)

df = df.sort(["permno", "rdq_trad", "date"])

# Assign direction indicator: 0 = rdq, positive = after, negative = before
df = df.with_columns(
    pl.when(col("date") == col("rdq_trad")).then(pl.lit(0))
    .when(col("date") > col("rdq_trad")).then(pl.lit(1))
    .when(col("date") < col("rdq_trad")).then(pl.lit(-1))
    .alias("c_1")
)

# Trading days before rdq_trad (count descending: -1, -2, -3, ...)
df_before = (
    df.filter(col("c_1") == -1)
    .sort(["permno", "rdq_trad", "date"], descending=[False, False, True])
    .with_columns(
        (-(pl.int_range(pl.len()).over(["permno", "rdq_trad"]) + 1)).alias("count")
    )
    .sort(["permno", "rdq_trad", "date"])
)

# Trading days on or after rdq_trad (count: 0, 1, 2, ...)
df_after = (
    df.filter(col("c_1") >= 0)
    .with_columns(
        pl.int_range(pl.len()).over(["permno", "rdq_trad"]).alias("count")
    )
)

df = pl.concat([df_before, df_after])

###############################
#    Calculate ABR            #
###############################
print("=" * 10, "Calculating ABR", "=" * 10)

# Filter to event window [-2, +1]
df = df.filter((col("count") >= -2) & (col("count") <= 1))

# Sum abnormal returns by group
df_abr = (
    df.group_by(["permno", "rdq_trad"])
    .agg(col("abrd").sum().alias("abr"))
)

# Join ABR back and keep only count == 1 rows (rdq + 1 day)
df = (
    df.join(df_abr, on=["permno", "rdq_trad"], how="left")
    .filter(col("count") == 1)
    .rename({"date": "rdq_plus_1d"})
    .select(["gvkey", "permno", "datadate", "rdq", "rdq_plus_1d", "abr"])
)

###############################
#    Populate to monthly      #
###############################
print("=" * 10, "Populating to monthly", "=" * 10)

# Get monthly dates from CRSP monthly file
crsp_msf = (
    pl.scan_parquet(INPUT_PATH + "crsp_msf.parquet")
    .select(col("mthcaldt").cast(pl.Date).alias("date"))
    .unique()
    .collect()
)

# Add 12-month forward bound
df = df.with_columns(
    (col("datadate") + pl.duration(days=365)).dt.month_end().alias("plus12m")
)

# Use DuckDB for the range join (reuse connection)
con.register("df_data", df.to_arrow())
con.register("msf_data", crsp_msf.to_arrow())

df = con.execute("""
    SELECT a.gvkey, a.permno, a.datadate, a.rdq, a.rdq_plus_1d, a.abr, b.date
    FROM df_data a 
    LEFT JOIN msf_data b 
    ON a.rdq_plus_1d < b.date
    AND a.plus12m >= b.date
    ORDER BY a.permno, b.date, a.datadate DESC
""").pl()

con.close()

# Drop duplicates keeping first (most recent datadate per permno-month)
df = (
    df.unique(["permno", "date"], keep="first", maintain_order=True)
    .filter(col("date").is_not_null())
    .select(["gvkey", "permno", "datadate", "rdq", "rdq_plus_1d", "abr", "date"])
)

###############################
#    Write output             #
###############################
print("=" * 10, "Writing output", "=" * 10)

df.write_parquet(OUTPUT_PATH + "abr.parquet")
print(f"ABR data written to abr.parquet")
print(f"Total rows: {df.shape[0]:,}")
