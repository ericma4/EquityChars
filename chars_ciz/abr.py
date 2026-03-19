# Calculate HSZ Replicating Anomalies
# ABR: Cumulative abnormal stock returns around earnings announcements
# Optimized version with Polars + DuckDB

import polars as pl
import duckdb
import os
from functions import INPUT_PATH, OUTPUT_PATH

num_threads = max(1, os.cpu_count() // 2)

###################
# Compustat Block #
###################
comp = (
    pl.scan_parquet(INPUT_PATH + "comp_fundq.parquet")
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr"])
    .with_columns(
        pl.col("datadate").cast(pl.Date),
        pl.col("rdq").cast(pl.Date),
    )
    .collect()
)

###################
#    CCM Block    #
###################
ccm = (
    pl.scan_parquet(INPUT_PATH + "ccm.parquet")
    .select([
        "gvkey",
        pl.col("permno").cast(pl.Int64),
        pl.col("linkdt").cast(pl.Date),
        pl.col("linkenddt").cast(pl.Date),
    ])
    .with_columns(
        pl.col("linkenddt").fill_null(pl.date(2099, 12, 31))
    )
    .collect()
)

# Join comp with CCM and filter by link dates
ccm_merged = (
    comp.join(ccm, on="gvkey", how="left")
    .filter(
        (pl.col("datadate") >= pl.col("linkdt"))
        & (pl.col("datadate") <= pl.col("linkenddt"))
    )
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr", "permno"])
)

###################
#    CRSP Block   #
###################
# Map RDQ to first trading day on or after
# Get trading days from CRSP daily data
trading_days = (
    pl.scan_parquet(INPUT_PATH + "crsp_dsf.parquet")
    .select(pl.col("dlycaldt").cast(pl.Date).alias("date"))
    .unique()
    .sort("date")
    .collect()
)

# Find the closest trading day (rdq_trad) on or after rdq
ccm1 = (
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

#############################
#    CRSP abnormal return   #
#############################
# Load CRSP daily data
crsp_d = (
    pl.scan_parquet(INPUT_PATH + "crsp_dsf.parquet")
    .select([
        pl.col("permno").cast(pl.Int64),
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("dlyret").alias("ret"),
    ])
    .collect()
)

# Load S&P 500 returns from index data
sp500 = (
    pl.scan_parquet(INPUT_PATH + "crsp_ind.parquet")
    .select([
        pl.col("dlycaldt").cast(pl.Date).alias("date"),
        pl.col("dlyprcret").alias("sprtrn"),
    ])
    .collect()
)

# Join to get abnormal returns
crsp_d = (
    crsp_d.join(sp500, on="date", how="left")
    .with_columns(
        (pl.col("ret") - pl.col("sprtrn")).alias("abrd")
    )
    .select(["date", "permno", "ret", "sprtrn", "abrd"])
)

################################
#    Event window range join   #
################################
# Add window bounds to nearest trading day of rdq
ccm1 = ccm1.filter(pl.col("rdq_trad").is_not_null()).with_columns([
    (pl.col("rdq_trad") - pl.duration(days=10)).alias("minus10d"),
    (pl.col("rdq_trad") + pl.duration(days=5)).alias("plus5d"),
])

# Make sure the trading day version of rdq is within the window bounds
con = duckdb.connect(":memory:")
try:
    con.execute(f"SET threads TO {num_threads};")

    con.register("ccm_data", ccm1.to_arrow())
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
    df = df.filter(pl.col("abrd").is_not_null())

    ###############################
    #    Count trading days       #
    ###############################
    df = df.sort(["permno", "rdq_trad", "date"])

    # Assign direction indicator: 0 = rdq, positive = after, negative = before
    # This is used to count trading days before and after rdq_trad
    df = df.with_columns(
        pl.when(pl.col("date") == pl.col("rdq_trad")).then(pl.lit(0))
        .when(pl.col("date") > pl.col("rdq_trad")).then(pl.lit(1))
        .when(pl.col("date") < pl.col("rdq_trad")).then(pl.lit(-1))
        .alias("c_1")
    )

    # Trading days before rdq_trad (count descending: -1, -2, -3, ...)
    df_before = (
        df.filter(pl.col("c_1") == -1)
        .sort(["permno", "rdq_trad", "date"], descending=[False, False, True])
        .with_columns(
            (-(pl.col("date").cum_count().over(["permno", "rdq_trad"]))).alias("count")
        )
        .sort(["permno", "rdq_trad", "date"])
    )

    # Trading days on or after rdq_trad (count: 0, 1, 2, ...)
    df_after = (
        df.filter(pl.col("c_1") >= 0)
        .with_columns(
            (pl.col("date").cum_count().over(["permno", "rdq_trad"]) - 1).alias("count")
        )
    )

    df = pl.concat([df_before, df_after])

    ###############################
    #    Calculate ABR            #
    ###############################
    # Filter to event window [-2, +1]
    df = df.filter((pl.col("count") >= -2) & (pl.col("count") <= 1))

    # Sum abnormal returns by group
    df_abr = (
        df.group_by(["permno", "rdq_trad"])
        .agg(pl.col("abrd").sum().alias("abr"))
    )

    # Join ABR back and keep only count == 1 rows (rdq + 1 day)
    df = (
        df.join(df_abr, on=["permno", "rdq_trad"], how="left")
        .filter(pl.col("count") == 1)
        .rename({"date": "rdq_plus_1d"})
        .select(["gvkey", "permno", "datadate", "rdq", "rdq_plus_1d", "abr"])
    )

    ###############################
    #    Populate to monthly      #
    ###############################
    # Make sure the abr is used between rdq_plus_1d and plus12m
    # Get monthly dates from CRSP monthly file
    crsp_msf = (
        pl.scan_parquet(INPUT_PATH + "crsp_msf.parquet")
        .select(pl.col("mthcaldt").cast(pl.Date).alias("date"))
        .unique()
        .collect()
    )

    # Add 12-month forward bound
    df = df.with_columns(
        pl.col("datadate").dt.offset_by("12mo").dt.month_end().alias("plus12m")
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
finally:
    con.close()

# Drop duplicates keeping first (most recent datadate per permno-month)
df = (
    df.unique(["permno", "date"], keep="first", maintain_order=True)
    .filter(pl.col("date").is_not_null())
    .select(["gvkey", "permno", "datadate", "rdq", "rdq_plus_1d", "abr", "date"])
)

###############################
#    Write output             #
###############################
df.write_parquet(OUTPUT_PATH + "abr.parquet")
print(f"ABR data written to abr.parquet")
