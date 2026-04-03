# Calculate HSZ Replicating Anomalies
# SUE: Standardized Unexpected Earnings (Earnings surprise)

import polars as pl
import duckdb
from pathlib import Path
from datetime import date
from functions import INPUT_PATH, OUTPUT_PATH

###################
# Compustat Block #
###################
comp = (
    pl.scan_parquet(INPUT_PATH + "comp_fundq.parquet")
    .select(["gvkey", "datadate", "fyearq", "fqtr", "epspxq", "ajexq"])
    .collect()
)

###################
#    CCM Block    #
###################
ccm = (
    pl.scan_parquet(INPUT_PATH + "ccm.parquet")
    .with_columns([
        pl.col("linkenddt").fill_null(date.today())
    ])
    .collect()
)

# Merge comp with ccm
ccm1 = comp.join(ccm, on="gvkey", how="left")

# Set link date bounds
ccm2 = (
    ccm1
    .filter(
        (pl.col("datadate") >= pl.col("linkdt"))
        & (pl.col("datadate") <= pl.col("linkenddt"))
    )
    .select(["gvkey", "permno", "datadate", "fyearq", "fqtr", "epspxq", "ajexq"])
)

# Calculate EPS = epspxq / ajexq
ccm2 = ccm2.with_columns([
    (pl.col("epspxq") / pl.col("ajexq").replace(0, None)).alias("eps")
])
ccm2 = ccm2.unique(subset=["permno", "datadate"])

# Filter out null eps and sort
ccm2 = ccm2.filter(pl.col("eps").is_not_null())
ccm2 = ccm2.sort(["permno", "datadate"])

# Create count for each permno
ccm2 = ccm2.with_columns([
    pl.col("eps").cum_count().over("permno").alias("count")
])

# Create lag variables e1 to e12
ccm2 = ccm2.with_columns([
    pl.col("eps").shift(1).over("permno").alias("e1"),
    pl.col("eps").shift(2).over("permno").alias("e2"),
    pl.col("eps").shift(3).over("permno").alias("e3"),
    pl.col("eps").shift(4).over("permno").alias("e4"),
    pl.col("eps").shift(5).over("permno").alias("e5"),
    pl.col("eps").shift(6).over("permno").alias("e6"),
    pl.col("eps").shift(7).over("permno").alias("e7"),
    pl.col("eps").shift(8).over("permno").alias("e8"),
    pl.col("eps").shift(9).over("permno").alias("e9"),
    pl.col("eps").shift(10).over("permno").alias("e10"),
    pl.col("eps").shift(11).over("permno").alias("e11"),
    pl.col("eps").shift(12).over("permno").alias("e12"),
])

# Compute YoY differences: e{i}_diff = e{i} - e{i+4}
# Matches SIZ: std is computed on 4-quarter EPS differences, not raw levels
ccm2 = ccm2.with_columns([
    (pl.col(f"e{i}") - pl.col(f"e{i+4}")).alias(f"e{i}_diff")
    for i in range(1, 9)
])

# Calculate sue_std based on count
# Using row-wise std on YoY differences, requires >= 6 non-null values
cols_6 = ["e6_diff", "e5_diff", "e4_diff", "e3_diff", "e2_diff", "e1_diff"]
cols_7 = ["e7_diff", "e6_diff", "e5_diff", "e4_diff", "e3_diff", "e2_diff", "e1_diff"]
cols_8 = ["e8_diff", "e7_diff", "e6_diff", "e5_diff", "e4_diff", "e3_diff", "e2_diff", "e1_diff"]

def std_with_zero_handling(cols, min_valid=6):
    """Row-wise std of YoY EPS differences; returns None if < min_valid non-null, 0 if all equal."""
    concat_expr = pl.concat_list(cols)
    valid_count = concat_expr.list.eval(pl.element().is_not_null().cast(pl.Int32).sum()).list.first()
    all_equal = concat_expr.list.max() == concat_expr.list.min()
    std_val = concat_expr.list.eval(pl.element().std()).list.first()
    return (
        pl.when(valid_count < min_valid).then(pl.lit(None, dtype=pl.Float64))
        .when(all_equal).then(pl.lit(0.0))
        .otherwise(std_val)
    )

ccm2 = ccm2.with_columns([
    pl.when(pl.col("count") <= 6).then(None)
      .when(pl.col("count") == 7).then(std_with_zero_handling(cols_6))
      .when(pl.col("count") == 8).then(std_with_zero_handling(cols_7))
      .otherwise(std_with_zero_handling(cols_8))
      .alias("sue_std")
])

# Calculate SUE
ccm2 = ccm2.with_columns([
    ((pl.col("eps") - pl.col("e4")) / pl.col("sue_std").replace(0, None)).alias("sue")
])

print(f"Calculated SUE: {ccm2.shape}")

###################
# Monthly CRSP    #
###################
crsp_msf = (
    pl.scan_parquet(INPUT_PATH + "crsp_msf.parquet")
    .select(pl.col("mthcaldt").alias("date"))
    .unique()
    .collect()
)

# Add plus12m column
ccm2 = ccm2.with_columns([
    (pl.col("datadate").dt.offset_by("12mo").dt.month_end()).alias("plus12m")
])

###################
# Populate to Monthly (using DuckDB for inequality join)
###################
con = duckdb.connect(":memory:")
con.register("ccm2", ccm2.to_arrow())
con.register("crsp_msf", crsp_msf.to_arrow())

df = con.execute("""
    SELECT a.gvkey, a.permno, a.datadate, b.date, a.sue
    FROM ccm2 a 
    LEFT JOIN crsp_msf b 
        ON a.datadate <= b.date
        AND a.plus12m >= b.date
    ORDER BY a.permno, b.date, a.datadate DESC
""").pl()

con.close()

# Keep first (most recent datadate) for each permno-date
df = df.unique(subset=["permno", "date"], keep="first")
df = df.filter(pl.col("date").is_not_null())  # Remove rows where no CRSP date matched
df = df.select(["gvkey", "permno", "datadate", "date", "sue"])

print(f"Final shape: {df.shape}")

###################
# Save Output     #
###################
df.write_parquet(OUTPUT_PATH + "sue.parquet")
print(f"Saved to {OUTPUT_PATH + 'sue.parquet'}")