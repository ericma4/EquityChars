# Calculate HXZ Replicating Anomalies
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
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr", "epspxq", "ajexq"])
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
    .select(["gvkey", "permno", "datadate", "rdq", "fyearq", "fqtr", "epspxq", "ajexq"])
)

# Calculate EPS = epspxq / ajexq
ccm2 = ccm2.with_columns([
    (pl.col("epspxq") / pl.col("ajexq").replace(0, None)).alias("eps")
])
ccm2 = ccm2.unique(subset=["permno", "datadate"])

# Filter out null eps
ccm2 = ccm2.filter(pl.col("eps").is_not_null())

# 20261030fix-HXZ A.1.1: an rdq on or before the quarter end is erroneous; drop it.
ccm2 = ccm2.with_columns(pl.when(pl.col("rdq") > pl.col("datadate")).then(pl.col("rdq")).alias("rdq"))

# 20261030fix-Lag k = the same company's quarter 3k months earlier (row shifts skipped missing
# quarters); history comes from Compustat by gvkey, including quarters before the CRSP link.
hist = (comp.select("gvkey", "datadate", (pl.col("epspxq") / pl.col("ajexq").replace(0, None)).alias("eps"))
        .drop_nulls("eps").unique(["gvkey", "datadate"]))
for k in range(1, 13):
    ccm2 = ccm2.join(hist.with_columns(pl.col("datadate").dt.offset_by(f"{3 * k}mo").dt.month_end())
                     .rename({"eps": f"e{k}"}), on=["gvkey", "datadate"], how="left")

# Compute YoY differences: e{i}_diff = e{i} - e{i+4}
# Matches SIZ: std is computed on 4-quarter EPS differences, not raw levels
ccm2 = ccm2.with_columns([
    (pl.col(f"e{i}") - pl.col(f"e{i+4}")).alias(f"e{i}_diff")
    for i in range(1, 9)
])

# Std of YoY differences over the prior eight quarters, at least six non-null (HXZ A.1.1)
cols_8 = ["e8_diff", "e7_diff", "e6_diff", "e5_diff", "e4_diff", "e3_diff", "e2_diff", "e1_diff"]

def std_with_zero_handling(cols, min_valid=6):
    """Row-wise std of YoY EPS differences; returns None if < min_valid non-null, 0 if all equal."""
    # 20261030fix-Count non-null differences directly instead of counting records.
    concat_expr = pl.concat_list(cols).list.drop_nulls()
    return (
        pl.when(concat_expr.list.len() < min_valid).then(pl.lit(None, dtype=pl.Float64))
        .when(concat_expr.list.max() == concat_expr.list.min()).then(pl.lit(0.0))
        .otherwise(concat_expr.list.std(ddof=1))
    )

ccm2 = ccm2.with_columns([std_with_zero_handling(cols_8).alias("sue_std")])

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
    .select(pl.col("mthcaldt").dt.month_end().alias("date"))
    .unique()
    .collect()
)

# 20261030fix-HXZ A.1.1: the quarter must end within six months before formation;
# before 1972 use quarters ending at least four months earlier, from 1972 on use rdq.
ccm2 = ccm2.with_columns([
    (pl.col("datadate").dt.offset_by("6mo").dt.month_end()).alias("plus6m"),
    (pl.col("datadate").dt.offset_by("4mo").dt.month_end()).alias("plus4m")
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
        ON a.datadate < b.date
        AND a.plus6m >= b.date
        -- 20261030fix-was datadate <= date, which used SUE before it was announced
        AND (CASE WHEN b.date < DATE '1972-01-01' THEN a.plus4m ELSE a.rdq END) <= b.date
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