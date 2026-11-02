# Calculate HXZ Replicating Anomalies
# SUE: Standardized Unexpected Earnings (Earnings surprise)

import polars as pl
import duckdb
from functions import company_months, INPUT_PATH, OUTPUT_PATH

###################
# Compustat Block #
###################
comp = (
    pl.scan_parquet(INPUT_PATH + "comp_fundq.parquet")
    .select(["gvkey", "datadate", "rdq", "fyearq", "fqtr", "epspxq", "ajexq"]) # 20261030fix-Read rdq, the Compustat quarterly earnings announcement date, to determine when earnings become available.
    .collect()
)

# Compute company earnings before stock/CCM filtering so the current company
# can use its own pre-link financial history. Link companies to stocks by month below.
ccm2 = comp

# Calculate EPS = epspxq / ajexq
# 20261030fix-Use Float64 to reduce numerical error while retaining the original EPS formula.
ccm2 = ccm2.with_columns([
    pl.col("epspxq").cast(pl.Float64),
    pl.col("ajexq").cast(pl.Float64),
])
ccm2 = ccm2.with_columns([
    (pl.col("epspxq") / pl.col("ajexq").replace(0, None)).alias("eps")
])
# 20261030fix-Match actual fiscal quarters and available announcements; row shifts and arbitrary deduplication can misalign history.

# 20261030fix-Keep missing quarters and use rdq for availability; leave conflicting EPS values within an announcement null.
ccm2 = ccm2.select("gvkey", "datadate", "rdq", "fyearq", "fqtr", "eps")
# 20261030fix-Leave conflicting EPS values for the same period and announcement null instead of selecting arbitrarily.
ccm2 = ccm2.with_columns(
    pl.when(pl.col("eps").drop_nulls().n_unique().over(["gvkey", "datadate", "rdq"]) <= 1)
    .then(pl.col("eps")).alias("eps")
)
# 20261030fix-Disabled: requiring RDQ for every year removes pre-1972 earnings histories.
# 20261030fix-HXZ A.1.1 uses a four-month lag before 1972; retain actual RDQ separately.
ccm2 = ccm2.with_columns(
    pl.when(pl.col("datadate") < pl.date(1972, 1, 1))
    .then(pl.max_horizontal(pl.col("datadate").dt.offset_by("4mo").dt.month_end(), "rdq"))
    .otherwise(pl.col("rdq")).alias("_sue_date")
).filter(pl.col("_sue_date").is_not_null() &
         (pl.col("rdq").is_null() | (pl.col("rdq") > pl.col("datadate"))))
ccm2 = (
    ccm2
    .with_columns(
        (pl.col("datadate").dt.year() * 12 + pl.col("datadate").dt.month()).alias("period_id"),
    )
    .unique(maintain_order=True)
    .with_row_index("event_id")
)
con = duckdb.connect()
con.register("ccm2", ccm2.to_arrow())
# 20261030fix-Match history by 3*k months and require prior announcements to be available.
for k in range(1, 13):
    history = con.sql(f"""
        SELECT a.event_id, b.eps AS e{k}
        FROM ccm2 a LEFT JOIN ccm2 b
          ON a.gvkey=b.gvkey AND b.period_id=a.period_id-{3*k}
         -- 20261030fix-Use the pre-1972 reporting lag where RDQ is unavailable.
         AND b._sue_date<=a._sue_date
        QUALIFY row_number() OVER (
          PARTITION BY a.event_id
          ORDER BY b.rdq DESC, b.eps DESC NULLS LAST, b.fyearq DESC, b.fqtr DESC
        )=1
    """).pl()
    ccm2 = ccm2.join(history, on="event_id", how="left")
con.close()

# Compute YoY differences: e{i}_diff = e{i} - e{i+4}
# Matches SIZ: std is computed on 4-quarter EPS differences, not raw levels
ccm2 = ccm2.with_columns([
    (pl.col(f"e{i}") - pl.col(f"e{i+4}")).alias(f"e{i}_diff")
    for i in range(1, 9)
])

# Calculate sue_std based on count
# Using row-wise std on YoY differences, requires >= 6 non-null values
cols_8 = ["e8_diff", "e7_diff", "e6_diff", "e5_diff", "e4_diff", "e3_diff", "e2_diff", "e1_diff"]

def std_with_zero_handling(cols, min_valid=6):
    """Row-wise std of YoY EPS differences; returns None if < min_valid non-null, 0 if all equal."""
    # 20261030fix-Count and measure the same non-null differences directly.
    concat_expr = pl.concat_list(cols).list.drop_nulls()
    valid_count = concat_expr.list.len()
    all_equal = concat_expr.list.max() == concat_expr.list.min()
    # 20261030fix-Exclude missing values explicitly before computing sample standard deviation to preserve verified results.
    std_val = concat_expr.list.std(ddof=1)
    return (
        pl.when(valid_count < min_valid).then(pl.lit(None, dtype=pl.Float64))
        .when(all_equal).then(pl.lit(0.0))
        .otherwise(std_val)
    )

# 20261030fix-Count valid year-over-year changes within eight quarters rather than counting records.
ccm2 = ccm2.with_columns([
    std_with_zero_handling(list(reversed(cols_8))).alias("sue_std")
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
    # 20261030fix-Keep the stock identifier for monthly matching.
    .select(pl.col("permno").cast(pl.Int64), pl.col("mthcaldt").alias("date"))
    .collect()
)

# Earnings belong to the company linked at the observation month, not a predecessor.
company_keys = crsp_msf.select('permno', pl.col('date').dt.month_end().alias('jdate'))
crsp_msf = (crsp_msf.with_columns(pl.col('date').dt.month_end().alias('jdate'))
    .join(company_months(company_keys, INPUT_PATH), on=['permno', 'jdate'],
          how='left', validate='m:1').drop('jdate'))

# Add plus6m column
ccm2 = ccm2.with_columns([
    (pl.col("datadate").dt.offset_by("6mo").dt.month_end()).alias("plus6m")
])

###################
# Populate to Monthly (using DuckDB for inequality join)
###################
con = duckdb.connect(":memory:")
con.register("ccm2", ccm2.to_arrow())
con.register("crsp_msf", crsp_msf.to_arrow())

# 20261030fix-Match by stock and announcement date to prevent pre-announcement values.
# 20261030fix-Preserve the verified stock-month grid, announcement dates, and existing keep-first ordering.
# 20261030fix-Select the two HXZ reporting regimes separately; a conditional outer join causes a full-history scan.
df = con.sql("""
    SELECT a.gvkey, b.permno, a.datadate, b.date, a.sue,
           a.fyearq, a.fqtr, a.rdq
    FROM (SELECT * FROM crsp_msf WHERE date < DATE '1972-01-01') b
    LEFT JOIN ccm2 a ON a.gvkey = b._current_gvkey
        AND a._sue_date <= last_day(b.date)
        AND a.plus6m >= b.date
    UNION ALL
    SELECT a.gvkey, b.permno, a.datadate, b.date, a.sue,
           a.fyearq, a.fqtr, a.rdq
    FROM (SELECT * FROM crsp_msf WHERE date >= DATE '1972-01-01') b
    LEFT JOIN ccm2 a ON a.gvkey = b._current_gvkey
        AND a.rdq < b.date
        AND a.plus6m >= b.date
    ORDER BY permno, date, datadate DESC, rdq DESC, fyearq DESC, fqtr DESC
""").pl()

con.close()

# Keep first (most recent datadate) for each permno-date
df = df.unique(subset=["permno", "date"], keep="first")
# 20261030fix-Keep announcement dates to verify when the signal becomes available.
df = df.select(["permno", "date", "gvkey", "datadate", "fyearq", "fqtr", "rdq", "sue"])

print(f"Final shape: {df.shape}")

###################
# Save Output     #
###################
df.write_parquet(OUTPUT_PATH + "sue.parquet")
print(f"Saved to {OUTPUT_PATH + 'sue.parquet'}")
