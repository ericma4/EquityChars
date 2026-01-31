"""
CAPM Beta Calculation
"""

import polars as pl
from polars import col
import time
import os


def measure_time(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function       : {func.__name__.upper()}", flush=True)
        print(f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}", flush=True)
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}", flush=True)
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        print(f"Execution time : {minutes} minutes and {seconds:.2f} seconds", flush=True)
        print()
        return result
    return wrapper


def gen_MMYY_column(date_col):
    """Generate YYYYMM integer from date column for grouping"""
    return (col(date_col).dt.year() * 100 + col(date_col).dt.month()).cast(pl.Int32)


def gen_consecutive_lists(input_list, k):
    """Split a list into consecutive, non-overlapping sublists of length k"""
    return [
        input_list[i : i + k]
        for i in range(0, len(input_list), k)
        if len(input_list[i : i + k]) == k
    ]


def build_groups(input_list, k):
    """Build k staggered groupings (offset windows) over a list"""
    return [gen_consecutive_lists(input_list[offset:], k) for offset in range(k)]


def group_mapping_dfs(input_list, k):
    """
    Create mapping DataFrames linking aux_date to group_number, 
    and group_number to new (max) aux_date
    """
    groups = build_groups(input_list, k)
    dfs = [
        pl.DataFrame({"aux_date": group}).with_columns(
            group_number=pl.cum_count("aux_date"), 
            new_date=col("aux_date").list.max()
        )
        for group in groups
    ]
    return [
        {
            "group_map": df.explode("aux_date")
            .select([col("aux_date").cast(pl.Int32), "group_number"])
            .lazy(),
            "date_map": df.select([
                "group_number", 
                col("new_date").alias("aux_date")
            ])
            .unique()
            .sort(["group_number"])
            .lazy(),
        }
        for df in dfs
    ]


def gen_aux_maps(n_months):
    """
    Generate rolling window mappings for n_months
    Returns list of {group_map, date_map} dicts
    """
    # Get unique months from 202001 to current
    # For 3-month windows, k=3
    k = n_months
    # Generate a range of YYYYMM values
    # This will be populated from actual data
    return k


def capm_beta(df, min_obs=21):
    """
    Compute CAPM beta using Polars group-by
    Formula: beta = cov(ret_exc, mktrf) / var(mktrf)
    """
    result = (
        df.group_by(["permno", "group_number"])
        .agg([
            (pl.cov("ret_exc", "mktrf") / pl.var("mktrf")).alias("beta"),
            pl.count("ret_exc").alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )
    return result


def process_map_chunk(base_data, mapping, min_obs=21):
    """
    Execute rolling computation for a mapping:
    1. Join base_data with mapping['group_map'] on aux_date
    2. Compute beta per (permno, group_number)
    3. Join mapping['date_map'] to remap to end date
    """
    result = (
        base_data
        .join(mapping["group_map"], on="aux_date", how="inner")
        .pipe(capm_beta, min_obs=min_obs)
        .join(mapping["date_map"], on="group_number", how="inner")
        .select(["permno", "aux_date", "beta"])
    )
    return result


@measure_time
def compute_rolling_beta(input_path, output_path, n_months=3, min_obs=21):
    """
    Main function to compute rolling CAPM beta
    
    Parameters:
    -----------
    input_path : str
        Path to input parquet file with daily returns and factors
    output_path : str
        Path to output parquet file
    n_months : int
        Number of months in rolling window (default: 3)
    min_obs : int
        Minimum observations required per window (default: 21)
    """
    
    # 1. Load data
    print(f"Loading data from {input_path}...", flush=True)
    df = pl.scan_parquet(input_path)
    
    # 2. Prepare data
    print("Preparing data...", flush=True)
    df = (
        df
        .rename({"dlycaldt": "date", "dlyret": "ret", "dlyvol": "vol"})
        .with_columns([
            col("date").cast(pl.Date),
            col("permno").cast(pl.Int64),
        ])
        .with_columns([
            # Compute excess return
            (col("ret") - col("rf")).alias("ret_exc"),
            # Add end-of-month date
            col("date").dt.month_end().alias("eom"),
        ])
        .with_columns([
            # Add aux_date (YYYYMM integer) for grouping
            gen_MMYY_column("eom").alias("aux_date"),
        ])
        .filter(
            col("ret_exc").is_not_null() & 
            col("mktrf").is_not_null() &
            col("vol").is_not_null()
        )
        .sort(["permno", "date"])
    )
    
    # 3. Get unique months for window mapping
    print("Generating rolling window mappings...", flush=True)
    unique_months = (
        df.select("aux_date")
        .unique()
        .sort("aux_date")
        .collect()["aux_date"]
        .to_list()
    )
    
    # Create rolling window mappings
    aux_maps = group_mapping_dfs(unique_months, n_months)
    
    # 4. Compute beta for each mapping and concatenate
    print(f"Computing {n_months}-month rolling betas (min {min_obs} obs)...", flush=True)
    results = []
    for i, mapping in enumerate(aux_maps):
        print(f"  Processing mapping {i+1}/{len(aux_maps)}...", flush=True)
        chunk_result = process_map_chunk(df, mapping, min_obs=min_obs)
        results.append(chunk_result)
    
    # Concatenate all results
    print("Concatenating results...", flush=True)
    final_result = pl.concat(results)
    
    # 5. Map back to actual dates and format output
    print("Mapping to final dates...", flush=True)
    
    # Create date mapping: aux_date -> last trading day of that month
    # This matches the original Pandas behavior
    date_map = (
        df.select(["aux_date", "date"])
        .group_by("aux_date")
        .agg(col("date").max().alias("date"))  # Last trading day of month
        .collect()
    )
    
    # Join to get actual dates
    output = (
        final_result
        .collect()
        .join(date_map, on="aux_date", how="inner")
        .select(["permno", "date", "beta"])
        .sort(["permno", "date"])
    )
    
    # 6. Write output
    print(f"Writing output to {output_path}...", flush=True)
    output.write_parquet(output_path)
    
    print(f"✓ Completed! Output shape: {output.shape}", flush=True)
    print(f"  Unique permnos: {output['permno'].n_unique()}", flush=True)
    print(f"  Date range: {output['date'].min()} to {output['date'].max()}", flush=True)
    
    return output


if __name__ == "__main__":
    # Configuration
    INPUT_PATH = "crsp_dsf_temp.parquet"
    OUTPUT_PATH = "beta_polars.parquet"
    N_MONTHS = 3  # 3-month rolling window
    MIN_OBS = 21  # Minimum 21 observations
    
    # Run computation
    result = compute_rolling_beta(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        n_months=N_MONTHS,
        min_obs=MIN_OBS
    )
    
    # Display sample results
    print("\nSample results:")
    print(result.head(10))
