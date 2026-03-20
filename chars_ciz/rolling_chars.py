"""
Rolling Window Characteristics using Polars
Rewritten from Pandas+multiprocessing to Polars for 10-100x speedup

Characteristics included:
- beta: CAPM beta
- beta_ff5: Fama-French 5-factor market beta
- baspread: Bid-ask spread
- ill: Amihud (2002) illiquidity measure
- maxret: Maximum daily return
- rvar_capm: CAPM residual variance
- rvar_ff3: FF3 residual variance
- rvar_mean: Return variance
- std_dolvol: Std of log dollar volume
- std_turn: Std of turnover
- zerotrade: Zero trading days measure
"""

import polars as pl
from polars import col
import polars_ols as pls
import time
from functions import INPUT_PATH, OUTPUT_PATH

# =============================================================================
# Utility Functions
# =============================================================================

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


# =============================================================================
# Characteristic Calculation Functions
# =============================================================================

def capm_beta(df, min_obs=21):
    """
    CAPM beta = cov(exret, mktrf) / var(mktrf)
    Also computes variance of residuals from regression with intercept
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            (pl.cov("exret", "mktrf") / pl.var("mktrf")).alias("beta"),
            # Residual variance from regression with intercept:
            # residual = (exret - mean(exret)) - beta * (mktrf - mean(mktrf))
            (
                (col("exret") - pl.mean("exret")) 
                - (pl.cov("exret", "mktrf") / pl.var("mktrf")) * (col("mktrf") - pl.mean("mktrf"))
            ).var().alias("rvar_capm"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def maxret(df, min_obs=21):
    """
    Maximum daily return over the rolling window
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            pl.max("ret").alias("maxret"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def rvar_mean(df, min_obs=21):
    """
    Variance of raw returns (no factor adjustment)
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            pl.var("ret").alias("rvar_mean"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def baspread(df, min_obs=21):
    """
    Mean relative bid-ask spread: (askhi - bidlo) / ((askhi + bidlo) / 2)
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            ((col("askhi") - col("bidlo")) / ((col("askhi") + col("bidlo")) / 2)).mean().alias("baspread"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def std_dolvol(df, min_obs=21):
    """
    Standard deviation of log dollar volume: std(log(|vol * prc|))
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            ((col("vol") * col("prc").abs()).log()).std().alias("std_dolvol"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def std_turn(df, min_obs=21):
    """
    Standard deviation of daily turnover: std(vol / shrout)
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            (col("vol") / col("shrout")).std().alias("std_turn"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def zerotrade(df, min_obs=21):
    """
    Zero trading days measure:
    zerotrade = (zero_count + (1/turnover_sum)/11000) * 63 / n_obs
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            (col("vol") == 0).sum().alias("zero_count"),
            (col("vol") / col("shrout")).sum().alias("turn_sum"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .with_columns([
            (
                (col("zero_count") + (1.0 / col("turn_sum")) / 11000) 
                * 63.0 / col("n_obs")
            ).alias("zerotrade")
        ])
        .select(["permno", "group_number", "zerotrade"])
    )


def ill(df, min_obs=21):
    """
    Amihud (2002) illiquidity measure:
    ill = mean(abs(ret) / (abs(prc) * vol))
    """
    return (
        df.group_by(["permno", "group_number"])
        .agg([
            (col("ret").abs() / (col("prc").abs() * col("vol"))).mean().alias("ill"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )


def beta_ff5(df, min_obs=21):
    """
    Fama-French 5-factor market beta using polars_ols
    Returns the coefficient on mktrf from regression:
    exret ~ mktrf + smb + hml + rmw + cma
    """
    # Use polars_ols least_squares method with string column names
    res_exp = pl.col("exret").least_squares.ols(
        "mktrf", "smb", "hml", "rmw", "cma",
        add_intercept=True, 
        mode="coefficients"
    )
    result = (
        df.filter(
            col("mktrf").is_not_null() & 
            col("smb").is_not_null() & 
            col("hml").is_not_null() &
            col("rmw").is_not_null() &
            col("cma").is_not_null()
        )
        .group_by(["permno", "group_number"])
        .agg([
            res_exp.first().struct.field("mktrf").alias("beta_ff5"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )
    return result


def rvar_ff3(df, min_obs=21):
    """
    Fama-French 3-factor residual variance using polars_ols
    Computes var(residuals) from: exret ~ mktrf + smb + hml
    """
    res_exp = pl.col("exret").least_squares.ols(
        "mktrf", "smb", "hml",
        add_intercept=True,
        mode="residuals"
    )
    result = (
        df.filter(
            col("mktrf").is_not_null() & 
            col("smb").is_not_null() & 
            col("hml").is_not_null()
        )
        .group_by(["permno", "group_number"])
        .agg([
            res_exp.var().alias("rvar_ff3"),
            pl.len().alias("n_obs"),
        ])
        .filter(col("n_obs") >= min_obs)
        .drop("n_obs")
    )
    return result


# =============================================================================
# Processing Functions
# =============================================================================

def process_map_chunk(base_data, mapping, char_func, min_obs=21):
    """
    Execute rolling computation for a mapping:
    1. Join base_data with mapping['group_map'] on aux_date
    2. Compute characteristic per (permno, group_number)
    3. Join mapping['date_map'] to remap to end date
    """
    result = (
        base_data
        .join(mapping["group_map"], on="aux_date", how="inner")
        .pipe(char_func, min_obs=min_obs)
        .join(mapping["date_map"], on="group_number", how="inner")
    )
    return result


def compute_single_char(df, aux_maps, char_func, char_names, min_obs=21):
    """
    Compute a single characteristic across all mapping chunks
    
    Parameters:
    -----------
    char_names : str or list of str
        Column name(s) to extract from the characteristic function
    """
    results = []
    for mapping in aux_maps:
        chunk_result = process_map_chunk(df, mapping, char_func, min_obs=min_obs)
        results.append(chunk_result)
    
    # Handle both single column name (str) and multiple column names (list)
    if isinstance(char_names, str):
        char_names = [char_names]
    
    combined = pl.concat(results).select(["permno", "aux_date"] + char_names)
    return combined


@measure_time
def compute_all_rolling_chars(input_path, output_path, n_months=3, min_obs=21):
    """
    Main function to compute all rolling characteristics
    
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
    
    # 2. Prepare data - cast Decimal to Float64 for polars_ols compatibility
    print("Preparing data...", flush=True)
    df = (
        df
        .rename({
            "dlycaldt": "date", 
            "dlyret": "ret", 
            "dlyvol": "vol",
            "dlyprc": "prc",
            "dlyhigh": "askhi",
            "dlylow": "bidlo",
        })
        .with_columns([
            col("date").cast(pl.Date),
            col("permno").cast(pl.Int64),
            (col("shrout") * 1000).cast(pl.Float64).alias("shrout"),
            # Cast all Decimal columns to Float64
            col("ret").cast(pl.Float64),
            col("vol").cast(pl.Float64),
            col("prc").cast(pl.Float64),
            col("askhi").cast(pl.Float64),
            col("bidlo").cast(pl.Float64),
            col("rf").cast(pl.Float64),
            col("mktrf").cast(pl.Float64),
            col("smb").cast(pl.Float64),
            col("hml").cast(pl.Float64),
            col("rmw").cast(pl.Float64),
            col("cma").cast(pl.Float64),
        ])
        .with_columns([
            (col("ret") - col("rf")).alias("exret"),
            col("date").dt.month_end().alias("monthend"),
        ])
        .with_columns([
            gen_MMYY_column("monthend").alias("aux_date"),
        ])
        .filter(
            col("ret").is_not_null() & 
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
    
    aux_maps = group_mapping_dfs(unique_months, n_months)
    
    # 4. Compute each characteristic
    print(f"Computing {n_months}-month rolling characteristics (min {min_obs} obs)...", flush=True)
    
    char_configs = [
        (capm_beta, ["beta", "rvar_capm"]),  # Extract both columns from capm_beta
        (maxret, "maxret"),
        (rvar_mean, "rvar_mean"),
        (baspread, "baspread"),
        (std_dolvol, "std_dolvol"),
        (std_turn, "std_turn"),
        (zerotrade, "zerotrade"),
        (ill, "ill"),
        (beta_ff5, "beta_ff5"),
        (rvar_ff3, "rvar_ff3"),
    ]
    
    all_results = None
    for char_func, char_names in char_configs:
        # Handle display name for logging
        display_name = char_names if isinstance(char_names, str) else ", ".join(char_names)
        print(f"  Computing {display_name}...", flush=True)
        char_result = compute_single_char(df, aux_maps, char_func, char_names, min_obs).collect()
        
        if all_results is None:
            all_results = char_result
        else:
            all_results = all_results.join(
                char_result, 
                on=["permno", "aux_date"], 
                how="full",
                coalesce=True
            )
    
    # 5. Map back to actual dates
    print("Mapping to final dates...", flush=True)
    
    date_map = (
        df.select(["aux_date", "date"])
        .group_by("aux_date")
        .agg(col("date").max().alias("date"))
        .collect()
    )
    
    output = (
        all_results
        .join(date_map, on="aux_date", how="inner")
        .drop("aux_date")
        .sort(["permno", "date"])
    )
    
    # 6. Write output
    print(f"Writing output to {output_path}...", flush=True)
    output.write_parquet(output_path)
    
    print(f"✓ Completed! Output shape: {output.shape}", flush=True)
    print(f"  Unique permnos: {output['permno'].n_unique()}", flush=True)
    print(f"  Date range: {output['date'].min()} to {output['date'].max()}", flush=True)
    print(f"  Columns: {output.columns}", flush=True)
    
    return output


if __name__ == "__main__":
    # Configuration
    INPUT_PATH = INPUT_PATH + "crsp_dsf.parquet"
    OUTPUT_PATH = OUTPUT_PATH + "rolling_chars.parquet"
    N_MONTHS = 3
    MIN_OBS = 21
    
    result = compute_all_rolling_chars(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        n_months=N_MONTHS,
        min_obs=MIN_OBS
    )
    
    print("\nSample results:")
    print(result.head(10))
