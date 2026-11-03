import os
import sys
import argparse
import getpass
import duckdb
import time
from dateutil.relativedelta import relativedelta


# Configuration
OUTPUT_PATH = "../data/raw/"
FULL_START = '1940-01-01'

os.makedirs(OUTPUT_PATH, exist_ok=True)


def measure_time(func):
    """
    Decorator to time a function and print start/end timestamps and elapsed minutes:seconds.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        print(f"Function       : {func.__name__.upper()}", flush=True)
        print(
            f"Start          : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}",
            flush=True,
        )
        result = func(*args, **kwargs)
        end_time = time.time()
        print(
            f"End            : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
            flush=True,
        )
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        print(
            f"Execution time : {minutes} minutes and {seconds:.2f} seconds", flush=True
        )
        print()
        return result
    return wrapper


def gen_wrds_connection_info(user, password=None):
    """Generate WRDS PostgreSQL connection string for DuckDB (password may come from ~/.pgpass)."""
    return (
        f"host=wrds-pgdata.wharton.upenn.edu "
        f"port=9737 dbname=wrds "
        f"user={user} " + (f"password={password} " if password else "") + "sslmode=require"
    )


def _execute_download(con, table_name, query, output_file):
    """
    Helper function to execute a single table download.
    
    Args:
        con: DuckDB connection (already attached to WRDS)
        table_name: Name of table for logging
        query: SQL query to execute on WRDS
        output_file: Path to save parquet file
    """
    print(f"Downloading {table_name}...", flush=True)
    
    con.execute(f"""
        COPY (
            SELECT * FROM postgres_query('wrds', '{query}')
        )
        TO '{output_file}' (FORMAT PARQUET);
    """)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"{table_name} saved to {output_file} ({file_size_mb:.1f} MB)", flush=True)


# ====================================================================================================
# TABLE DEFINITIONS - Edit these queries to modify what data to download
# ====================================================================================================

def get_tables_config(start_date=FULL_START):
    """
    Define all tables to download with their queries.
    Edit the queries in this function to modify what data to download.
    
    Args:
        start_date: Start date for filtering (format: 'YYYY-MM-DD')
        
    Returns:
        Dictionary of table configurations with 'output' and 'query' keys
    """
    return {
        'comp_funda': {
            'output': os.path.join(OUTPUT_PATH, 'comp_funda.parquet'),
            'date_col': 'datadate', 'overlap_months': 24,   # late filings and restatements
            'query': f"""
                SELECT 
                    f.gvkey, f.cusip, f.datadate, f.fyear, c.cik, substr(c.sic,1,2) as sic2, c.sic, c.naics,
                    
                    /* income statement */
                    f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda,
                    f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni, f.txfed, f.txfo, f.txt, f.xint,
                    
                    /* CF statement and others */
                    f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo, f.mib, f.oiadp, f.ivao,
                    
                    /* assets */
                    f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,
                    
                    /* liabilities */
                    f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, 
                    f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi,
                    
                    /* equity and other */
                    f.ceq, f.scstkc, f.emp, f.csho, f.seq, f.txditc, f.pstkrv, f.pstkl, f.np, f.txdc, f.dpc, f.ajex, f.conm,
                    
                    /* market */
                    ABS(f.prcc_f) AS prcc_f
                FROM comp.funda AS f
                LEFT JOIN comp.company AS c 
                    ON f.gvkey = c.gvkey
                WHERE f.indfmt = ''INDL'' 
                AND f.datafmt = ''STD''
                AND f.popsrc = ''D''
                AND f.consol = ''C''
                AND f.datadate >= ''{start_date}''
            """
        },
# (fixed): add cfacshr(crsp_msf: mthcumfacshr), wcaptq (is wcapq)
        'comp_fundq': {
            'output': os.path.join(OUTPUT_PATH, 'comp_fundq.parquet'),
            'date_col': 'datadate', 'overlap_months': 24,
            'query': f"""
                SELECT 
                    /*header info*/
                    c.gvkey, f.cusip, f.datadate, f.fyearq,  substr(c.sic,1,2) as sic2, c.sic, f.fqtr, f.rdq,

                    /*income statement*/
                    f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq, f.revty, f.cogsy, f.saley,

                    /*balance sheet items*/
                    f.atq, f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq, f.ppegtq, f.txpq,

                    /*others*/
                    abs(f.prccq) as prccq, abs(f.prccq)*f.cshoq as mveq_f, f.ceqq, f.seqq, f.pstkq, f.ltq,
                    f.pstkrq, f.gdwlq, f.intanq, f.mibq, f.oiadpq, f.ivaoq, f.conm,
                    
                    /* v3 my formula add*/
                    f.ajexq, f.cshoq, f.txditcq, f.npq, f.xrdy, f.xrdq, f.dpq, f.xintq, f.invtq, f.scstkcy, f.niq,
                    f.oancfy, f.wcapq, f.dlttq, f.rectq, f.acoq, f.apq, f.lcoq, f.loq, f.aoq,
                    
                    /* SUE calculation */
                    f.epspxq

                FROM comp.fundq as f
                LEFT JOIN comp.company as c
                ON f.gvkey = c.gvkey

                /*get consolidated, standardized, industrial format statements*/
                WHERE f.indfmt = ''INDL'' 
                AND f.datafmt = ''STD''
                AND f.popsrc = ''D''
                AND f.consol = ''C''
                AND f.datadate >= ''{start_date}''
            """
        },
# cfacshr - is mthcumfacshr
        'crsp_msf': {
            'output': os.path.join(OUTPUT_PATH, 'crsp_msf.parquet'),
            'date_col': 'mthcaldt', 'overlap_months': 3,
            'query': f"""
                SELECT 
                    mthprc, mthret, mthretx, mthvol,
                    shrout, mthcumfacpr, mthcumfacshr,
                    permno, permco, mthcaldt, ticker, cusip, hdrcusip,
                    issuernm, issuertype, securitytype, securitysubtype, sharetype, usincflg,
                    primaryexch, conditionaltype, TradingStatusFlg
                FROM crspq.msf_v2
                WHERE mthcaldt >= ''{start_date}''
            """
        },

        # 2026-02-10 updates: confirm and use ccmxpf_lnkhist
        'ccm': {
            'output': os.path.join(OUTPUT_PATH, 'ccm.parquet'),
            'date_col': None, 'overlap_months': 0,   # small link table; links change in place, always full
            'query': """
                SELECT 
                    gvkey, lpermno as permno, linktype, linkprim, 
                    linkdt, linkenddt
                FROM crsp.ccmxpf_lnkhist
                WHERE linktype IN (''LC'', ''LU'', ''LS'')
            """
        },

        'crsp_dsf': {
            'output': os.path.join(OUTPUT_PATH, 'crsp_dsf.parquet'),
            'date_col': 'dlycaldt', 'overlap_months': 3,
            'query': f"""
                SELECT 
                    a.permno, a.permco, a.dlycaldt, a.dlyret, a.dlyvol, a.dlyprc, a.dlyhigh, a.dlylow, 
                    a.shrout, a.dlydelflg, a.dlycumfacpr, a.dlycumfacshr, 
                    a.primaryexch, a.conditionaltype, a.tradingstatusflg,
                    a.cusip, a.hdrcusip, a.siccd, 
                    b.rf, b.mktrf, b.smb, b.hml, b.umd, b.rmw, b.cma
                FROM crspq.dsf_v2 as a
                LEFT JOIN ff_all.fivefactors_daily as b
                ON a.dlycaldt = b.date
                WHERE a.dlycaldt >= ''{start_date}''
            """
        },

        'crsp_ind': {
            'output': os.path.join(OUTPUT_PATH, 'crsp_ind.parquet'),
            'date_col': 'dlycaldt', 'overlap_months': 3,
            'query': f"""
                SELECT
                    dlycaldt, dlyprcret
                FROM crspq.inddlyseriesdata_ind
                WHERE indno = 1000502  /*industry code for S&P 500 Composite*/
                AND dlycaldt >= ''{start_date}''
            """
        },

        'ibes': {
            'output': os.path.join(OUTPUT_PATH, 'ibes.parquet'),
            # a forecast month appears only after the actual is announced (statpers < anndats_act),
            # up to about two years later for FY2
            'date_col': 'statpers', 'overlap_months': 24,
            'query': f"""
                SELECT
                    ticker, statpers, meanest, fpedats, anndats_act, curr_act, fpi, medest
                FROM ibes.statsum_epsus
                WHERE
                    /* filtering IBES */
                    statpers < ANNDATS_ACT      /*only keep summarized forecasts prior to earnings annoucement*/
                AND measure=''EPS''
                /* 20261030fix-HXZ A.1.3: FY1 stays the current fiscal year after its end until the
                   actual is announced; (fpedats-statpers)>=0 dropped those months. */
                AND CURCODE=''USD''
                AND fpi in (''1'',''2'')
                AND statpers >= ''{start_date}''
            """
        }
    }

# ====================================================================================================


def connect_wrds(username=None, password=None):
    """Log in to WRDS once. A failed login stops the script: repeated Duo failures lock the account."""
    if username is None and os.path.exists(os.path.expanduser('~/.pgpass')):
        for line in open(os.path.expanduser('~/.pgpass')):
            if line.startswith('wrds-pgdata.wharton.upenn.edu'):
                username = line.split(':')[3]   # libpq reads the password from ~/.pgpass itself
                break
    if username is None:
        username = input("Enter WRDS username: ")
        password = getpass.getpass("Enter WRDS password: ")

    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    # One connection only: DuckDB opens up to 64 parallel Postgres connections by default,
    # and each new connection is a new WRDS login (Duo push).
    con.execute("SET threads = 1;")
    con.execute("SET pg_connection_limit = 1;")
    print(f"Connecting to WRDS as {username} (approve the Duo push)...", flush=True)
    try:
        con.execute(f"ATTACH '{gen_wrds_connection_info(username, password)}' AS wrds (TYPE postgres, READ_ONLY)")
    except Exception as e:
        print(f"WRDS login failed - stopping, not retrying. Error: {e}")
        sys.exit(1)
    return con


def update_start(config, full):
    """First date to download: everything for a new/full table, otherwise the last
    `overlap_months` before the latest date already on disk (re-downloaded to pick up revisions)."""
    if full or config['date_col'] is None or not os.path.exists(config['output']):
        return None
    last = duckdb.sql(f"SELECT max({config['date_col']}) FROM read_parquet('{config['output']}')").fetchone()[0]
    return (last.replace(day=1) - relativedelta(months=config['overlap_months'])).isoformat()


def merge_update(config, new_file, start):
    """Replace rows dated >= start in the existing file with the newly downloaded rows."""
    out, col = config['output'], config['date_col']
    old_cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{out}')").fetchall()
    new_cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{new_file}')").fetchall()
    if {c[0] for c in old_cols} != {c[0] for c in new_cols}:
        raise RuntimeError(f"columns of {out} differ from the query; rerun with --full for this table")
    tmp = out + '.tmp'
    duckdb.sql(f"""
        COPY (
            SELECT * FROM read_parquet('{out}') WHERE {col} < DATE '{start}' OR {col} IS NULL
            UNION ALL BY NAME
            SELECT * FROM read_parquet('{new_file}')
        ) TO '{tmp}' (FORMAT PARQUET);
    """)
    os.replace(tmp, out)
    os.remove(new_file)


@measure_time
def download_all_tables(tables=None, full=(), username=None, password=None):
    """
    Bring the WRDS tables in OUTPUT_PATH up to date over a single WRDS login.

    A table without a local file (or listed in `full`) is downloaded from FULL_START. Otherwise only
    rows from `overlap_months` before its latest date are downloaded and replace the local rows from
    that date on. If any table fails the script stops instead of reconnecting; tables already
    finished are saved, so the next run only redoes the rest.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    names = tables or list(get_tables_config())
    starts = {name: update_start(get_tables_config()[name], name in full) for name in names}

    con = connect_wrds(username, password)
    try:
        for name in names:
            start = starts[name]
            config = get_tables_config(start or FULL_START)[name]
            print(f"{name}: " + (f"update from {start}" if start else "full download"), flush=True)
            if start is None:
                tmp = config['output'] + '.tmp'
                _execute_download(con, name, config['query'], tmp)
                os.replace(tmp, config['output'])
            else:
                new_file = config['output'] + '.new'
                _execute_download(con, name, config['query'], new_file)
                merge_update(config, new_file, start)
    finally:
        con.close()

    print("All tables downloaded successfully!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or update the WRDS tables in " + OUTPUT_PATH)
    parser.add_argument('--tables', nargs='+', choices=list(get_tables_config()),
                        help="only these tables (default: all)")
    parser.add_argument('--full', nargs='+', default=[], choices=list(get_tables_config()),
                        help="re-download these tables from " + FULL_START)
    args = parser.parse_args()
    download_all_tables(tables=args.tables, full=args.full)
