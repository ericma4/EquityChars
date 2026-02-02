import os
import duckdb
import polars as pl
import time


# Configuration
OUTPUT_PATH = "../data/raw/"


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


def gen_wrds_connection_info(user, password):
    """Generate WRDS PostgreSQL connection string for DuckDB."""
    return (
        f"host=wrds-pgdata.wharton.upenn.edu "
        f"port=9737 dbname=wrds "
        f"user={user} password={password} sslmode=require"
    )


def _execute_download(con, wrds_conninfo, table_name, query, output_file):
    """
    Helper function to execute a single table download.
    
    Args:
        con: DuckDB connection
        wrds_conninfo: WRDS connection string
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

def get_tables_config(start_date='2020-01-01'):
    """
    Define all tables to download with their queries.
    Edit the queries in this function to modify what data to download.
    
    Args:
        start_date: Start date for filtering (format: 'YYYY-MM-DD')
        
    Returns:
        Dictionary of table configurations with 'output' and 'query' keys
    """
    return {
        # 'comp_funda': {
        #     'output': os.path.join(OUTPUT_PATH, 'comp_funda.parquet'),
        #     'query': f"""
        #         SELECT 
        #             f.gvkey, f.cusip, f.datadate, f.fyear, c.cik, substr(c.sic,1,2) as sic2, c.sic, c.naics,
                    
        #             /* income statement */
        #             f.sale, f.revt, f.cogs, f.xsga, f.dp, f.xrd, f.xad, f.ib, f.ebitda,
        #             f.ebit, f.nopi, f.spi, f.pi, f.txp, f.ni, f.txfed, f.txfo, f.txt, f.xint,
                    
        #             /* CF statement and others */
        #             f.capx, f.oancf, f.dvt, f.ob, f.gdwlia, f.gdwlip, f.gwo, f.mib, f.oiadp, f.ivao,
                    
        #             /* assets */
        #             f.rect, f.act, f.che, f.ppegt, f.invt, f.at, f.aco, f.intan, f.ao, f.ppent, f.gdwl, f.fatb, f.fatl,
                    
        #             /* liabilities */
        #             f.lct, f.dlc, f.dltt, f.lt, f.dm, f.dcvt, f.cshrc, 
        #             f.dcpstk, f.pstk, f.ap, f.lco, f.lo, f.drc, f.drlt, f.txdi,
                    
        #             /* equity and other */
        #             f.ceq, f.scstkc, f.emp, f.csho, f.seq, f.txditc, f.pstkrv, f.pstkl, f.np, f.txdc, f.dpc, f.ajex, f.conm,
                    
        #             /* market */
        #             ABS(f.prcc_f) AS prcc_f
        #         FROM comp.funda AS f
        #         LEFT JOIN comp.company AS c 
        #             ON f.gvkey = c.gvkey
        #         WHERE f.indfmt = ''INDL'' 
        #         AND f.datafmt = ''STD''
        #         AND f.popsrc = ''D''
        #         AND f.consol = ''C''
        #         AND f.datadate >= ''{start_date}''
        #     """
        # },

        # 'comp_fundq': {
        #     'output': os.path.join(OUTPUT_PATH, 'comp_fundq.parquet'),
        #     'query': f"""
        #         SELECT 
        #             /*header info*/
        #             c.gvkey, f.cusip, f.datadate, f.fyearq,  substr(c.sic,1,2) as sic2, c.sic, f.fqtr, f.rdq,

        #             /*income statement*/
        #             f.ibq, f.saleq, f.txtq, f.revtq, f.cogsq, f.xsgaq, f.revty, f.cogsy, f.saley,

        #             /*balance sheet items*/
        #             f.atq, f.actq, f.cheq, f.lctq, f.dlcq, f.ppentq, f.ppegtq, f.txpq,

        #             /*others*/
        #             abs(f.prccq) as prccq, abs(f.prccq)*f.cshoq as mveq_f, f.ceqq, f.seqq, f.pstkq, f.ltq,
        #             f.pstkrq, f.gdwlq, f.intanq, f.mibq, f.oiadpq, f.ivaoq, f.conm,
                    
        #             /* v3 my formula add*/
        #             f.ajexq, f.cshoq, f.txditcq, f.npq, f.xrdy, f.xrdq, f.dpq, f.xintq, f.invtq, f.scstkcy, f.niq,
        #             f.oancfy, f.dlttq, f.rectq, f.acoq, f.apq, f.lcoq, f.loq, f.aoq

        #         FROM comp.fundq as f
        #         LEFT JOIN comp.company as c
        #         ON f.gvkey = c.gvkey

        #         /*get consolidated, standardized, industrial format statements*/
        #         WHERE f.indfmt = ''INDL'' 
        #         AND f.datafmt = ''STD''
        #         AND f.popsrc = ''D''
        #         AND f.consol = ''C''
        #         AND f.datadate >= ''{start_date}''
        #     """
        # },
        
        # 'crsp_msf': {
        #     'output': os.path.join(OUTPUT_PATH, 'crsp_msf.parquet'),
        #     'query': f"""
        #         SELECT 
        #             mthprc, mthret, mthretx, mthvol,
        #             shrout, mthcumfacpr, mthcumfacshr,
        #             permno, permco, mthcaldt, ticker, cusip, hdrcusip,
        #             issuernm, issuertype, securitytype, securitysubtype, sharetype, usincflg,
        #             primaryexch, conditionaltype, TradingStatusFlg
        #         FROM crspq.msf_v2
        #         WHERE mthcaldt >= ''{start_date}''
        #     """
        # },
        #@TODO: original accounting and abr does not have consistent filter
        # 'ccm': {
        #     'output': os.path.join(OUTPUT_PATH, 'ccm.parquet'),
        #     'query': """
        #         SELECT 
        #             gvkey, lpermno as permno, linktype, linkprim, 
        #             linkdt, linkenddt
        #         FROM crsp.ccmxpf_linktable
        #         WHERE substr(linktype,1,1)=''L''
        #         AND (linkprim =''C'' or linkprim=''P'')
        #     """
        # }

        # 'crsp_dsf': {
        #     'output': os.path.join(OUTPUT_PATH, 'crsp_dsf.parquet'),
        #     'query': f"""
        #         SELECT 
        #             a.permno, a.permco, a.dlycaldt, a.dlyret, a.dlyvol, a.dlyprc, a.dlyhigh, a.dlylow, 
        #             a.shrout, a.dlydelflg, a.dlycumfacpr, a.dlycumfacshr, 
        #             a.primaryexch, a.conditionaltype, a.tradingstatusflg,
        #             a.cusip, a.hdrcusip, a.siccd, 
        #             b.rf, b.mktrf, b.smb, b.hml, b.umd, b.rmw, b.cma
        #         FROM crspq.dsf_v2 as a
        #         LEFT JOIN ff_all.fivefactors_daily as b
        #         ON a.dlycaldt = b.date
        #         WHERE a.dlycaldt >= ''{start_date}''
        #     """
        # }

        'crsp_ind': {
            'output': os.path.join(OUTPUT_PATH, 'crsp_ind.parquet'),
            'query': f"""
                SELECT
                    dlycaldt, dlyprcret
                FROM crspq.inddlyseriesdata_ind
                WHERE indno = 1000502  /*industry code for S&P 500 Composite*/
                AND dlycaldt >= ''{start_date}''
            """
        }
    }

# ====================================================================================================


@measure_time
def download_all_tables(username, password, start_date='2023-01-01'):
    """
    Download all required tables from WRDS in a single session to avoid connection timeouts.
    
    Args:
        username: WRDS username
        password: WRDS password
        start_date: Filter for datadate >= start_date (default '2020-01-01')
        
    Output:
        Parquet files for comp_funda, crsp_msf, and ccm.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    wrds_conninfo = gen_wrds_connection_info(username, password)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{wrds_conninfo}' AS wrds (TYPE postgres, READ_ONLY)")
    
    # Get table configurations
    tables = get_tables_config(start_date)
    
    # Download all tables in sequence
    for table_name, config in tables.items():
        _execute_download(
            con=con,
            wrds_conninfo=wrds_conninfo,
            table_name=table_name,
            query=config['query'],
            output_file=config['output']
        )
    
    con.close()
    print("All tables downloaded successfully!", flush=True)


# Individual download functions for flexibility
def download_comp_funda(username, password, start_date='2020-01-01'):
    """Download Compustat annual fundamentals (funda) with company info."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, 'comp_funda.parquet')
    
    wrds_conninfo = gen_wrds_connection_info(username, password)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{wrds_conninfo}' AS wrds (TYPE postgres, READ_ONLY)")
    
    query = f"""
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
    
    _execute_download(con, wrds_conninfo, 'comp_funda', query, output_file)
    con.close()
    return output_file


def download_crsp_msf(username, password, start_date='2020-01-01'):
    """Download CRSP monthly stock file with security names."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, 'crsp_msf.parquet')
    
    wrds_conninfo = gen_wrds_connection_info(username, password)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{wrds_conninfo}' AS wrds (TYPE postgres, READ_ONLY)")
    
    query = f"""
        SELECT 
            a.prc, a.ret, a.retx, a.shrout, a.vol, a.cfacpr, a.cfacshr, a.date, a.permno, a.permco,
            b.ticker, b.ncusip, b.shrcd, b.exchcd, b.comnam
        FROM crsp.msf as a
        LEFT JOIN crsp.msenames as b
            ON a.permno=b.permno
            AND b.namedt<=a.date
            AND a.date<=b.nameendt
        WHERE a.date >= ''{start_date}''
        AND b.exchcd between 1 and 3
    """
    
    _execute_download(con, wrds_conninfo, 'crsp_msf', query, output_file)
    con.close()
    return output_file


def download_ccm(username, password):
    """Download CRSP-Compustat Merged (CCM) link table."""
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_file = os.path.join(OUTPUT_PATH, 'ccm.parquet')
    
    wrds_conninfo = gen_wrds_connection_info(username, password)
    con = duckdb.connect(":memory:")
    con.execute("INSTALL postgres; LOAD postgres;")
    con.execute(f"ATTACH '{wrds_conninfo}' AS wrds (TYPE postgres, READ_ONLY)")
    
    query = """
        SELECT 
            gvkey, lpermno as permno, linktype, linkprim, 
            linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE substr(linktype,1,1)=''L''
        AND (linkprim =''C'' or linkprim=''P'')
    """
    
    _execute_download(con, wrds_conninfo, 'ccm', query, output_file)
    con.close()
    return output_file


if __name__ == "__main__":
    # Prompt for WRDS credentials
    username = input("Enter WRDS username: ")
    password = input("Enter WRDS password: ")
    
    # Download all tables in one session (recommended to avoid connection timeouts)
    download_all_tables(username, password, start_date='2024-01-01')