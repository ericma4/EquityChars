import polars as pl
from functions import *
import datetime

# Configuration
INPUT_PATH = "../data/raw/"


#######################################################################################################################
#                                                  Compustat Block                                                    #
#######################################################################################################################
comp = pl.read_parquet(INPUT_PATH + 'comp_funda.parquet')

# convert datadate to date fmt and sort/clean up
comp = (comp
    .with_columns([
        pl.col('datadate').cast(pl.Date)
    ])
    .sort(['gvkey', 'datadate'])
    .unique()
)

# clean up csho and calculate market equity
comp = comp.with_columns([
    # Replace 0 with null in csho
    pl.when(pl.col('csho') == 0)
      .then(None)
      .otherwise(pl.col('csho'))
      .alias('csho'),
    
    # Calculate Compustat market equity
    (pl.col('csho') * pl.col('prcc_f')).alias('mve_f')
])

# do some clean up for dr
comp = comp.with_columns([
    pl.when(pl.col('drc').is_not_null() & pl.col('drlt').is_not_null())
      .then(pl.col('drc') + pl.col('drlt'))
      .when(pl.col('drc').is_not_null() & pl.col('drlt').is_null())
      .then(pl.col('drc'))
      .when(pl.col('drlt').is_not_null() & pl.col('drc').is_null())
      .then(pl.col('drlt'))
      .otherwise(None)
      .alias('dr')
])

# do some clean up for dc
comp = comp.with_columns([
    pl.when(pl.col('dcvt').is_null() & 
            pl.col('dcpstk').is_not_null() & 
            pl.col('pstk').is_not_null() & 
            (pl.col('dcpstk') > pl.col('pstk')))
      .then(pl.col('dcpstk') - pl.col('pstk'))
      .when(pl.col('dcvt').is_null() & 
            pl.col('dcpstk').is_not_null() & 
            pl.col('pstk').is_null())
      .then(pl.col('dcpstk'))
      .otherwise(None)
      .cast(pl.Float64)
      .alias('dc')
])

# Fill dc with dcvt if dc is null
comp = comp.with_columns([
    pl.when(pl.col('dc').is_null())
      .then(pl.col('dcvt'))
      .otherwise(pl.col('dc'))
      .alias('dc')
])

# Create xint0 and xsga0 (filled with 0 if null)
comp = comp.with_columns([
    pl.col('xint').fill_null(0).alias('xint0'),
    pl.col('xsga').fill_null(0).alias('xsga0')
])

# Replace 0 with null in ceq and at, then filter out null at
comp = (comp
    .with_columns([
        pl.when(pl.col('ceq') == 0).then(None).otherwise(pl.col('ceq')).alias('ceq'),
        pl.when(pl.col('at') == 0).then(None).otherwise(pl.col('at')).alias('at')
    ])
    .filter(pl.col('at').is_not_null())
)

#######################################################################################################################
#                                                       CRSP Block                                                    #
#######################################################################################################################
# Create a CRSP Subsample with Monthly Stock and Event Variables
# Restrictions will be applied later
# Select variables from the CRSP monthly stock and event datasets
crsp = pl.read_parquet(INPUT_PATH + 'crsp_msf.parquet')

# rename cusip as cusip_crsp
crsp = crsp.rename({'cusip': 'cusip_crsp'})

# filter exchcd & shrcd
# equivalent to legacy code exchcd = 1, 2 or 3
crsp = crsp.filter(
    (pl.col('primaryexch').is_in(['N', 'A', 'Q'])) &
    (pl.col('conditionaltype') == 'RW') &
    (pl.col('tradingstatusflg') == 'A')
)
# crsp['exchcd'] = crsp['primaryexch'].map({'N': 1, 'A': 2, 'Q': 3})
# # equivalent to legacy code shrcd = 10 or 11
# crsp = crsp.loc[(crsp.sharetype == 'NS') &
#                 (crsp.securitytype == 'EQTY') &
#                 (crsp.securitysubtype == 'COM') &
#                 (crsp.usincflg == 'Y') &
#                 (crsp.issuertype.isin(['ACOR', 'CORP']))]

# Mapping CIZ variables to SIZ varialbles
crsp = crsp.rename({
    'mthprc': 'prc',
    'mthret': 'ret',
    'mthretx': 'retx',  
    'mthvol': 'vol',
    'mthcumfacpr': 'cfacpr',
    'mthcumfacshr': 'cfacshr',
    'mthcaldt': 'date',
    'issuernm': 'comnam'
})

# change variable format to int
crsp = crsp.with_columns([
    pl.col('permco').cast(pl.Int64),
    pl.col('permno').cast(pl.Int64)
])

# Line up date to be end of month
# set all the date to the standard end date of month
crsp = crsp.with_columns([
    pl.col('date').dt.month_end().alias('monthend')
])

# Drop nulls and calculate market equity
crsp = (crsp
    .filter(pl.col('prc').is_not_null())
    .with_columns([
        (pl.col('prc').abs() * pl.col('shrout')).alias('me')
    ])
)

# if Market Equity is Nan then let return equals to 0
crsp = crsp.with_columns([
    pl.col('ret').fill_null(0),
    pl.col('retx').fill_null(0)
])

# impute me - sort and deduplicate
crsp = crsp.sort(['permno', 'date']).unique()

# Forward fill me within each permno group
crsp = crsp.with_columns([
    pl.col('me').forward_fill().over('permno').alias('me')
])

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''

# sum of me across different permno belonging to same permco a given date
crsp_summe = (crsp
    .group_by(['monthend', 'permco'])
    .agg(pl.col('me').sum())
)

# largest mktcap within a permco/date
crsp_maxme = (crsp
    .group_by(['monthend', 'permco'])
    .agg(pl.col('me').max())
)

# join by monthend/maxme to find the permno
crsp1 = crsp.join(crsp_maxme, on=['monthend', 'permco', 'me'], how='inner')

# join with sum of me to get the correct market cap info
# (no need to drop 'me' column first since we're joining and the sum will replace it)
crsp2 = (crsp1
    .drop('me')
    .join(crsp_summe, on=['monthend', 'permco'], how='inner')
    .sort(['permno', 'monthend'])
    .unique()
)

#######################################################################################################################
#                                                        CCM Block                                                    #
#######################################################################################################################
# merge CRSP and Compustat
# reference: https://wrds-www.wharton.upenn.edu/pages/support/applications/linking-databases/linking-crsp-and-compustat/
ccm = pl.read_parquet(INPUT_PATH + 'ccm.parquet')

# convert the permno to int64
ccm = ccm.with_columns([
    pl.col('permno').cast(pl.Int64)
])

# if linkenddt is missing then set to today date
ccm = ccm.with_columns([
    pl.col('linkenddt').fill_null(pl.lit(datetime.date.today()))
])

# merge ccm and comp
ccm1 = comp.join(ccm, on='gvkey', how='left')

# we can only get the accounting data after the firm public their report
# for annual data, we use 4, 5 or 6 months lagged data, now we follow Hou, Xue and Zhang (2015) use 4 months lag
ccm1 = ccm1.with_columns([
    # Year end: December 31st of the same year
    pl.date(pl.col('datadate').dt.year(), 12, 31).alias('yearend'),
    # jdate: 4 months after datadate, then month end
    pl.col('datadate').dt.offset_by('4mo').dt.month_end().alias('jdate')
])

# set link date bounds
ccm2 = ccm1.filter(
    (pl.col('jdate') >= pl.col('linkdt')) &
    (pl.col('jdate') <= pl.col('linkenddt'))
)

# link comp and crsp
crsp2 = crsp2.rename({'monthend': 'jdate'})
data_rawa = crsp2.join(ccm2, on=['permno', 'jdate'], how='inner')

# filter exchcd & shrcd and at least more than 1 year data
# Already filtered earlier in crsp

# process Market Equity
'''
Note: me is CRSP market equity, mve_f is Compustat market equity. Please choose the me below.
'''
data_rawa = data_rawa.with_columns([
    (pl.col('me') / 1000).alias('me')  # CRSP ME in millions
])
# data_rawa['me'] = data_rawa['mve_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
data_rawa = (data_rawa
    .with_columns([
        pl.when(pl.col('me') == 0)
          .then(None)
          .otherwise(pl.col('me'))
          .alias('me')
    ])
    .filter(pl.col('me').is_not_null())
)

# count single stock years
data_rawa = data_rawa.with_columns([
    (pl.col('gvkey').cum_count().over('gvkey')).alias('count')
])

# # deal with the duplicates
# @todo: check if there are any duplicates with full data
# data_rawa.filter(data_rawa.is_duplicated))

# # Keep first occurrence within each group
# data_rawa = (data_rawa
#     .with_columns([
#         pl.lit(1).alias('temp').over(['datadate', 'permno', 'linkprim']).cum_count()
#     ])
#     .filter(pl.col('temp') == 1)
#     .drop('temp')
# )

# # Keep last occurrence within each group
# data_rawa = (data_rawa
#     .sort(['permno', 'yearend', 'datadate'])
#     .with_columns([
#         pl.lit(1).alias('temp').over(['permno', 'yearend', 'datadate']).cum_count()
#     ])
#     .with_columns([
#         pl.col('temp').max().over(['permno', 'yearend', 'datadate']).alias('max_temp')
#     ])
#     .filter(pl.col('temp') == pl.col('max_temp'))
#     .drop(['temp', 'max_temp'])
# )

# Sort
data_rawa = data_rawa.sort(['permno', 'jdate'])

# fama-french 49 industry
data_rawa = data_rawa.with_columns([
    pl.col('sic').cast(pl.Int64)
])

# Apply ffi49 function (assuming it returns a Series/column)
data_rawa = data_rawa.with_columns([
    pl.Series('ffi49', ffi49(data_rawa))
])

data_rawa = data_rawa.with_columns([
    pl.col('ffi49').fill_nan(49).cast(pl.Int64)
])
#######################################################################################################################
#                                                  Annual Variables                                                   #
#######################################################################################################################
# preferrerd stock
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('pstkrv').is_null())
      .then(pl.col('pstkl'))
      .otherwise(pl.col('pstkrv'))
      .alias('ps')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('ps').is_null())
      .then(pl.col('pstk'))
      .otherwise(pl.col('ps'))
      .alias('ps')
])

data_rawa = data_rawa.with_columns([
    pl.col('ps').fill_null(0),
    pl.col('txditc').fill_null(0)
])

# book equity
data_rawa = data_rawa.with_columns([
    (pl.col('seq') + pl.col('txditc') - pl.col('ps')).alias('be')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('be') > 0)
      .then(pl.col('be'))
      .otherwise(None)
      .alias('be')
])

# acc - lagged variables
data_rawa = data_rawa.with_columns([
    pl.col('act').shift(1).over('permno').alias('act_l1'),
    pl.col('lct').shift(1).over('permno').alias('lct_l1'),
    pl.col('at').shift(1).over('permno').alias('at_l1')
])

# #################### Add np lag (also fixed row 272 below) on 2025.02.23 ####################
# data_rawa['np_l1'] = data_rawa.groupby(['permno'])['np'].shift(1)

# condlist = [data_rawa['np'].isnull(),
#             data_rawa['act'].isnull() | data_rawa['lct'].isnull()]
# choicelist = [((data_rawa['act'] - data_rawa['lct']) - (data_rawa['act_l1'] - data_rawa['lct_l1']) / (data_rawa['be'])),
#               (data_rawa['ib'] - data_rawa['oancf']) / (data_rawa['be'])] ##### Delete "10*" on 2025.02.26 #####
# data_rawa['acc'] = np.select(condlist,
#                              choicelist,
#                              default=((data_rawa['act'] - data_rawa['lct'] + data_rawa['np']) -
#                                       (data_rawa['act_l1'] - data_rawa['lct_l1'] + data_rawa['np_l1'])) / (data_rawa['be']))

#################### Add Sloan(1996) or HXZ and GHZ operating accruals on 2025.02.28 ####################
# More lagged variables
data_rawa = data_rawa.with_columns([
    pl.col('che').shift(1).over('permno').alias('che_l1'),
    pl.col('dlc').shift(1).over('permno').alias('dlc_l1'),
    pl.col('txp').shift(1).over('permno').alias('txp_l1')
])

# acc calculation
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('oancf').is_null())
      .then(
        (((pl.col('act') - pl.col('act_l1')) - (pl.col('che') - pl.col('che_l1')) -
          (pl.col('lct') - pl.col('lct_l1')) + (pl.col('dlc') - pl.col('dlc_l1')) +
          (pl.col('txp') - pl.col('txp_l1')).fill_null(0) - pl.col('dp')) / 
         ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
      )
      .otherwise(
        (pl.col('ib') - pl.col('oancf')) / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)
      )
      .alias('acc')
])

# absacc
data_rawa = data_rawa.with_columns([
    pl.col('acc').abs().alias('absacc')
])

# agr
data_rawa = data_rawa.with_columns([
    ((pl.col('at') - pl.col('at_l1')) / pl.col('at_l1').replace(0, None)).alias('agr')
])

# bm
# data_rawa['bm'] = data_rawa['be'] / data_rawa['me']

# cfp
# condlist = [data_rawa['dp'].isnull(),
#             data_rawa['ib'].isnull()]
# choicelist = [data_rawa['ib']/data_rawa['me'],
#               np.nan]
# data_rawa['cfp'] = np.select(condlist, choicelist, default=(data_rawa['ib']+data_rawa['dp'])/data_rawa['me'])

# ep
# data_rawa['ep'] = data_rawa['ib']/data_rawa['me']

# ni
data_rawa = data_rawa.with_columns([
    pl.col('csho').shift(1).over('permno').alias('csho_l1'),
    pl.col('ajex').shift(1).over('permno').alias('ajex_l1')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('gvkey') != pl.col('gvkey').shift(1))
      .then(None)
      .otherwise(
        (pl.col('csho') * pl.col('ajex')).log()
          .fill_nan(0)
          .fill_null(0) -
        (pl.col('csho_l1') * pl.col('ajex_l1')).log()
          .fill_nan(0)
          .fill_null(0)
      )
      .alias('ni')
])

# op
data_rawa = data_rawa.with_columns([
    pl.col('cogs').fill_null(0).alias('cogs0'),
    pl.col('xint').fill_null(0).alias('xint0'),
    pl.col('xsga').fill_null(0).alias('xsga0')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('revt').is_null())
      .then(None)
      .when(pl.col('be').is_null())
      .then(None)
      .otherwise(
        (pl.col('revt') - pl.col('cogs0') - pl.col('xsga0') - pl.col('xint0')) / pl.col('be').replace(0, None)
      )
      .alias('op')
])

# rsup
data_rawa = data_rawa.with_columns([
    pl.col('sale').shift(1).over('permno').alias('sale_l1')
])
# data_rawa['rsup'] = (data_rawa['sale']-data_rawa['sale_l1'])/data_rawa['me']

# cash
data_rawa = data_rawa.with_columns([
    (pl.col('che') / pl.col('at').replace(0, None)).alias('cash')
])

# lev
# data_rawa['lev'] = data_rawa['lt']/data_rawa['me']

# sp
# data_rawa['sp'] = data_rawa['sale']/data_rawa['me']

# rd_sale
data_rawa = data_rawa.with_columns([
    (pl.col('xrd') / pl.col('sale').replace(0, None)).alias('rd_sale')
])

# rdm
# data_rawa['rdm'] = data_rawa['xrd']/data_rawa['me']

# adm hxz adm
# data_rawa['adm'] = data_rawa['xad']/data_rawa['me']

# gma
data_rawa = data_rawa.with_columns([
    ((pl.col('revt') - pl.col('cogs')) / pl.col('at_l1').replace(0, None)).alias('gma')
])

# chcsho
data_rawa = data_rawa.with_columns([
    ((pl.col('csho') / pl.col('csho_l1').replace(0, None)) - 1).alias('chcsho')
])

# lgr
data_rawa = data_rawa.with_columns([
    pl.col('lt').shift(1).over('permno').alias('lt_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('lt') / pl.col('lt_l1').replace(0, None)) - 1).alias('lgr')
])

#################### Follow Hafzalla, Lundholm, and Van Winkle (2011) and GHZ on 2025.02.28 ####################
# pctacc
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('ib') == 0)
      .then((pl.col('ib') - pl.col('oancf')) / 0.01)
      .when(pl.col('oancf').is_null())
      .then(
        (((pl.col('act') - pl.col('act_l1')) - (pl.col('che') - pl.col('che_l1'))) -
         ((pl.col('lct') - pl.col('lct_l1')) - pl.col('dlc') - pl.col('dlc_l1') -
          ((pl.col('txp') - pl.col('txp_l1')).fill_null(0) - pl.col('dp')))) / pl.col('ib').abs()
      )
      .when(pl.col('oancf').is_null() & (pl.col('ib') == 0))
      .then(
        (((pl.col('act') - pl.col('act_l1')) - (pl.col('che') - pl.col('che_l1'))) -
         ((pl.col('lct') - pl.col('lct_l1')) - pl.col('dlc') - pl.col('dlc_l1') -
          ((pl.col('txp') - pl.col('txp_l1')).fill_null(0) - pl.col('dp')))) / 0.01
      )
      .otherwise(
        (pl.col('ib') - pl.col('oancf')) / pl.col('ib').replace(0, None).abs()
      )
      .alias('pctacc')
])

# sgr
data_rawa = data_rawa.with_columns([
    ((pl.col('sale') / pl.col('sale_l1').replace(0, None)) - 1).alias('sgr')
])

# chato
data_rawa = data_rawa.with_columns([
    pl.col('at').shift(2).over('permno').alias('at_l2')
])
data_rawa = data_rawa.with_columns([
    ((pl.col('sale') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)) -
     (pl.col('sale_l1') / ((pl.col('at') + pl.col('at_l2')) / 2).replace(0, None))).alias('chato')
])

# chtx
data_rawa = data_rawa.with_columns([
    pl.col('txt').shift(1).over('permno').alias('txt_l1')
])
data_rawa = data_rawa.with_columns([
    ((pl.col('txt') - pl.col('txt_l1')) / pl.col('at_l1').replace(0, None)).alias('chtx')
])

# noa
data_rawa = data_rawa.with_columns([
    (((pl.col('at') - pl.col('che') - pl.col('ivao').fill_null(0)) -
      (pl.col('at') - pl.col('dlc').fill_null(0) - pl.col('dltt').fill_null(0) - 
       pl.col('mib').fill_null(0) - pl.col('pstk').fill_null(0) - pl.col('ceq'))) / 
     pl.col('at_l1').replace(0, None)).alias('noa')
])

# rna
data_rawa = data_rawa.with_columns([
    pl.col('noa').shift(1).over('permno').alias('noa_l1')
])
data_rawa = data_rawa.with_columns([
    (pl.col('oiadp') / pl.col('noa_l1').replace(0, None)).alias('rna')
])

# pm
data_rawa = data_rawa.with_columns([
    (pl.col('oiadp') / pl.col('sale').replace(0, None)).alias('pm')
])

# ato
data_rawa = data_rawa.with_columns([
    (pl.col('sale') / pl.col('noa_l1').replace(0, None)).alias('ato')
])

# depr
data_rawa = data_rawa.with_columns([
    (pl.col('dp') / pl.col('ppent').replace(0, None)).alias('depr')
])

# invest
data_rawa = data_rawa.with_columns([
    pl.col('ppent').shift(1).over('permno').alias('ppent_l1'),
    pl.col('invt').shift(1).over('permno').alias('invt_l1')
])
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('ppegt').is_null())
      .then(
        ((pl.col('ppent') - pl.col('ppent_l1')) + 
         (pl.col('invt') - pl.col('invt_l1'))) / pl.col('at_l1').replace(0, None)
      )
      .otherwise(
        ((pl.col('ppegt') - pl.col('ppent_l1')) + 
         (pl.col('invt') - pl.col('invt_l1'))) / pl.col('at_l1').replace(0, None)
      )
      .alias('invest')
])

# egr
data_rawa = data_rawa.with_columns([
    pl.col('ceq').shift(1).over('permno').alias('ceq_l1')
])
data_rawa = data_rawa.with_columns([
    ((pl.col('ceq') - pl.col('ceq_l1')) / pl.col('ceq_l1').replace(0, None)).alias('egr')
])

# cashdebt
data_rawa = data_rawa.with_columns([
    ((pl.col('ib') + pl.col('dp')) / 
     ((pl.col('lt') + pl.col('lt_l1')) / 2).replace(0, None)).alias('cashdebt')
])

# rd
data_rawa = data_rawa.with_columns([
    (pl.col('xrd') / pl.col('at_l1').replace(0, None)).alias('xrd/at_l1')
])
data_rawa = data_rawa.with_columns([
    pl.col('xrd/at_l1').shift(1).over('permno').alias('xrd/at_l1_l1')
])
data_rawa = data_rawa.with_columns([
    pl.when(
        (((pl.col('xrd') / pl.col('at').replace(0, None)) - pl.col('xrd/at_l1_l1')) / 
         pl.col('xrd/at_l1_l1').replace(0, None)) > 0.05
    )
      .then(1)
      .otherwise(0)
      .alias('rd')
])

# roa
data_rawa = data_rawa.with_columns([
    (pl.col('ib') / pl.col('at_l1').replace(0, None)).alias('roa')
])

# roe
data_rawa = data_rawa.with_columns([
    (pl.col('ib') / pl.col('ceq_l1').replace(0, None)).alias('roe')
])

# dy
# data_rawa['dy'] = data_rawa['dvt']/data_rawa['me']

################## Added on 2020.07.28 ##################

# roic
data_rawa = data_rawa.with_columns([
    ((pl.col('ebit') - pl.col('nopi')) /
     (pl.col('ceq') + pl.col('lt') - pl.col('che')).replace(0, None)
    ).alias('roic')
])

# chinv
data_rawa = data_rawa.with_columns([
    ((pl.col('invt') - pl.col('invt_l1')) /
     ((pl.col('at') + pl.col('at_l2')) / 2).replace(0, None)
    ).alias('chinv')
])

# pchsale_pchinvt
data_rawa = data_rawa.with_columns([
    (((pl.col('sale') - pl.col('sale_l1')) / pl.col('sale_l1').replace(0, None)) -
     ((pl.col('invt') - pl.col('invt_l1')) / pl.col('invt_l1').replace(0, None))
    ).alias('pchsale_pchinvt')
])

# pchsale_pchrect
data_rawa = data_rawa.with_columns([
    pl.col('rect').shift(1).over('permno').alias('rect_l1')
])

data_rawa = data_rawa.with_columns([
    (((pl.col('sale') - pl.col('sale_l1')) / pl.col('sale_l1').replace(0, None)) -
     ((pl.col('rect') - pl.col('rect_l1')) / pl.col('rect_l1').replace(0, None))
    ).alias('pchsale_pchrect')
])

# pchgm_pchsale
data_rawa = data_rawa.with_columns([
    pl.col('cogs').shift(1).over('permno').alias('cogs_l1')
])

data_rawa = data_rawa.with_columns([
    ((((pl.col('sale') - pl.col('cogs')) - (pl.col('sale_l1') - pl.col('cogs_l1'))) /
      (pl.col('sale_l1') - pl.col('cogs_l1')).replace(0, None)) -
     ((pl.col('sale') - pl.col('sale_l1')) / pl.col('sale').replace(0, None))
    ).alias('pchgm_pchsale')
])

# pchsale_pchxsga
data_rawa = data_rawa.with_columns([
    pl.col('xsga').shift(1).over('permno').alias('xsga_l1')
])

data_rawa = data_rawa.with_columns([
    (((pl.col('sale') - pl.col('sale_l1')) / pl.col('sale_l1').replace(0, None)) -
     ((pl.col('xsga') - pl.col('xsga_l1')) / pl.col('xsga_l1').replace(0, None))
    ).alias('pchsale_pchxsga')
])

# pchdepr
data_rawa = data_rawa.with_columns([
    pl.col('dp').shift(1).over('permno').alias('dp_l1')
])

data_rawa = data_rawa.with_columns([
    (((pl.col('dp') / pl.col('ppent').replace(0, None)) - 
      (pl.col('dp_l1') / pl.col('ppent_l1').replace(0, None))) /
     (pl.col('dp_l1') / pl.col('ppent').replace(0, None)).replace(0, None))
    .alias('pchdepr')
])

# chadv
data_rawa = data_rawa.with_columns([
    pl.col('xad').shift(1).over('permno').alias('xad_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('xad') + 1).log() - (pl.col('xad_l1') + 1).log()).alias('chadv')
])

# pchcapx
data_rawa = data_rawa.with_columns([
    pl.col('capx').shift(1).over('permno').alias('capx_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('capx') - pl.col('capx_l1')) / pl.col('capx_l1').replace(0, None))
    .alias('pchcapx')
])

# grcapx
data_rawa = data_rawa.with_columns([
    pl.col('capx').shift(2).over('permno').alias('capx_l2')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('capx') - pl.col('capx_l2')) / pl.col('capx_l2').replace(0, None))
    .alias('grcapx')
])

# grGW
data_rawa = data_rawa.with_columns([
    pl.col('gdwl').shift(1).over('permno').alias('gdwl_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('gdwl') - pl.col('gdwl_l1')) / pl.col('gdwl').replace(0, None))
    .alias('grGW')
])

data_rawa = data_rawa.with_columns([
    pl.when((pl.col('gdwl') == 0) | pl.col('gdwl').is_null())
      .then(0)
      .when(pl.col('gdwl').is_not_null() & (pl.col('gdwl') != 0) & pl.col('grGW').is_null())
      .then(1)
      .otherwise(pl.col('grGW'))
      .alias('grGW')
])

# currat
data_rawa = data_rawa.with_columns([
    (pl.col('act') / pl.col('lct').replace(0, None)).alias('currat')
])

# pchcurrat
data_rawa = data_rawa.with_columns([
    (((pl.col('act') / pl.col('lct').replace(0, None)) -
      (pl.col('act_l1') / pl.col('lct_l1').replace(0, None))) /
     (pl.col('act_l1') / pl.col('lct_l1').replace(0, None)).replace(0, None))
    .alias('pchcurrat')
])

# quick
data_rawa = data_rawa.with_columns([
    ((pl.col('act') - pl.col('invt')) / pl.col('lct').replace(0, None)).alias('quick')
])

# pchquick
data_rawa = data_rawa.with_columns([
    ((((pl.col('act') - pl.col('invt')) / pl.col('lct').replace(0, None)) -
      ((pl.col('act_l1') - pl.col('invt_l1')) / pl.col('lct_l1').replace(0, None))) /
     ((pl.col('act_l1') - pl.col('invt_l1')) / pl.col('lct_l1').replace(0, None)).replace(0, None))
    .alias('pchquick')
])

# salecash
data_rawa = data_rawa.with_columns([
    (pl.col('sale') / pl.col('che').replace(0, None)).alias('salecash')
])

# salerec
data_rawa = data_rawa.with_columns([
    (pl.col('sale') / pl.col('rect').replace(0, None)).alias('salerec')
])

# saleinv
data_rawa = data_rawa.with_columns([
    (pl.col('sale') / pl.col('invt').replace(0, None)).alias('saleinv')
])

# pchsaleinv
data_rawa = data_rawa.with_columns([
    (((pl.col('sale') / pl.col('invt').replace(0, None)) - (pl.col('sale_l1') / pl.col('invt_l1').replace(0, None))) /
     (pl.col('sale_l1') / pl.col('invt_l1').replace(0, None)).replace(0, None)).alias('pchsaleinv')
])

# realestate
data_rawa = data_rawa.with_columns([
    ((pl.col('fatb') + pl.col('fatl')) / pl.col('ppegt').replace(0, None)).alias('realestate')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('ppegt').is_null())
      .then((pl.col('fatb') + pl.col('fatl')) / pl.col('ppent').replace(0, None))
      .otherwise(pl.col('realestate'))
      .alias('realestate')
])

# obklg
data_rawa = data_rawa.with_columns([
    (pl.col('ob') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)).alias('obklg')
])

# chobklg
data_rawa = data_rawa.with_columns([
    pl.col('ob').shift(1).over('permno').alias('ob_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('ob') - pl.col('ob_l1')) /
     ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)).alias('chobklg')
])

# grltnoa
data_rawa = data_rawa.with_columns([
    pl.col('aco').shift(1).over('permno').alias('aco_l1'),
    pl.col('intan').shift(1).over('permno').alias('intan_l1'),
    pl.col('ao').shift(1).over('permno').alias('ao_l1'),
    pl.col('ap').shift(1).over('permno').alias('ap_l1'),
    pl.col('lco').shift(1).over('permno').alias('lco_l1'),
    pl.col('lo').shift(1).over('permno').alias('lo_l1'),
    pl.col('rect').shift(1).over('permno').alias('rect_l1')
])

data_rawa = data_rawa.with_columns([
    (((pl.col('rect') + pl.col('invt') + pl.col('ppent') + pl.col('aco') + pl.col('intan') +
       pl.col('ao') - pl.col('ap') - pl.col('lco') - pl.col('lo')) -
      (pl.col('rect_l1') + pl.col('invt_l1') + pl.col('ppent_l1') + pl.col('aco_l1') +
       pl.col('intan_l1') + pl.col('ao_l1') - pl.col('ap_l1') - pl.col('lco_l1') - pl.col('lo_l1')) -
      (pl.col('rect') - pl.col('rect_l1') + pl.col('invt') - pl.col('invt_l1') +
       pl.col('aco') - pl.col('aco_l1') -
       (pl.col('ap') - pl.col('ap_l1') + pl.col('lco') - pl.col('lco_l1')) - pl.col('dp'))) /
     ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
     .alias('grltnoa')
])

# conv
data_rawa = data_rawa.with_columns([
    (pl.col('dc') / pl.col('dltt').replace(0, None)).alias('conv')
])

# convind
data_rawa = data_rawa.with_columns([
    pl.when(
        ((pl.col('dc').is_not_null()) & (pl.col('dc') != 0)) |
        ((pl.col('cshrc').is_not_null()) & (pl.col('cshrc') != 0))
    )
      .then(1)
      .otherwise(0)
      .alias('convind')
])

# chdrc
data_rawa = data_rawa.with_columns([
    pl.col('dr').shift(1).over('permno').alias('dr_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('dr') - pl.col('dr_l1')) /
     ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
    .alias('chdrc')
])

# rdbias
data_rawa = data_rawa.with_columns([
    pl.col('xrd').shift(1).over('permno').alias('xrd_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('xrd') / pl.col('xrd_l1').replace(0, None)) - 1 - 
     (pl.col('ib') / pl.col('ceq_l1').replace(0, None)))
    .alias('rdbias')
])

# operprof
data_rawa = data_rawa.with_columns([
    ((pl.col('revt') - pl.col('cogs') - pl.col('xsga0') - pl.col('xint0')) /
     pl.col('ceq_l1').replace(0, None))
    .alias('operprof')
])

# cfroa
data_rawa = data_rawa.with_columns([
    (pl.col('oancf') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
    .alias('cfroa')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('oancf').is_null())
      .then(
        (pl.col('ib') + pl.col('dp')) / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None)
      )
      .otherwise(pl.col('cfroa'))
      .alias('cfroa')
])

# xrdint
data_rawa = data_rawa.with_columns([
    (pl.col('xrd') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
    .alias('xrdint')
])

# capxint
data_rawa = data_rawa.with_columns([
    (pl.col('capx') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
    .alias('capxint')
])

# xadint
data_rawa = data_rawa.with_columns([
    (pl.col('xad') / ((pl.col('at') + pl.col('at_l1')) / 2).replace(0, None))
    .alias('xadint')
])

# chpm
data_rawa = data_rawa.with_columns([
    pl.col('ib').shift(1).over('permno').alias('ib_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('ib') / pl.col('sale').replace(0, None)) - 
     (pl.col('ib_l1') / pl.col('sale_l1').replace(0, None))).alias('chpm')
])

# ala
data_rawa = data_rawa.with_columns([
    pl.col('gdwl').fill_null(0),
    pl.col('intan').fill_null(0)
])

data_rawa = data_rawa.with_columns([
    (pl.col('che') + 0.75 * (pl.col('act') - pl.col('che')) -
     0.5 * (pl.col('at') - pl.col('act') - pl.col('gdwl') - pl.col('intan'))).alias('ala')
])

# alm
data_rawa = data_rawa.with_columns([
    (pl.col('ala') /
     (pl.col('at') + pl.col('prcc_f') * pl.col('csho') - pl.col('ceq')).replace(0, None))
    .alias('alm')
])

# hire
data_rawa = data_rawa.with_columns([
    pl.col('emp').shift(1).over('permno').alias('emp_l1')
])

data_rawa = data_rawa.with_columns([
    ((pl.col('emp') - pl.col('emp_l1')) / pl.col('emp_l1').replace(0, None)).alias('hire')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('emp').is_null() | pl.col('emp_l1').is_null())
      .then(0)
      .otherwise(pl.col('hire'))
      .alias('hire')
])

# herf
df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('sale').sum().alias('indsale'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    ((pl.col('sale') / pl.col('indsale').replace(0, None)) * 
     (pl.col('sale') / pl.col('indsale').replace(0, None)))
    .alias('herf')
])

df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('herf').sum())
)

data_rawa = data_rawa.drop('herf')
data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')
################## Added on 2022.09.06 ##################
# age
data_rawa = data_rawa.with_columns([
    pl.col('count').alias('age')
])

# cashpr
# data_rawa['cashpr'] = ((data_rawa['me'] + data_rawa['dltt'] - data_rawa['at']) / data_rawa['che'])

# chempia
df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('hire').mean().alias('hire_ind'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    (pl.col('hire') - pl.col('hire_ind')).alias('chempia')
])

# chpmia
df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('chpm').mean().alias('chpm_ind'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    (pl.col('chpm') - pl.col('chpm_ind')).alias('chpmia')
])

# chatoia
df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('chato').mean().alias('chato_ind'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    (pl.col('chato') - pl.col('chato_ind')).alias('chatoia')
])

# divi
data_rawa = data_rawa.with_columns([
    pl.col('dvt').shift(1).over('permno').alias('dvt_l1')
])

data_rawa = data_rawa.with_columns([
    pl.when(
        (pl.col('dvt').is_not_null()) & (pl.col('dvt') > 0) &
        ((pl.col('dvt_l1') == 0) | pl.col('dvt_l1').is_null())
    )
      .then(1)
      .otherwise(0)
      .alias('divi')
])

# divo
data_rawa = data_rawa.with_columns([
    pl.when(
        (pl.col('dvt').is_null()) | (pl.col('dvt') == 0) &
        ((pl.col('dvt_l1') > 0) | pl.col('dvt_l1').is_not_null())
    )
      .then(1)
      .otherwise(0)
      .alias('divo')
])

# Mohanram (2005) score (Annual Related)
df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('roa').median().alias('md_roa'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('cfroa').median().alias('md_cfroa'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('oancf').median().alias('md_oancf'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('xrdint').median().alias('md_xrdint'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('capxint').median().alias('md_capxint'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

df_temp = (data_rawa
    .group_by(['fyear', 'ffi49'])
    .agg(pl.col('xadint').median().alias('md_xadint'))
)
data_rawa = data_rawa.join(df_temp, on=['fyear', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('roa') > pl.col('md_roa')).then(1).otherwise(0).alias('m1'),
    pl.when(pl.col('cfroa') > pl.col('md_cfroa')).then(1).otherwise(0).alias('m2'),
    pl.when(pl.col('oancf') > pl.col('md_oancf')).then(1).otherwise(0).alias('m3'),
    pl.when(pl.col('xrdint') > pl.col('md_xrdint')).then(1).otherwise(0).alias('m4'),
    pl.when(pl.col('capxint') > pl.col('md_capxint')).then(1).otherwise(0).alias('m5'),
    pl.when(pl.col('xadint') > pl.col('md_xadint')).then(1).otherwise(0).alias('m6')
])

# pchcapx_ia
df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('pchcapx').mean().alias('pchcapx_ind'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    (pl.col('pchcapx') - pl.col('pchcapx_ind')).alias('pchcapx_ia')
])

# secured
data_rawa = data_rawa.with_columns([
    (pl.col('dm') / pl.col('dltt').replace(0, None)).alias('secured')
])

# securedind
data_rawa = data_rawa.with_columns([
    pl.when((pl.col('dm').is_not_null()) & (pl.col('dm') != 0))
      .then(1)
      .otherwise(0)
      .alias('securedind')
])

# sin
data_rawa = data_rawa.with_columns([
    pl.when(
        ((pl.col('sic') >= 2100) & (pl.col('sic') <= 2199)) |
        ((pl.col('sic') >= 2080) & (pl.col('sic') <= 2085)) |
        (pl.col('naics') == '7132') |
        (pl.col('naics') == '71312') |
        (pl.col('naics') == '713210') |
        (pl.col('naics') == '71329') |
        (pl.col('naics') == '713290') |
        (pl.col('naics') == '72112') |
        (pl.col('naics') == '721120')
    )
      .then(1)
      .otherwise(0)
      .alias('sin')
])

# tang
data_rawa = data_rawa.with_columns([
    ((pl.col('che') + pl.col('rect') * 0.715 +
      pl.col('invt') * 0.547 + pl.col('ppent') * 0.535) / pl.col('at').replace(0, None))
    .alias('tang')
])

# tb, Lev and Nissim (2004)
data_rawa = data_rawa.with_columns([
    pl.when(pl.col('fyear') <= 1978)
      .then(0.48)
      .when((pl.col('fyear') >= 1979) & (pl.col('fyear') <= 1986))
      .then(0.46)
      .when(pl.col('fyear') == 1987)
      .then(0.4)
      .when((pl.col('fyear') >= 1988) & (pl.col('fyear') <= 1992))
      .then(0.34)
      .when(pl.col('fyear') >= 1993)
      .then(0.35)
      .otherwise(None)
      .alias('tr')
])

data_rawa = data_rawa.with_columns([
    (((pl.col('txfo') + pl.col('txfed').replace(0, None)) / 
      pl.col('tr').replace(0, None)) / 
      pl.col('ib').replace(0, None))
    .alias('tb_1')
])

data_rawa = data_rawa.with_columns([
    pl.when(pl.col('txfo').is_null() | pl.col('txfed').is_null())
      .then(
        ((pl.col('txt') - pl.col('txdi')) / pl.col('tr').replace(0, None)) / 
        pl.col('ib').replace(0, None)
      )
      .otherwise(pl.col('tb_1'))
      .alias('tb_1')
])

data_rawa = data_rawa.with_columns([
    pl.when(
        (((pl.col('txfo') + pl.col('txfed') > 0) | (pl.col('txt') > pl.col('txdi'))) &
         (pl.col('ib') <= 0))
    )
      .then(1)
      .otherwise(pl.col('tb_1'))
      .alias('tb_1')
])

df_temp = (data_rawa
    .group_by(['datadate', 'ffi49'])
    .agg(pl.col('tb_1').mean().alias('tb_1_ind'))
)

data_rawa = data_rawa.join(df_temp, on=['datadate', 'ffi49'], how='left')

data_rawa = data_rawa.with_columns([
    (pl.col('tb_1') - pl.col('tb_1_ind')).alias('tb')
])

print("Finish Annual Variables Calculation! \n")

#######################################################################################################################
#                                              Compustat Quarterly Raw Info                                           #
#######################################################################################################################
comp = pl.read_parquet(INPUT_PATH + 'comp_fundq.parquet')

# rename cusip as cusip_comp
comp.rename(columns={'cusip': 'cusip_comp'}, inplace=True)

# comp['cusip6'] = comp['cusip'].str.strip().str[0:6]
comp = comp.dropna(subset=['ibq']).reset_index(drop=True)

# sort and clean up
comp = comp.sort_values(by=['gvkey', 'datadate']
                        ).drop_duplicates().reset_index(drop=True)
comp['cshoq'] = np.where(comp['cshoq'] == 0, np.nan, comp['cshoq'])
comp['ceqq'] = np.where(comp['ceqq'] == 0, np.nan, comp['ceqq'])
comp['atq'] = np.where(comp['atq'] == 0, np.nan, comp['atq'])
comp = comp.dropna(subset=['atq']).reset_index(drop=True)

# convert datadate to date fmt
comp['datadate'] = pd.to_datetime(comp['datadate'])

# merge ccm and comp
# Lag rule: Following Hou, Xue and Zhang (2015), We use earnings immediately after the announcement day
# For those data with missing announcement date record, we straightly let the data available after 4 month
ccm1 = pd.merge(comp, ccm, how='left', on=['gvkey']).reset_index(drop=True)
ccm1['yearend'] = ccm1['datadate'] + YearEnd(0)
ccm1['jdate'] = ccm1['datadate'] + MonthEnd(4)  # we change quarterly lag here

# deal with ibq to make it as up-to-date as possible
ccm1['rdq'] = pd.to_datetime(ccm1['rdq']) + MonthEnd(0)
ccm1['rdq'] = np.where(ccm1['rdq'].isnull(), ccm1['jdate'], ccm1['rdq'])
# compare next quarter's announcement date with jdate
ccm1['rdq_temp'] = ccm1.groupby(['permno'])['rdq'].shift(-1)
ccm1['rdq_temp'] = np.where(ccm1['rdq_temp'].isnull(
), ccm1['jdate'], ccm1['rdq_temp'])  # if rdq is NaN, let it be jdate
# compare next quarter's announcement date with jdate
ccm1['ibq_diff'] = ccm1['jdate'] - ccm1['rdq_temp']
ccm1['ibq_diff'] = ccm1['ibq_diff'].dt.days
ccm1['ibq_new'] = ccm1.groupby(
    ['permno'])['ibq'].shift(-1)  # next quarter's ibq
ccm1 = ccm1.rename(columns={'ibq': 'ibq_old'})  # original ibq
'''
if the announcement date is same or in front of jdate, we can use the up-to-date ibq.
otherwise, we consider the up-to-date ibq is not available and still use the lag-4-months ibq
'''
ccm1['ibq'] = np.where(ccm1['ibq_diff'] >= 0, ccm1['ibq_new'], ccm1['ibq_old'])
# for most recent record we can only use the lag-4-months ibq
ccm1['ibq'] = np.where(ccm1['ibq'].isnull(), ccm1['ibq_old'], ccm1['ibq'])

# set link date bounds
ccm2 = ccm1[(ccm1['jdate'] >= ccm1['linkdt']) & (
    ccm1['jdate'] <= ccm1['linkenddt'])].reset_index(drop=True)

# merge ccm2 and crsp2
# crsp2['jdate'] = crsp2['monthend']
data_rawq = pd.merge(crsp2, ccm2, how='inner', on=[
                     'permno', 'jdate']).reset_index(drop=True)

# # filter exchcd & shrcd and at least one year data after the IPO
# data_rawq = data_rawq[((data_rawq['exchcd'] == 1) | (data_rawq['exchcd'] == 2) | (data_rawq['exchcd'] == 3)) &
#                       ((data_rawq['shrcd'] == 10) | (data_rawq['shrcd'] == 11))].reset_index(drop=True)

# process Market Equity
'''
Note: me is CRSP market equity, mveq_f is Compustat market equity. Please choose the me below.
'''
data_rawq['me'] = data_rawq['me'] / 1000  # CRSP ME
# data_rawq['me'] = data_rawq['mveq_f']  # Compustat ME

# there are some ME equal to zero since this company do not have price or shares data, we drop these observations
data_rawq['me'] = np.where(data_rawq['me'] == 0, np.nan, data_rawq['me'])
data_rawq = data_rawq.dropna(subset=['me']).reset_index(drop=True)

# deal with the duplicates
data_rawq.loc[data_rawq.groupby(
    ['datadate', 'permno', 'linkprim'], as_index=False).nth([0]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()].reset_index(drop=True)

data_rawq.loc[data_rawq.groupby(
    ['permno', 'yearend', 'datadate'], as_index=False).nth([-1]).index, 'temp'] = 1
data_rawq = data_rawq[data_rawq['temp'].notna()].reset_index(drop=True)

data_rawq = data_rawq.sort_values(
    by=['permno', 'jdate']).reset_index(drop=True)

# add industry code for quarterly data
data_rawq = data_rawq.dropna(subset=['sic']).reset_index(
    drop=True)  # gvkey 039750 does not have sic
data_rawq['sic'] = data_rawq['sic'].astype(int)
data_rawq['ffi49'] = ffi49(data_rawq)
data_rawq['ffi49'] = data_rawq['ffi49'].fillna(49)
data_rawq['ffi49'] = data_rawq['ffi49'].astype(int)
#######################################################################################################################
#                                                   Quarterly Variables                                               #
#######################################################################################################################
# prepare be
data_rawq['beq'] = np.where(
    data_rawq['seqq'] > 0, data_rawq['seqq']+data_rawq['txditcq']-data_rawq['pstkq'], np.nan)
data_rawq['beq'] = np.where(data_rawq['beq'] <= 0, np.nan, data_rawq['beq'])

# dy
# data_rawq['me_l1'] = data_rawq.groupby(['permno'])['me'].shift(1)
# data_rawq['retdy'] = data_rawq['ret'] - data_rawq['retx']
# data_rawq['mdivpay'] = data_rawq['retdy']*data_rawq['me_l1']
#
# data_rawq['dy'] = ttm12(series='mdivpay', df=data_rawq)/data_rawq['me']

# chtx
data_rawq['txtq_l4'] = data_rawq.groupby(['permno'])['txtq'].shift(4)
data_rawq['atq_l4'] = data_rawq.groupby(['permno'])['atq'].shift(4)
data_rawq['chtx'] = (data_rawq['txtq']-data_rawq['txtq_l4']
                     )/data_rawq['atq_l4']

# roa
data_rawq['atq_l1'] = data_rawq.groupby(['permno'])['atq'].shift(1)
data_rawq['roa'] = data_rawq['ibq']/data_rawq['atq_l1']

# cash
data_rawq['cash'] = data_rawq['cheq']/data_rawq['atq']

# acc
data_rawq['actq_l4'] = data_rawq.groupby(['permno'])['actq'].shift(4)
data_rawq['lctq_l4'] = data_rawq.groupby(['permno'])['lctq'].shift(4)

# data_rawq['npq_l4'] = data_rawq.groupby(['permno'])['npq'].shift(4)
# condlist = [data_rawq['npq'].isnull(),
#             data_rawq['actq'].isnull() | data_rawq['lctq'].isnull()]
# choicelist = [((data_rawq['actq']-data_rawq['lctq'])-(data_rawq['actq_l4']-data_rawq['lctq_l4']))/(data_rawq['beq']),
#               np.nan] ##### Delete "10*" on 2025.02.26 #####
# data_rawq['acc'] = np.select(condlist, choicelist,
#                           default=((data_rawq['actq']-data_rawq['lctq']+data_rawq['npq'])-
#                                    (data_rawq['actq_l4']-data_rawq['lctq_l4']+data_rawq['npq_l4']))/(data_rawq['beq']))

#################### Added Sloan(1996) or HXZ and GHZ operating accruals on 2025.02.28 ####################
data_rawq['cheq_l4'] = data_rawq.groupby(['permno'])['cheq'].shift(4)
data_rawq['dlcq_l4'] = data_rawq.groupby(['permno'])['dlcq'].shift(4)
data_rawq['txpq_l4'] = data_rawq.groupby(['permno'])['txpq'].shift(4)

data_rawq['acc'] = np.where(data_rawq['oancfy'].isnull(),
                            ((data_rawq['actq'] - data_rawq['actq_l4']) - (data_rawq['cheq'] - data_rawq['cheq_l4']) -
                            (data_rawq['lctq'] - data_rawq['lctq_l4']) + (data_rawq['dlcq'] - data_rawq['dlcq_l4']) +
                            (data_rawq['txpq'] - data_rawq['txpq_l4']).fillna(0) - data_rawq['dpq']) / ((data_rawq['atq'] + data_rawq['atq_l4']) / 2),
                            (data_rawq['ibq'] - data_rawq['oancfy']) / ((data_rawq['atq'] + data_rawq['atq_l4']) / 2))

# absacc
data_rawq['absacc'] = abs(data_rawq['acc'])

# bm
# data_rawq['bm'] = data_rawq['beq']/data_rawq['me']

# cfp
data_rawq['ibq4'] = ttm4('ibq', data_rawq)
data_rawq['dpq4'] = ttm4('dpq', data_rawq)
# data_rawq['cfp'] = np.where(data_rawq['dpq'].isnull(),
#                             data_rawq['ibq4']/data_rawq['me'],
#                             (data_rawq['ibq4']+data_rawq['dpq4'])/data_rawq['me'])

# ep
# data_rawq['ep'] = data_rawq['ibq4']/data_rawq['me']

# agr
data_rawq['agr'] = (data_rawq['atq']-data_rawq['atq_l4'])/data_rawq['atq_l4']

# ni
data_rawq['cshoq_l4'] = data_rawq.groupby(['permno'])['cshoq'].shift(4)
data_rawq['ajexq_l4'] = data_rawq.groupby(['permno'])['ajexq'].shift(4)
data_rawq['ni'] = np.where(data_rawq['cshoq'].isnull(), np.nan,
                           np.log(data_rawq['cshoq']*data_rawq['ajexq']).replace(-np.inf, 0)-np.log(data_rawq['cshoq_l4']*data_rawq['ajexq_l4']))

# op
data_rawq['xintq0'] = np.where(
    data_rawq['xintq'].isnull(), 0, data_rawq['xintq'])
data_rawq['xsgaq0'] = np.where(
    data_rawq['xsgaq'].isnull(), 0, data_rawq['xsgaq'])
data_rawq['beq_l4'] = data_rawq.groupby(['permno'])['beq'].shift(4)

data_rawq['op'] = (ttm4('revtq', data_rawq)-ttm4('cogsq', data_rawq) -
                   ttm4('xsgaq0', data_rawq)-ttm4('xintq0', data_rawq))/data_rawq['beq_l4']

# chcsho
data_rawq['chcsho'] = (data_rawq['cshoq']/data_rawq['cshoq_l4'])-1

# cashdebt
data_rawq['ltq_l4'] = data_rawq.groupby(['permno'])['ltq'].shift(4)
data_rawq['cashdebt'] = (ttm4('ibq', data_rawq) + ttm4('dpq',
                         data_rawq))/((data_rawq['ltq']+data_rawq['ltq_l4'])/2)

# rd
data_rawq['xrdq4'] = ttm4('xrdq', data_rawq)
data_rawq['xrdq4'] = np.where(
    data_rawq['xrdq4'].isnull(), data_rawq['xrdy'], data_rawq['xrdq4'])

data_rawq['xrdq4/atq_l4'] = data_rawq['xrdq4']/data_rawq['atq_l4']
data_rawq['xrdq4/atq_l4_l4'] = data_rawq.groupby(
    ['permno'])['xrdq4/atq_l4'].shift(4)
data_rawq['rd'] = np.where(((data_rawq['xrdq4']/data_rawq['atq']) -
                           data_rawq['xrdq4/atq_l4_l4'])/data_rawq['xrdq4/atq_l4_l4'] > 0.05, 1, 0)

#################### Follow Hafzalla, Lundholm, and Van Winkle (2011) and GHZ on 2025.02.28 ####################

# # pctacc
# condlist = [data_rawq['npq'].isnull(),
#             data_rawq['actq'].isnull() | data_rawq['lctq'].isnull()]
# choicelist = [((data_rawq['actq']-data_rawq['lctq'])-(data_rawq['actq_l4']-data_rawq['lctq_l4']))/abs(ttm4('ibq', data_rawq)), np.nan]
# data_rawq['pctacc'] = np.select(condlist, choicelist,
#                               default=((data_rawq['actq']-data_rawq['lctq']+data_rawq['npq'])-(data_rawq['actq_l4']-data_rawq['lctq_l4']+data_rawq['npq_l4']))/
#                                       abs(ttm4('ibq', data_rawq)))

condlist = [data_rawq['ibq'] == 0,
            data_rawq['oancfy'].isnull(),
            data_rawq['oancfy'].isnull() & data_rawq['ibq'] == 0]
choicelist = [(data_rawq['ibq'] - data_rawq['oancfy']) / 0.01,
              ((data_rawq['actq'] - data_rawq['actq_l4']) - (data_rawq['cheq'] - data_rawq['cheq_l4']) -
               (data_rawq['lctq'] - data_rawq['lctq_l4']) + (data_rawq['dlcq'] - data_rawq['dlcq_l4']) +
               (data_rawq['txpq'] - data_rawq['txpq_l4']).fillna(0) - data_rawq['dpq']) / data_rawq['ibq'].abs(),
              ((data_rawq['actq'] - data_rawq['actq_l4']) - (data_rawq['cheq'] - data_rawq['cheq_l4']) -
               (data_rawq['lctq'] - data_rawq['lctq_l4']) + (data_rawq['dlcq'] - data_rawq['dlcq_l4']) +
               (data_rawq['txpq'] - data_rawq['txpq_l4']).fillna(0) - data_rawq['dpq']) / 0.01]
data_rawq['pctacc'] = np.select(condlist, choicelist,
                                default=(data_rawq['ibq'] - data_rawq['oancfy']) / data_rawq['ibq'].abs())

# gma
data_rawq['revtq4'] = ttm4('revtq', data_rawq)
data_rawq['cogsq4'] = ttm4('cogsq', data_rawq)
data_rawq['gma'] = (data_rawq['revtq4']-data_rawq['cogsq4']
                    )/data_rawq['atq_l4']

# lev
# data_rawq['lev'] = data_rawq['ltq']/data_rawq['me']

# rdm
# data_rawq['rdm'] = data_rawq['xrdq4']/data_rawq['me']

# sgr
data_rawq['saleq4'] = ttm4('saleq', data_rawq)
data_rawq['saleq4'] = np.where(
    data_rawq['saleq4'].isnull(), data_rawq['saley'], data_rawq['saleq4'])

data_rawq['saleq4_l4'] = data_rawq.groupby(['permno'])['saleq4'].shift(4)
data_rawq['sgr'] = (data_rawq['saleq4']/data_rawq['saleq4_l4'])-1

# sp
# data_rawq['sp'] = data_rawq['saleq4']/data_rawq['me']

# invest
data_rawq['ppentq_l4'] = data_rawq.groupby(['permno'])['ppentq'].shift(4)
data_rawq['invtq_l4'] = data_rawq.groupby(['permno'])['invtq'].shift(4)
data_rawq['ppegtq_l4'] = data_rawq.groupby(['permno'])['ppegtq'].shift(4)

data_rawq['invest'] = np.where(data_rawq['ppegtq'].isnull(), ((data_rawq['ppentq']-data_rawq['ppentq_l4']) +
                                                              (data_rawq['invtq']-data_rawq['invtq_l4']))/data_rawq['atq_l4'],
                               ((data_rawq['ppegtq']-data_rawq['ppegtq_l4'])+(data_rawq['invtq']-data_rawq['invtq_l4']))/data_rawq['atq_l4'])

# rd_sale
data_rawq['rd_sale'] = data_rawq['xrdq4']/data_rawq['saleq4']

# lgr
data_rawq['lgr'] = (data_rawq['ltq']/data_rawq['ltq_l4'])-1

# depr
data_rawq['depr'] = ttm4('dpq', data_rawq)/data_rawq['ppentq']

# egr
data_rawq['ceqq_l4'] = data_rawq.groupby(['permno'])['ceqq'].shift(4)
data_rawq['egr'] = (data_rawq['ceqq']-data_rawq['ceqq_l4']) / \
    data_rawq['ceqq_l4']

# chpm
data_rawq['ibq4_l1'] = data_rawq.groupby(['permno'])['ibq4'].shift(1)
data_rawq['saleq4_l1'] = data_rawq.groupby(['permno'])['saleq4'].shift(1)

data_rawq['chpm'] = (data_rawq['ibq4']/data_rawq['saleq4']) - \
    (data_rawq['ibq4_l1']/data_rawq['saleq4_l1'])

# chato
data_rawq['atq_l8'] = data_rawq.groupby(['permno'])['atq'].shift(8)
data_rawq['chato'] = (data_rawq['saleq4']/((data_rawq['atq']+data_rawq['atq_l4'])/2)) - \
    (data_rawq['saleq4_l4']/((data_rawq['atq_l4']+data_rawq['atq_l8'])/2))

# chatoia
df_temp = data_rawq.groupby(['datadate', 'ffi49'], as_index=False)[
    'chato'].mean()
df_temp = df_temp.rename(columns={'chato': 'chato_ind'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawq['chatoia'] = data_rawq['chato'] - data_rawq['chato_ind']

# noa
data_rawq['ivaoq'] = np.where(data_rawq['ivaoq'].isnull(), 0, 1)
data_rawq['dlcq'] = np.where(data_rawq['dlcq'].isnull(), 0, 1)
data_rawq['dlttq'] = np.where(data_rawq['dlttq'].isnull(), 0, 1)
data_rawq['mibq'] = np.where(data_rawq['mibq'].isnull(), 0, 1)
data_rawq['pstkq'] = np.where(data_rawq['pstkq'].isnull(), 0, 1)
data_rawq['noa'] = (data_rawq['atq']-data_rawq['cheq']-data_rawq['ivaoq']) -\
    (data_rawq['atq']-data_rawq['dlcq']-data_rawq['dlttq']-data_rawq['mibq'] -
     data_rawq['pstkq']-data_rawq['ceqq'])/data_rawq['atq_l4']

# rna
data_rawq['noa_l4'] = data_rawq.groupby(['permno'])['noa'].shift(4)
data_rawq['rna'] = data_rawq['oiadpq']/data_rawq['noa_l4']

# pm
data_rawq['pm'] = data_rawq['oiadpq']/data_rawq['saleq']

# ato
data_rawq['ato'] = data_rawq['saleq']/data_rawq['noa_l4']

# roe
data_rawq['ceqq_l1'] = data_rawq.groupby(['permno'])['ceqq'].shift(1)
data_rawq['roe'] = data_rawq['ibq']/data_rawq['ceqq_l1']

################################## New Added ##################################

# grltnoa
data_rawq['rectq_l4'] = data_rawq.groupby(['permno'])['rectq'].shift(4)
data_rawq['acoq_l4'] = data_rawq.groupby(['permno'])['acoq'].shift(4)
data_rawq['apq_l4'] = data_rawq.groupby(['permno'])['apq'].shift(4)
data_rawq['lcoq_l4'] = data_rawq.groupby(['permno'])['lcoq'].shift(4)
data_rawq['loq_l4'] = data_rawq.groupby(['permno'])['loq'].shift(4)
data_rawq['invtq_l4'] = data_rawq.groupby(['permno'])['invtq'].shift(4)
data_rawq['ppentq_l4'] = data_rawq.groupby(['permno'])['ppentq'].shift(4)
data_rawq['atq_l4'] = data_rawq.groupby(['permno'])['atq'].shift(4)

data_rawq['grltnoa'] = ((data_rawq['rectq']+data_rawq['invtq']+data_rawq['ppentq']+data_rawq['acoq']+data_rawq['intanq'] +
                         data_rawq['aoq']-data_rawq['apq']-data_rawq['lcoq']-data_rawq['loq']) -
                        (data_rawq['rectq_l4']+data_rawq['invtq_l4']+data_rawq['ppentq_l4']+data_rawq['acoq_l4']-data_rawq['apq_l4']-data_rawq['lcoq_l4']-data_rawq['loq_l4']) -
                        (data_rawq['rectq']-data_rawq['rectq_l4']+data_rawq['invtq']-data_rawq['invtq_l4']+data_rawq['acoq'] -
                         (data_rawq['apq']-data_rawq['apq_l4']+data_rawq['lcoq']-data_rawq['lcoq_l4']) -
                         ttm4('dpq', data_rawq)))/((data_rawq['atq']+data_rawq['atq_l4'])/2)

# scal
# condlist = [data_rawq['seqq'].isnull(),
#             data_rawq['seqq'].isnull() & (data_rawq['ceqq'].isnull() | data_rawq['pstk'].isnull())]
# choicelist = [data_rawq['ceqq']+data_rawq['pstk'],
#               data_rawq['atq']-data_rawq['ltq']]
# data_rawq['scal'] = np.select(condlist, choicelist, default=data_rawq['seqq'])

# ala
data_rawq['gdwlq'] = np.where(
    data_rawq['gdwlq'].isnull(), 0, data_rawq['gdwlq'])
data_rawq['intanq'] = np.where(
    data_rawq['intanq'].isnull(), 0, data_rawq['intanq'])
data_rawq['ala'] = data_rawq['cheq'] + 0.75*(data_rawq['actq']-data_rawq['cheq']) +\
    0.5*(data_rawq['atq']-data_rawq['actq'] -
         data_rawq['gdwlq']-data_rawq['intanq'])

# alm
# data_rawq['alm'] = data_rawq['ala']/(data_rawq['atq']+data_rawq['me']-data_rawq['ceqq'])

# rsup
data_rawq['saleq_l4'] = data_rawq.groupby(['permno'])['saleq'].shift(4)
# data_rawq['rsup'] = (data_rawq['saleq'] - data_rawq['saleq_l4'])/data_rawq['me']

# stdsacc
data_rawq['actq_l1'] = data_rawq.groupby(['permno'])['actq'].shift(1)
data_rawq['cheq_l1'] = data_rawq.groupby(['permno'])['cheq'].shift(1)
data_rawq['lctq_l1'] = data_rawq.groupby(['permno'])['lctq'].shift(1)
data_rawq['dlcq_l1'] = data_rawq.groupby(['permno'])['dlcq'].shift(1)

data_rawq['sacc'] = ((data_rawq['actq']-data_rawq['actq_l1'] - (data_rawq['cheq']-data_rawq['cheq_l1']))
                     - ((data_rawq['lctq']-data_rawq['lctq_l1'])-(data_rawq['dlcq']-data_rawq['dlcq_l1'])))/data_rawq['saleq']
data_rawq['sacc'] = np.where(data_rawq['saleq'] <= 0, ((data_rawq['actq']-data_rawq['actq_l1'] - (data_rawq['cheq']-data_rawq['cheq_l1']))
                                                       - ((data_rawq['lctq']-data_rawq['lctq_l1'])-(data_rawq['dlcq']-data_rawq['dlcq_l1'])))/0.01, data_rawq['sacc'])


def chars_std(start, end, df, chars):
    """

    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :param chars: lag chars
    :return: std of factor
    """
    lag = pd.DataFrame()
    lag_list = []
    for i in range(start, end):
        lag['chars_l%s' % i] = df.groupby(['permno'])['%s' % chars].shift(i)
        lag_list.append('chars_l%s' % i)
    result = lag[lag_list].std(axis=1)
    return result


data_rawq['stdacc'] = chars_std(0, 16, data_rawq, 'sacc')

# roavol
data_rawq['roavol'] = chars_std(0, 16, data_rawq, 'roa')

# stdcf
data_rawq['scf'] = (data_rawq['ibq']/data_rawq['saleq']) - data_rawq['sacc']
data_rawq['scf'] = np.where(
    data_rawq['saleq'] <= 0, (data_rawq['ibq']/0.01) - data_rawq['sacc'], data_rawq['sacc'])

data_rawq['stdcf'] = chars_std(0, 16, data_rawq, 'scf')

# cinvest
data_rawq['ppentq_l1'] = data_rawq.groupby(['permno'])['ppentq'].shift(1)
data_rawq['ppentq_l2'] = data_rawq.groupby(['permno'])['ppentq'].shift(2)
data_rawq['ppentq_l3'] = data_rawq.groupby(['permno'])['ppentq'].shift(3)
data_rawq['ppentq_l4'] = data_rawq.groupby(['permno'])['ppentq'].shift(4)
data_rawq['saleq_l1'] = data_rawq.groupby(['permno'])['saleq'].shift(1)
data_rawq['saleq_l2'] = data_rawq.groupby(['permno'])['saleq'].shift(2)
data_rawq['saleq_l3'] = data_rawq.groupby(['permno'])['saleq'].shift(3)

data_rawq['c_temp1'] = (data_rawq['ppentq_l1'] -
                        data_rawq['ppentq_l2']) / data_rawq['saleq_l1']
data_rawq['c_temp2'] = (data_rawq['ppentq_l2'] -
                        data_rawq['ppentq_l3']) / data_rawq['saleq_l2']
data_rawq['c_temp3'] = (data_rawq['ppentq_l3'] -
                        data_rawq['ppentq_l4']) / data_rawq['saleq_l3']

data_rawq['cinvest'] = ((data_rawq['ppentq'] - data_rawq['ppentq_l1']) / data_rawq['saleq'])\
    - (data_rawq[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1))

data_rawq['c_temp1'] = (data_rawq['ppentq_l1'] - data_rawq['ppentq_l2']) / 0.01
data_rawq['c_temp2'] = (data_rawq['ppentq_l2'] - data_rawq['ppentq_l3']) / 0.01
data_rawq['c_temp3'] = (data_rawq['ppentq_l3'] - data_rawq['ppentq_l4']) / 0.01

data_rawq['cinvest'] = np.where(data_rawq['saleq'] <= 0, ((data_rawq['ppentq'] - data_rawq['ppentq_l1']) / 0.01)
                                - (data_rawq[['c_temp1', 'c_temp2', 'c_temp3']].mean(axis=1)), data_rawq['cinvest'])

data_rawq = data_rawq.drop(['c_temp1', 'c_temp2', 'c_temp3'], axis=1)

# nincr
data_rawq['ibq_l1'] = data_rawq.groupby(['permno'])['ibq'].shift(1)
data_rawq['ibq_l2'] = data_rawq.groupby(['permno'])['ibq'].shift(2)
data_rawq['ibq_l3'] = data_rawq.groupby(['permno'])['ibq'].shift(3)
data_rawq['ibq_l4'] = data_rawq.groupby(['permno'])['ibq'].shift(4)
data_rawq['ibq_l5'] = data_rawq.groupby(['permno'])['ibq'].shift(5)
data_rawq['ibq_l6'] = data_rawq.groupby(['permno'])['ibq'].shift(6)
data_rawq['ibq_l7'] = data_rawq.groupby(['permno'])['ibq'].shift(7)
data_rawq['ibq_l8'] = data_rawq.groupby(['permno'])['ibq'].shift(8)

data_rawq['nincr_temp1'] = np.where(
    data_rawq['ibq'] > data_rawq['ibq_l1'], 1, 0)
data_rawq['nincr_temp2'] = np.where(
    data_rawq['ibq_l1'] > data_rawq['ibq_l2'], 1, 0)
data_rawq['nincr_temp3'] = np.where(
    data_rawq['ibq_l2'] > data_rawq['ibq_l3'], 1, 0)
data_rawq['nincr_temp4'] = np.where(
    data_rawq['ibq_l3'] > data_rawq['ibq_l4'], 1, 0)
data_rawq['nincr_temp5'] = np.where(
    data_rawq['ibq_l4'] > data_rawq['ibq_l5'], 1, 0)
data_rawq['nincr_temp6'] = np.where(
    data_rawq['ibq_l5'] > data_rawq['ibq_l6'], 1, 0)
data_rawq['nincr_temp7'] = np.where(
    data_rawq['ibq_l6'] > data_rawq['ibq_l7'], 1, 0)
data_rawq['nincr_temp8'] = np.where(
    data_rawq['ibq_l7'] > data_rawq['ibq_l8'], 1, 0)

data_rawq['nincr'] = (data_rawq['nincr_temp1']
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2'])
                      + (data_rawq['nincr_temp1'] *
                         data_rawq['nincr_temp2']*data_rawq['nincr_temp3'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']
                         * data_rawq['nincr_temp3']*data_rawq['nincr_temp4'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2'] *
                         data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']
                         * data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3'] *
                         data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6']*data_rawq['nincr_temp7'])
                      + (data_rawq['nincr_temp1']*data_rawq['nincr_temp2']*data_rawq['nincr_temp3']*data_rawq['nincr_temp4']*data_rawq['nincr_temp5']*data_rawq['nincr_temp6']*data_rawq['nincr_temp7']*data_rawq['nincr_temp8']))

data_rawq = data_rawq.drop(['ibq_l1', 'ibq_l2', 'ibq_l3', 'ibq_l4', 'ibq_l5', 'ibq_l6', 'ibq_l7', 'ibq_l8', 'nincr_temp1',
                            'nincr_temp2', 'nincr_temp3', 'nincr_temp4', 'nincr_temp5', 'nincr_temp6', 'nincr_temp7',
                            'nincr_temp8'], axis=1)

# performance score
data_rawq['niq4'] = ttm4(series='niq', df=data_rawq)
data_rawq['niq4_l4'] = data_rawq.groupby(['permno'])['niq4'].shift(4)
data_rawq['dlttq_l4'] = data_rawq.groupby(['permno'])['dlttq'].shift(4)
data_rawq['p_temp1'] = np.where(data_rawq['niq4'] > 0, 1, 0)
data_rawq['p_temp2'] = np.where(data_rawq['oancfy'] > 0, 1, 0)
data_rawq['p_temp3'] = np.where(
    data_rawq['niq4']/data_rawq['atq'] > data_rawq['niq4_l4']/data_rawq['atq_l4'], 1, 0)
data_rawq['p_temp4'] = np.where(data_rawq['oancfy'] > data_rawq['niq4'], 1, 0)
data_rawq['p_temp5'] = np.where(
    data_rawq['dlttq']/data_rawq['atq'] < data_rawq['dlttq_l4']/data_rawq['atq_l4'], 1, 0)
data_rawq['p_temp6'] = np.where(
    data_rawq['actq']/data_rawq['lctq'] > data_rawq['actq_l4']/data_rawq['lctq_l4'], 1, 0)
data_rawq['cogsq4_l4'] = data_rawq.groupby(['permno'])['cogsq4'].shift(4)
data_rawq['p_temp7'] = np.where((data_rawq['saleq4']-data_rawq['cogsq4']/data_rawq['saleq4']) > (
    data_rawq['saleq4_l4']-data_rawq['cogsq4_l4']/data_rawq['saleq4_l4']), 1, 0)
data_rawq['p_temp8'] = np.where(
    data_rawq['saleq4']/data_rawq['atq'] > data_rawq['saleq4_l4']/data_rawq['atq_l4'], 1, 0)
data_rawq['p_temp9'] = np.where(data_rawq['scstkcy'] == 0, 1, 0)

data_rawq['pscore'] = data_rawq['p_temp1']+data_rawq['p_temp2']+data_rawq['p_temp3']+data_rawq['p_temp4']\
    + data_rawq['p_temp5']+data_rawq['p_temp6']+data_rawq['p_temp7']+data_rawq['p_temp8']\
    + data_rawq['p_temp9']

data_rawq = data_rawq.drop(['p_temp1', 'p_temp2', 'p_temp3', 'p_temp4', 'p_temp5', 'p_temp6', 'p_temp7', 'p_temp8',
                            'p_temp9'], axis=1)

################## Added on 2022.09.06 ##################
# cashpr
# data_rawq['cashpr'] = ((data_rawq['me'] + data_rawq['dlttq'] - data_rawq['atq']) / data_rawq['cheq'])

print("Finish Quarterly Variables Calculation! \n")

#######################################################################################################################
#                                                       Momentum                                                      #
#######################################################################################################################
crsp = conn.raw_sql("""
                      select a.mthprc, a.mthret, a.mthretx, a.mthvol, a.shrout,
                      a.permno, a.permco, a.mthcaldt,
                      a.issuertype, a.securitytype, a.securitysubtype, a.sharetype, a.usincflg,
                      a.primaryexch, a.conditionaltype, a.TradingStatusFlg
                      from crspq.msf_v2 as a
                      where a.mthcaldt >= '01/01/1959'
                      """, date_cols=['mthcaldt'])

# equivalent to legacy code exchcd = 1, 2 or 3
crsp = crsp.loc[(crsp.primaryexch.isin(['N', 'A', 'Q'])) &
                (crsp.conditionaltype == 'RW') &
                (crsp.tradingstatusflg == 'A')]
# crsp['exchcd'] = crsp['primaryexch'].map({'N': 1, 'A': 2, 'Q': 3})
# # equivalent to legacy code shrcd = 10 or 11
# crsp = crsp.loc[(crsp.sharetype == 'NS') &
#                 (crsp.securitytype == 'EQTY') &
#                 (crsp.securitysubtype == 'COM') &
#                 (crsp.usincflg == 'Y') &
#                 (crsp.issuertype.isin(['ACOR', 'CORP']))]
crsp.drop(['primaryexch', 'conditionaltype', 'tradingstatusflg', 'securitytype', 'securitysubtype',
           'sharetype', 'usincflg', 'issuertype'], axis=1, inplace=True)

crsp.rename(columns={
    'mthprc': 'prc',
    'mthret': 'ret',
    'mthretx': 'retx',
    'mthvol': 'vol',
    'mthcaldt': 'date',
}, inplace=True)

crsp = crsp.dropna(subset=['ret', 'retx', 'prc']
                   ).reset_index(drop=True)  # 最后comment

# change variable format to int
crsp[['permco', 'permno']] = crsp[['permco', 'permno']].astype(int)

# Line up date to be end of month
# set all the date to the standard end date of month
crsp['jdate'] = crsp['date'] + MonthEnd(0)
crsp['ret'] = crsp['ret'].fillna(0)
crsp['retx'] = crsp['retx'].fillna(0)
crsp['me'] = crsp['prc'].abs() * crsp['shrout']

# Aggregate Market Cap
'''
There are cases when the same firm (permco) has two or more securities (permno) at same date.
For the purpose of ME for the firm, we aggregated all ME for a given permco, date.
This aggregated ME will be assigned to the permno with the largest ME.
'''
# sum of me across different permno belonging to same permco a given date
crsp_summe = crsp.groupby(['jdate', 'permco'])['me'].sum().reset_index()
# largest mktcap within a permco/date
crsp_maxme = crsp.groupby(['jdate', 'permco'])['me'].max().reset_index()
# join by monthend/maxme to find the permno
crsp1 = pd.merge(crsp, crsp_maxme, how='inner', on=[
                 'jdate', 'permco', 'me']).reset_index(drop=True)
# drop me column and replace with the sum me
crsp1 = crsp1.drop(['me'], axis=1)
# join with sum of me to get the correct market cap info
crsp2 = pd.merge(crsp1, crsp_summe, how='inner', on=[
                 'jdate', 'permco']).reset_index(drop=True)
# sort by permno and date and also drop duplicates
crsp2 = crsp2.sort_values(by=['permno', 'jdate']
                          ).drop_duplicates().reset_index(drop=True)

################## Added on 2025.02.23 ##################
crsp2['me'] = crsp2['me']/1000  # CRSP ME in million unit

crsp_mom = crsp2.copy()
crsp_mom = crsp_mom.sort_values(by=['permno', 'date']).reset_index(drop=True)

# No need to add delisting return in the new CIZ CRSP format


def mom(start, end, df):
    """

    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result = result * (1+lag['mom%s' % i])
    result = result - 1
    return result


def chmom(start, end, df):
    """

    :param start: Order of starting lag
    :param end: Order of ending lag
    :param df: Dataframe
    :return: Momentum factor
    """
    lag = pd.DataFrame()
    result_first_half = 1
    result_second_half = 1
    for i in range(start, end):
        lag['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result_first_half = result_first_half * (1+lag['mom%s' % i])
    lag = pd.DataFrame()
    for i in range(start + 6, end + 6):
        lag['mom%s' % i] = df.groupby(['permno'])['ret'].shift(i)
        result_second_half = result_second_half * (1 + lag['mom%s' % i])
    result_first_half = result_first_half - 1
    result_second_half = result_second_half - 1
    result = result_first_half - result_second_half
    return result


crsp_mom['chmom'] = chmom(1, 12, crsp_mom)

crsp_mom['mom60m'] = mom(12, 60, crsp_mom)
crsp_mom['mom12m'] = mom(1, 12, crsp_mom)
crsp_mom['mom1m'] = crsp_mom['ret']
crsp_mom['mom6m'] = mom(1, 6, crsp_mom)
crsp_mom['mom36m'] = mom(12, 36, crsp_mom)
crsp_mom['seas1a'] = crsp_mom.groupby(['permno'])['ret'].shift(11)

crsp_mom['vol_l1'] = crsp_mom.groupby(['permno'])['vol'].shift(1)
crsp_mom['vol_l2'] = crsp_mom.groupby(['permno'])['vol'].shift(2)
crsp_mom['vol_l3'] = crsp_mom.groupby(['permno'])['vol'].shift(3)
crsp_mom['prc_l2'] = crsp_mom.groupby(['permno'])['prc'].shift(2)
# crsp_mom['dolvol'] = np.log((crsp_mom['vol_l2']*100)*crsp_mom['prc_l2']).replace([np.inf, -np.inf], np.nan) ##### Added "*100" on 2025.02.23 (change "vol" unit from hundreds to one unit) #####
# crsp_mom['turn'] = ((crsp_mom['vol_l1']+crsp_mom['vol_l2']+crsp_mom['vol_l3'])/3/10)/crsp_mom['shrout'] ##### Added "/10" on 2025.02.23 (change "vol" unit from hundreds to thousand unit, same as shrout) #####

# 2025-07-06 updates: In SIZ version, vol(daily-1 units, monthly-100 units), shrout-1000 units.
# In CIZ version, vol(monthly-1 units)
crsp_mom['dolvol'] = np.log(
    (crsp_mom['vol_l2'])*crsp_mom['prc_l2']).replace([np.inf, -np.inf], np.nan)
crsp_mom['turn'] = ((crsp_mom['vol_l1']+crsp_mom['vol_l2'] +
                    crsp_mom['vol_l3'])/3/1000)/crsp_mom['shrout']

# dy
crsp_mom['me_l1'] = crsp_mom.groupby(['permno'])['me'].shift(1)
crsp_mom['retdy'] = crsp_mom['ret'] - crsp_mom['retx']
crsp_mom['mdivpay'] = crsp_mom['retdy']*crsp_mom['me_l1']

crsp_mom['dy'] = ttm12(series='mdivpay', df=crsp_mom)/crsp_mom['me']

# def moms(start, end, df):
#     """
#
#     :param start: Order of starting lag
#     :param end: Order of ending lag
#     :param df: Dataframe
#     :return: Momentum factor
#     """
#     lag = pd.DataFrame()
#     result = 1
#     for i in range(start, end):
#         lag['moms%s' % i] = df.groupby['permno']['ret'].shift(i)
#         result = result + lag['moms%s' % i]
#     result = result/11
#     return result
#
#
# crsp_mom['moms12m'] = moms(1, 12, crsp_mom)

# populate the chars to monthly

# data_rawa
data_rawa = data_rawa.drop(columns=[
                           'date', 'ret', 'retx', 'me', 'vol', 'permco', 'prc', 'shrout'], errors='ignore')
data_rawa = pd.merge(crsp_mom, data_rawa, how='left', on=[
                     'permno', 'jdate']).reset_index(drop=True)
data_rawa = data_rawa.sort_values(
    by=['permno', 'jdate']).reset_index(drop=True)
data_rawa['datadate'] = data_rawa.groupby(['permno'])['datadate'].fillna(
    method='ffill')  # avoid the bug of 'groupby' for py 3.8
data_rawa[['permno1', 'datadate1']] = data_rawa[['permno', 'datadate']]
data_rawa = data_rawa.groupby(
    ['permno1', 'datadate1'], as_index=False).fillna(method='ffill')
data_rawa = data_rawa.loc[
    (data_rawa['primaryexch'].isin(['N', 'A', 'Q'])) &
    (data_rawa['conditionaltype'] == 'RW') &
    (data_rawa['tradingstatusflg'] == 'A')
].reset_index(drop=True)

# data_rawq
data_rawq = data_rawq.drop(columns=[
                           'date', 'ret', 'retx', 'me', 'vol', 'permco', 'prc', 'shrout'], errors='ignore')
data_rawq = pd.merge(crsp_mom, data_rawq, how='left', on=[
                     'permno', 'jdate']).reset_index(drop=True)
data_rawq = data_rawq.sort_values(
    by=['permno', 'jdate']).reset_index(drop=True)
data_rawq['datadate'] = data_rawq.groupby(['permno'])['datadate'].fillna(
    method='ffill')  # avoid the bug of 'groupby' for py 3.8
data_rawq[['permno1', 'datadate1']] = data_rawq[['permno', 'datadate']]
data_rawq = data_rawq.groupby(
    ['permno1', 'datadate1'], as_index=False).fillna(method='ffill')
data_rawq = data_rawq.loc[
    (data_rawq['primaryexch'].isin(['N', 'A', 'Q'])) &
    (data_rawq['conditionaltype'] == 'RW') &
    (data_rawq['tradingstatusflg'] == 'A')
].reset_index(drop=True)

#######################################################################################################################
#                                                    Monthly ME                                                       #
#######################################################################################################################

########################################
#                Annual                #
########################################

# bm
data_rawa['bm'] = data_rawa['be'] / data_rawa['me']

# bm_ia
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)['bm'].mean()
df_temp = df_temp.rename(columns={'bm': 'bm_ind'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawa['bm_ia'] = data_rawa['bm'] - data_rawa['bm_ind']

# me_ia
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)['me'].mean()
df_temp = df_temp.rename(columns={'me': 'me_ind'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawa['me_ia'] = data_rawa['me'] - data_rawa['me_ind']

# cfp
condlist = [data_rawa['dp'].isnull(),
            data_rawa['ib'].isnull()]
choicelist = [data_rawa['ib']/data_rawa['me'],
              np.nan]
data_rawa['cfp'] = np.select(condlist, choicelist, default=(
    data_rawa['ib']+data_rawa['dp'])/data_rawa['me'])

# cfp_ia
df_temp = data_rawa.groupby(['datadate', 'ffi49'], as_index=False)[
    'cfp'].mean()
df_temp = df_temp.rename(columns={'cfp': 'cfp_ind'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawa['cfp_ia'] = data_rawa['cfp'] - data_rawa['cfp_ind']

# ep
data_rawa['ep'] = data_rawa['ib']/data_rawa['me']

# rsup
# data_rawa['sale_l1'] = data_rawa.groupby(['permno'])['sale'].shift(1)
data_rawa['rsup'] = (data_rawa['sale']-data_rawa['sale_l1'])/data_rawa['me']

# lev
data_rawa['lev'] = data_rawa['lt']/data_rawa['me']

# sp
data_rawa['sp'] = data_rawa['sale']/data_rawa['me']

# rdm
data_rawa['rdm'] = data_rawa['xrd']/data_rawa['me']

# adm hxz adm
data_rawa['adm'] = data_rawa['xad']/data_rawa['me']

# dy
data_rawa['dy'] = data_rawa['dvt']/data_rawa['me']

# cashpr
data_rawa['cashpr'] = (
    (data_rawa['me'] + data_rawa['dltt'] - data_rawa['at']) / data_rawa['che'])

# indmom
df_temp = data_rawa.groupby(['date', 'ffi49'], as_index=False)[
    'mom12m'].mean().rename(columns={'mom12m': 'indmom'})
data_rawa = pd.merge(data_rawa, df_temp, how='left', on=[
                     'date', 'ffi49']).reset_index(drop=True)

# Annual Accounting Variables
# replace 'exchcd','shrcd' with 'primaryexch', 'conditionaltype', 'tradingstatusflg', 'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype'
chars_a = data_rawa[['cusip_comp', 'cusip_crsp', 'hdrcusip', 'gvkey', 'permno', 'primaryexch', 'conditionaltype', 'tradingstatusflg',
                     'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype',
                     'datadate', 'jdate', 'ticker', 'conm', 'comnam', 'prc', 'shrout',
                     'sic', 'ret', 'retx', 'acc', 'agr', 'bm', 'cfp', 'ep', 'ni', 'op', 'rsup', 'cash', 'chcsho',
                     'rd', 'cashdebt', 'pctacc', 'gma', 'lev', 'rdm', 'adm', 'sgr', 'sp', 'invest', 'roe',
                     'rd_sale', 'lgr', 'roa', 'depr', 'egr', 'chato', 'chtx', 'noa', 'rna', 'pm', 'ato', 'dy',
                     'roic', 'chinv', 'pchsale_pchinvt', 'pchsale_pchrect', 'pchgm_pchsale', 'pchsale_pchxsga',
                     'pchdepr', 'chadv', 'pchcapx', 'grcapx', 'grGW', 'currat', 'pchcurrat', 'quick', 'pchquick',
                     'salecash', 'salerec', 'saleinv', 'pchsaleinv', 'realestate', 'obklg', 'chobklg', 'grltnoa',
                     'conv', 'chdrc', 'rdbias', 'operprof', 'capxint', 'xadint', 'chpm', 'ala', 'alm',
                     'mom1m', 'mom6m', 'mom12m', 'mom60m', 'mom36m', 'seas1a', 'me', 'hire', 'herf', 'bm_ia',
                     'me_ia', 'turn', 'dolvol', 'absacc', 'age', 'cashpr', 'chatoia', 'chempia', 'chmom', 'chpmia',
                     'convind', 'divi', 'divo', 'secured', 'securedind', 'sin', 'cfp_ia', 'indmom', 'pchcapx_ia',
                     'tang', 'tb', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6']]
chars_a.reset_index(drop=True, inplace=True)

########################################
#               Quarterly              #
########################################
# bm
data_rawq['bm'] = data_rawq['beq']/data_rawq['me']

# bm_ia
df_temp = data_rawq.groupby(['datadate', 'ffi49'], as_index=False)['bm'].mean()
df_temp = df_temp.rename(columns={'bm': 'bm_ind'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawq['bm_ia'] = data_rawq['bm'] - data_rawq['bm_ind']

# me_ia
df_temp = data_rawq.groupby(['datadate', 'ffi49'], as_index=False)['me'].mean()
df_temp = df_temp.rename(columns={'me': 'me_ind'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawq['me_ia'] = data_rawq['me'] - data_rawq['me_ind']

# cfp
data_rawq['cfp'] = np.where(data_rawq['dpq'].isnull(),
                            data_rawq['ibq4']/data_rawq['me'],
                            (data_rawq['ibq4']+data_rawq['dpq4'])/data_rawq['me'])

# cfp_ia
df_temp = data_rawq.groupby(['datadate', 'ffi49'], as_index=False)[
    'cfp'].mean()
df_temp = df_temp.rename(columns={'cfp': 'cfp_ind'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'datadate', 'ffi49']).reset_index(drop=True)
data_rawq['cfp_ia'] = data_rawq['cfp'] - data_rawq['cfp_ind']

# ep
data_rawq['ep'] = data_rawq['ibq4']/data_rawq['me']

# lev
data_rawq['lev'] = data_rawq['ltq']/data_rawq['me']

# rdm
data_rawq['rdm'] = data_rawq['xrdq4']/data_rawq['me']

# sp
data_rawq['sp'] = data_rawq['saleq4']/data_rawq['me']

# alm
data_rawq['alm'] = data_rawq['ala'] / \
    (data_rawq['atq']+data_rawq['me']-data_rawq['ceqq'])

# rsup
# data_rawq['saleq_l4'] = data_rawq.groupby(['permno'])['saleq'].shift(4)
data_rawq['rsup'] = (data_rawq['saleq'] -
                     data_rawq['saleq_l4'])/data_rawq['me']

# sgrvol
data_rawq['sgrvol'] = chars_std(0, 15, data_rawq, 'rsup')

# cashpr
data_rawq['cashpr'] = (
    (data_rawq['me'] + data_rawq['dlttq'] - data_rawq['atq']) / data_rawq['cheq'])

# indmom
df_temp = data_rawq.groupby(['date', 'ffi49'], as_index=False)[
    'mom12m'].mean().rename(columns={'mom12m': 'indmom'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'date', 'ffi49']).reset_index(drop=True)

# Mohanram (2005) score (Quarterly Related)
df_temp = data_rawq.groupby(['fyearq', 'fqtr', 'ffi49'], as_index=False)[
    'roavol'].median().rename(columns={'roavol': 'md_roavol'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'fyearq', 'fqtr', 'ffi49']).reset_index(drop=True)

df_temp = data_rawq.groupby(['fyearq', 'fqtr', 'ffi49'], as_index=False)[
    'sgrvol'].median().rename(columns={'sgrvol': 'md_sgrvol'})
data_rawq = pd.merge(data_rawq, df_temp, how='left', on=[
                     'fyearq', 'fqtr', 'ffi49']).reset_index(drop=True)

data_rawq['m7'] = np.where(data_rawq['roavol'] < data_rawq['md_roavol'], 1, 0)
data_rawq['m8'] = np.where(data_rawq['sgrvol'] < data_rawq['md_sgrvol'], 1, 0)

# Quarterly Accounting Variables
# replace 'exchcd','shrcd' with 'primaryexch', 'conditionaltype', 'tradingstatusflg', 'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype'
chars_q = data_rawq[['gvkey', 'permno', 'datadate', 'jdate', 'sic', 'primaryexch', 'conditionaltype', 'tradingstatusflg',
                     'sharetype', 'securitytype', 'securitysubtype', 'usincflg', 'issuertype', 'ticker', 'conm', 'comnam', 'prc', 'shrout',
                     'ret', 'retx', 'acc', 'bm', 'cfp',
                     'ep', 'agr', 'ni', 'op', 'cash', 'chcsho', 'rd', 'cashdebt', 'pctacc', 'gma', 'lev',
                     'rdm', 'sgr', 'sp', 'invest', 'rd_sale', 'lgr', 'roa', 'depr', 'egr', 'roe',
                     'chato', 'chpm', 'chtx', 'noa', 'rna', 'pm', 'ato', 'stdcf',
                     'grltnoa', 'ala', 'alm', 'rsup', 'stdacc', 'sgrvol', 'roavol', 'scf', 'cinvest',
                     'mom1m', 'mom6m', 'mom12m', 'mom60m', 'mom36m', 'seas1a', 'me', 'pscore', 'nincr',
                     'cfp_ia', 'bm_ia', 'me_ia', 'chatoia', 'chmom',
                     'turn', 'dolvol', 'cashpr', 'indmom', 'm7', 'm8']]
chars_q.reset_index(drop=True, inplace=True)

with open('chars_a_accounting.feather', 'wb') as f:
    feather.write_feather(chars_a, f)

with open('chars_q_accounting.feather', 'wb') as f:
    feather.write_feather(chars_q, f)

# Close the database connection
conn.close()