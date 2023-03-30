#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#fincap

Team GRFN

Compute the daily values for all measures.
"""

import os
from datetime import timedelta, datetime
import multiprocessing as mp
import time as tm
import calendar

import pandas as pd
import numpy as np

datadir = '../Data/'
rawdir = datadir + 'raw-data-from-deutsche-boerse-for-fincap/'



def winsorize_series(s, level):
    s = s.copy()
    bot = s.quantile(level / 2)
    top = s.quantile(1- (level / 2))
    s[s < bot] = bot
    s[s > top] = top
    return s

def compute_realized_spread(df, delta=timedelta(minutes=5),
                            suffix=''):
    """
    Computes the daily volume-weighted average realized spread for
    all trades and for client trades only.

    :param df: Dataframe containing the trade data.
    :param delta: Time offset for computing the realized spread.
    :param suffix: Suffix to append to variable names.
    :returns: Daily volume-weighted average realized spread for
    all trades and for client trades only.
    """
    df= df.copy()
    # For matching, get all the latest values per contract and timestamp.
    prices = df.groupby(['EXPIRATION',
                         'DATETIME'])['MATCH_PRICE'].last().reset_index()
    prices = prices.sort_values(['DATETIME', 'EXPIRATION'])
 
    # Match with future price. Note that in some cases, last trade at
    # tau + delta might be current trade.
    df['DATETIME_DELTA'] = df['DATETIME'] + delta
    df = pd.merge_asof(df, prices, suffixes=('', '_POST'),
                       left_on='DATETIME_DELTA',
                       right_on='DATETIME',
                       by='EXPIRATION',
                       allow_exact_matches=False,
                       direction='backward')
    
    df['SIGN'] = 1
    df.loc[df.BUY_SELL_ID == 'S', 'SIGN'] = -1
    
    # Keep only agressors, i.e. those using market order (or marketable limit
    # order)
    mkt_df = df[df.AGGRESSOR_FLAG == 'Y'].copy()
    mkt_df['REAL_SPREAD' + suffix] = (2 * mkt_df['SIGN'] *
                             (np.log(mkt_df['MATCH_PRICE']) - 
                              np.log(mkt_df['MATCH_PRICE_POST'])))
    
    # Winsorize
    mkt_df['REAL_SPREAD' + suffix] = \
        winsorize_series(mkt_df['REAL_SPREAD' + suffix], 0.01)
            
    # Compute daily value-weighted averages per expiration
    wm = lambda x: np.average(x, weights=df.loc[x.index, 'TRADE_SIZE'])
    rs_df = mkt_df.groupby([mkt_df['DATETIME'].dt.date, 
                            'EXPIRATION']).agg({'REAL_SPREAD' + suffix: wm})
    rs_df['REAL_SPREAD' + suffix + '_QTY'] =\
        mkt_df.groupby([mkt_df['DATETIME'].dt.date, 
                        'EXPIRATION']).agg({'TRADE_SIZE': 'sum'})
    
    # Re-do for clients only
    clients_mkt_df = mkt_df[mkt_df.ACCOUNT_ROLE == 'A']
    clients_rs_df = \
        clients_mkt_df.groupby([clients_mkt_df['DATETIME'].dt.date, 
            'EXPIRATION']).agg({'REAL_SPREAD' + suffix: wm})
    clients_rs_df.columns = ['CLIENTS_REAL_SPREAD' + suffix]
    clients_rs_df['CLIENTS_REAL_SPREAD' + suffix + '_QTY'] = \
        clients_mkt_df.groupby([clients_mkt_df['DATETIME'].dt.date, 
                                'EXPIRATION']).agg({'TRADE_SIZE': 'sum'})
        
    return rs_df, clients_rs_df


def compute_clients_share(df):
    """
    Computes the daily share of client volume in total volume.

    :param df: Dataframe containing the trade data.
    :returns: Daily share of client volume.
    """
    # Client trades
    clients_df = df[df.ACCOUNT_ROLE == 'A']
    
    # Share of client volume in total volume
    shr_client_vol = (clients_df.groupby([clients_df['DATETIME'].dt.date, 
                                         'EXPIRATION'])[['TRADE_SIZE']].sum() /
                      df.groupby([df['DATETIME'].dt.date,
                                  'EXPIRATION'])[['TRADE_SIZE']].sum())
    shr_client_vol.columns = ['SHR_CLIENT_VOL']
    shr_client_vol['TOTAL_QTY'] = \
        df.groupby([df['DATETIME'].dt.date, 
                    'EXPIRATION'])[['TRADE_SIZE']].sum()
    shr_client_vol['CLIENTS_QTY'] = \
        clients_df.groupby([df['DATETIME'].dt.date,
                            'EXPIRATION'])[['TRADE_SIZE']].sum()
    return shr_client_vol


def compute_clients_frac_mo(df):
    """
    Computes the daily fraction of client trades executed via market orders 
    and marketable limit orders.

    :param df: Dataframe containing the trade data.
    :returns: Daily fraction of client trades executed via market orders.
    """
    # Client trades
    clients_df = df[df.ACCOUNT_ROLE == 'A']
    # Client trades using market orders
    clients_mkt_df = clients_df[clients_df.AGGRESSOR_FLAG == 'Y']
    
    # Fraction of client trades executed via market orders and marketable
    # limit orders
    n_clients_mkt = clients_mkt_df.groupby(
        [clients_mkt_df['DATETIME'].dt.date,
         'EXPIRATION'])[['TRADE_SIZE']].count()
    n_clients = clients_df.groupby([clients_df['DATETIME'].dt.date, 
                                    'EXPIRATION'])[['TRADE_SIZE']].count()

    frac_clients_mkt = n_clients_mkt / n_clients
    # If no market orders or marketable limit orders
    frac_clients_mkt.columns = ['FRAC_CLIENTS_MKT']
    frac_clients_mkt['CLIENTS_MKT_QTY'] = \
        clients_mkt_df.groupby([df['DATETIME'].dt.date,
                            'EXPIRATION'])[['TRADE_SIZE']].sum()
    frac_clients_mkt = frac_clients_mkt.fillna(0)
    return frac_clients_mkt


def compute_vr(df, vr_freq_num='5T', vr_freq_denum = '1T', vr_n=5,
               suffix=''):
    """
    Computes the daily variance ratio.

    :param df: Dataframe containing the trade data.
    :param vr_freq_num: Resample frequency for returns in the numerator.
    :param vr_freq_denum: Resample frequency for returns in the denominator.
    :param vr_n: Ratio of vr_freq_num and vr_freq_denum.
    :param suffix: Suffix to append to variable names.
    :returns: Daily variance ratio.
    """['ABS_VR_30T5T',
 'REAL_SPREAD_5T',
 'CLIENTS_REAL_SPREAD_5T',
 'SHR_CLIENT_VOL',
 'FRAC_CLIENTS_MKT',
 'GTR']
    def compute_variance(df, freq):
        """
        Computes the daily variance from trades.

        :param df: Dataframe containing the trade data.
        :param freq: Resample frequency for returns.
        :returns: Daily variance.
        """
        df = df.set_index(['DATETIME'])
        df = df.groupby([df.index.date, 
                         'EXPIRATION']).resample(freq)[['MATCH_PRICE']].last()
        df.index.names = ['DATE', 'EXPIRATION', 'DATETIME']
        df['MATCH_PRICE'] = df.reset_index().groupby(['DATE',
            'EXPIRATION'])['MATCH_PRICE'].fillna(method='ffill').values
        df['RET'] = df.groupby(
            ['DATE','EXPIRATION'])['MATCH_PRICE'].pct_change()
        
        out = df.groupby(level=[0,1])['RET'].var()
        out.name = 'VAR'
        return out
    
    vr_num_df = compute_variance(df, vr_freq_num)
    vr_denum_df = compute_variance(df, vr_freq_denum)
    vr_df = vr_num_df / (vr_n * vr_denum_df)
    
    abs_vr_df = np.abs(vr_df - 1)
    abs_vr_df.name = 'ABS_VR' + suffix
    return pd.DataFrame(abs_vr_df)

def compute_clients_gtr(df):
    """
    Computes the daily volume-weighted average relative gross trading
    revenue for client trades.

    :param df: Dataframe containing the trade data.
    :returns: Daily volume-weighted average relative gross trading
    revenue for client trades.
    """
    df= df.copy()
    df['DATE'] = df['DATETIME'].dt.date
    # For matching, get all the last price per contract and date.
    prices = df.groupby(['EXPIRATION', 'DATE'])['MATCH_PRICE'].last().reset_index()
    prices = prices.sort_values(['DATE', 'EXPIRATION'])
 
    # Match with future price. Note that in some cases, last trade at
    #  might be current trade.
    df = pd.merge(df, prices, suffixes=('', '_POST'),
                       left_on=['DATE', 'EXPIRATION'],
                       right_on=['DATE', 'EXPIRATION'])
    
    df['SIGN'] = 1
    df.loc[df.BUY_SELL_ID == 'S', 'SIGN'] = -1
    
    df['TRADE_VALUE'] = df['TRADE_SIZE'] * df['MATCH_PRICE']
    
    # Keep only agressors, i.e. those using market order (or marketable limit
    # order)
    mkt_df = df[df.AGGRESSOR_FLAG == 'Y'].copy()
    mkt_df['Profit'] = (df['SIGN'] *
                     (df['MATCH_PRICE_POST'] - 
                      df['MATCH_PRICE']))
    
    # For clients only
    clients_mkt_df = mkt_df[mkt_df.ACCOUNT_ROLE == 'A']
    clients_gtr_df = \
        clients_mkt_df.groupby([clients_mkt_df['DATETIME'].dt.date, 
            'EXPIRATION']).agg({'Profit': sum})
    clients_gtr_df.columns = ['GTR']
    clients_gtr_df['GTR_VALUE'] = \
        clients_mkt_df.groupby([clients_mkt_df['DATETIME'].dt.date, 
                                'EXPIRATION']).agg({'TRADE_VALUE': 'sum'})
    clients_gtr_df['GTR'] = (clients_gtr_df['GTR'] /
                             clients_gtr_df['GTR_VALUE'])
        
    return clients_gtr_df

def process_monthly_file(fn): 
    """
    Process monthly trade file in csv.gz format.

    :param fn: Filename.
    :returns: Dataframe with daily measures.
    """
    # Read file
    df = pd.read_csv(rawdir + fn, sep=';',
                     parse_dates = ['DATETIME'])
    df.loc[df['AGGRESSOR_FLAG'].isnull(), 'AGGRESSOR_FLAG'] = ''
    month = fn[:7]
    
    # Do we have sign? (valid aggressor flag)
    # Let's create a dataframe with only signed trades
    # This is empty before Nov. 2009, contains all trades after Nov. 2009,
    # and has signed trades for Nov. 2009.
    sign_df = df[df.DATETIME >= datetime(2009, 11, 16)]
    
    # Variance ratios
    vr_freq_num = '30T'
    vr_freq_denum = '5T'
    vr_n = 6
    abs_vr_df = compute_vr(df, vr_freq_num, vr_freq_denum, vr_n, '_30T5T')
    
    # Realized spreads
    if len(sign_df) > 0:
        delta = timedelta(minutes=5)
        rs_df, clients_rs_df = compute_realized_spread(sign_df, delta,
                                                       suffix='_5T')
    else:
        rs_df = None
        clients_rs_df = None 
    
    # Share of client volume in total volume change
    shr_client_vol = compute_clients_share(df)
    
    # Fraction of client trades executed via market orders and marketable
    # limit orders
    if len(sign_df) > 0:
        frac_clients_mkt = compute_clients_frac_mo(sign_df)
    else:
        frac_clients_mkt = None
    
    # Gross trading revenue
    if len(sign_df) > 0:
        gtr_df = compute_clients_gtr(sign_df)
    else:
        gtr_df = None
        
    # Prepare output
    if len(sign_df) > 0:
        return pd.concat([abs_vr_df, rs_df, shr_client_vol, clients_rs_df,
                          frac_clients_mkt, gtr_df], axis=1).drop_duplicates()
    else:
        return pd.concat([abs_vr_df, shr_client_vol], axis=1).drop_duplicates()


if __name__ == '__main__':
    global_start = tm.time()
    
    # Make list of files to process
    files = os.listdir(rawdir)
    files = [f for f in files if f.endswith('.csv.gz')]
    files.sort()
    
    # Process in parallel using all cores.
    pool = mp.Pool(processes=mp.cpu_count())
    result = pool.map(process_monthly_file, files)
    pool.close()
    pool.join()
    
    # Output results
    df = pd.concat(result)
    df.to_parquet(datadir + 'DailyMeasures_v3.parquet')
    
    print('Finished in ' + str(tm.time()-global_start) + ' seconds.') 