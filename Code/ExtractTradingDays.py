#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
#fincap

Team GRFN

Extract the full list of trading days.
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

def get_third_friday(year, month):
    c = calendar.Calendar(firstweekday=calendar.SUNDAY)
    
    monthcal = c.monthdatescalendar(year,month)
    third_friday = [day for week in monthcal for day in week if \
                    day.weekday() == calendar.FRIDAY and \
                    day.month == month][2]
    return third_friday


def process_monthly_file(fn): 
    """
    Process monthly trade file in csv.gz format.

    :param fn: Filename.
    :returns: Dataframe with daily measures.
    """
    # Read file
    df = pd.read_csv(rawdir + fn, sep=';',
                     parse_dates = ['DATETIME'])
    out = pd.Series(df.DATETIME.dt.date.unique())
    return out


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
    df = pd.DataFrame(pd.concat(result).reset_index(drop=True))
    df.columns = ['DATE']
    df['DATE'] = pd.to_datetime(df.DATE)
    
    # Third Fridays
    expirations = [(y, m) for y in range(2002, 2019) for m in range(1, 13)]
    
    fridays = [get_third_friday(x[0], x[1]) for x in expirations]
    # Format to match dataset
    expirations = [x[0] * 100 + x[1] for x in expirations]
    dates_df = pd.DataFrame({'EXPIRATION': expirations,
                             'THIRD_FRIDAY': fridays})
    dates_df['THIRD_FRIDAY'] = pd.to_datetime(dates_df['THIRD_FRIDAY'])
    
    # Find last trading day up to third Friday
    merged = pd.merge_asof(dates_df, df, left_on='THIRD_FRIDAY',
                           right_on='DATE')
    # Keep expiration date
    merged = merged[['EXPIRATION', 'DATE']]
    merged.columns = ['EXPIRATION', 'EXPIRATION_DATE']
    
    merged.to_parquet(datadir + 'Expirations.parquet')
    
    print('Finished in ' + str(tm.time()-global_start) + ' seconds.') 