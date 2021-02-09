#
# datautils.py
#

import pandas as pd
import numpy as np
import os
opj = os.path.join

from hampel import hampel

def compare_dates(mdy1, mdy2):
    """
    Compares two dates
        mdy1 (str): MM/DD/YY
        mdy2 (str): MM/DD/YY
    Returns:
        equal (bool)
    """

    equal = np.all([int(x) == int(y) for x,y in zip(
        mdy1.split('/'), mdy2.split('/'))])

    return equal


def get_us_county(Combined_Key, dcutoff=1, 
        start_date='2/22/20',
        end_date = '9/27/20'):
    """
    Get US county data form Johns Hopkins dataset.
    Args:
        Combined_Key (str): county key for dataset
        dcutoff (int): cutoff for minimum deaths
        start_date (str): start date in MM/DD/YY
        end_date (str): end date in MM/DD/YY
    Returns:
        dates (pd.Series): dates
        series (np.array): (T,2) [confirmed, deaths] series
    """
    path = 'datasets/COVID-19/csse_covid_19_data/csse_covid_19_time_series'
    df_c = pd.read_csv(opj(path,'time_series_covid19_confirmed_US.csv'))
    df_d = pd.read_csv(opj(path,'time_series_covid19_deaths_US.csv'))

    df_c = df_c.loc[df_c['Combined_Key']==Combined_Key]
    df_d = df_d.loc[df_d['Combined_Key']==Combined_Key]

    dates_c = df_c.columns[12:]
    dates_d = df_d.columns[12:]

    confirmed = df_c.sum(axis=0)[12:].values.astype(np.float64)
    deaths = df_d.sum(axis=0)[12:].values.astype(np.float64)

    if start_date is not None:
        c_s = np.where([compare_dates(d,start_date) for d in dates_c])[0][0]
        d_s = np.where([compare_dates(d,start_date) for d in dates_d])[0][0]
    else:
        c_s = 0
        d_s = 0

    if end_date is not None:
        c_e = np.where([compare_dates(d,end_date) for d in dates_c])[0][0]
        d_e = np.where([compare_dates(d,end_date) for d in dates_d])[0][0]
    else:
        c_e = -1
        d_e = -1

    confirmed = confirmed[c_s:c_e]
    deaths = deaths[d_s:d_e]

    if len(confirmed) < len(deaths):
        deaths = deaths[:-1]

    series = np.stack([confirmed,deaths]).T

    T = len(confirmed)

    dates = pd.date_range(start_date, end_date)[:T]

    cutoff_ind = np.argwhere(deaths >= dcutoff)[0,0]

    series = series[cutoff_ind:]
    dates = dates[cutoff_ind:]

    return dates, series

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def smooth_and_diff(series, ma_days=7):
    """
    Args:
        series (np.array): (T,2) series to smooth
        ma_days (int): number of days for moving avg of diff
    """
    
    hampel_ = lambda x: hampel(pd.Series(np.diff(x)), window_size=5).values
    
    smooth_series = np.stack([moving_average(hampel_(series[:,0]), ma_days),
                           moving_average(hampel_(series[:,1]), ma_days)], axis=1)
    
    
    return smooth_series

