#
# eval_for_hist.py
#

"""
Optimize initial conditions and generate trajectories for SEIRD/R-SEIRD model
on the worst-hit county in each of the United States (except those in the training set).
"""

import matplotlib

import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from ceem.opt_criteria import *
from ceem.ceem import CEEM
from ceem import logger
from src.datautils import get_us_county, smooth_and_diff
from src.models import ReactiveTime, SEIRDModel
import os
import click

opj = os.path.join

import logging

import seaborn as sns

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()


@click.command()
@click.option('--const', '-c', type=int, default=-1)
def main(const):

    torch.manual_seed(1)

    # get counties
    path = 'datasets/COVID-19/csse_covid_19_data/csse_covid_19_time_series'
    df_d = pd.read_csv(opj(path,'time_series_covid19_deaths_US.csv'))
    idx = df_d.groupby('Province_State').transform(max)['1/7/21'] == df_d['1/7/21'] # worst hit counties
    keys = df_d[idx][['Province_State', 'Combined_Key']]
    cond = ~keys['Province_State'].str.contains('Guam|Princess|Islands|Rico|Samoa') # drop non-states
    counties = keys.loc[cond]['Combined_Key']
    counties = counties.values.tolist()

    # drop training counties
    counties_drop = ['Middlesex, Massachusetts, US',
                'Fairfield, Connecticut, US',
                'Kings, New York, US',
                'Los Angeles, California, US',
                'Miami-Dade, Florida, US',
                'Cook, Illinois, US']  
    for c in counties_drop:
        if c in counties:
            counties.remove(c)

    print(counties)


    # load data for each of the counties
    countydat = []
    dates = []

    minT = 1000

    mus = []

    counties_ = []

    for county_ in counties:
        dates_, dat = get_us_county(county_, start_date='2/22/20') 

        if len(dat) < 100:
            continue

        counties_.append(county_)

        mus.append(dat[-1,1] / dat[-1,0])
        dat = smooth_and_diff(dat)
        countydat.append(dat)
        dates.append(dates_)

        minT = min(minT, dat.shape[0])

    print(counties_)
    counties = counties_

    countydat = [d[:minT,:] for d in countydat]
    dates = [d[:minT] for d in dates]

    # load system
    if const < 0:
        system = ReactiveTime(betaE=0.3, mu=0.1)
        system.load_state_dict(torch.load('./trained_models/reactive.th'))
    else:
        system = SEIRDModel(betaE=0.3, mu=0.1, method='rk4')
        system.load_state_dict(torch.load('./trained_models/const.th'))
    
        

    # create t, y
    countydat = np.stack(countydat)

    B = countydat.shape[0]
    T = countydat.shape[1]

    t = torch.arange(T, dtype=dtype).unsqueeze(0).expand(B,T).detach()
    first_d = pd.Series([d[0] for d in dates])
    last_d = pd.Series([d[-1] for d in dates])
    minD = pd.Timestamp(2020,3,11)
    maxD = last_d.max()
    d_off = torch.tensor((first_d - minD).dt.days.values).reshape(B,1)
    t = t + d_off

    y = torch.tensor(countydat + 1, dtype=dtype).view(B,T,2).log().detach()

    torch.save(y, './data/comparisons/y_true.th')

    mu = np.mean(mus)


    llim = np.log(10.)
    rng = np.log(1e5) - np.log(10.)
    xsim = torch.rand(1000,B,T,4) * rng + llim 

    tsim = t.repeat(1000, 1)

    # run CEM
    for i in range(10):

        xsim = xsim.reshape(-1,T,4)

        with torch.no_grad():

            for t_ in range(1, T):
                xsim[:,t_:t_+1] = system.step(tsim[:,t_-1:t_], xsim[:, t_-1:t_])

            ysim = system.observe(tsim, xsim).reshape(-1,B,T,2)

        nlls = ((y.unsqueeze(0)[...,0] - ysim[...,0])**2).sum(dim=-1)

        print(nlls.mean())

        inds = nlls.argsort(dim=0)[:100]

        xsim = xsim.reshape(1000,B,T,4)
        xsim = torch.stack([xsim[inds[:,b], b] for b in range(B)], dim=1)

        xsim_ = xsim.clone()

        xsim = xsim.repeat(10,1,1,1)
        xsim += 0.1 * torch.randn_like(xsim)


    ysim = torch.stack([ysim[inds[:,b], b] for b in range(B)], dim=1).transpose(0,1)

    if const > 0:
        torch.save(ysim, 'data/comparisons/const_ysim.th')
    else:
        torch.save(ysim, 'data/comparisons/reactive_ysim.th')



    

if __name__ == '__main__':
    main()