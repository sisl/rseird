#
# plot_trajectories_train.py
#

"""
Optimize trajectories on both SEIRD/R-SEIRD on training set counties and plot them.
Produces Figure 1 in the paper.
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
@click.option('--show', '-s', type=int, default=0)
def main(show):
    """
    Args:
        show (int): will show the figure if show > 0
    Notes:
        Figure saved to figs/testcomparecemtrain.pdf
    """


    torch.manual_seed(1)

    # load the data
    counties = ['Middlesex, Massachusetts, US',
                'Fairfield, Connecticut, US',
                'Kings, New York, US',
                'Los Angeles, California, US',
                'Miami-Dade, Florida, US',
                'Cook, Illinois, US']   

    countydat = []
    dates = []

    minT = 1000

    mus = []

    for county_ in counties:
        dates_, dat = get_us_county(county_, start_date='2/22/20') 
        mus.append(dat[-1,1] / dat[-1,0])
        dat = smooth_and_diff(dat)
        countydat.append(dat)
        dates.append(dates_)

        minT = min(minT, dat.shape[0])

    countydat = [d[:minT,:] for d in countydat]
    dates = [d[:minT] for d in dates]


    # load the models
    system_r = ReactiveTime(betaE=0.3, mu=0.1)
    system_r.load_state_dict(torch.load('./trained_models/reactive.th'))

    system_c = SEIRDModel(betaE=0.3, mu=0.1, method='rk4')
    system_c.load_state_dict(torch.load('./trained_models/const.th'))
        

    # prepare t, y
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


    # find ysim
    ysims = []
    for system in [system_r, system_c]:
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

        ysims.append(ysim)

    # plot figure
    plt.figure(figsize=(4.5*3,3*2))

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='C0', linewidth=2),
                    Line2D([0], [0], color='C1', alpha=0.5),
                    Line2D([0], [0], color='C2', alpha=0.5),
                    Line2D([0], [0], color=[0.5,0.5,0.5], linestyle='--')
                    ]

    for i in range(B):
        plt.subplot(2,3,i+1)
        plt.plot(dates[i], ysims[1][i, :, :, 0].T.exp(), color='C2', alpha=0.05)
        plt.plot(dates[i], ysims[0][i, :, :, 0].T.exp(), color='C1', alpha=0.1)
        plt.plot(dates[i], y[i, :, 0].T.exp(), color='C0', linewidth=2)

        uq = np.quantile(ysims[0][i, :, :, 0].exp().detach().numpy(), 0.95, axis=0)
        plt.plot(dates[i], uq, color=[0.5,0.5,0.5], linestyle='--')


        lq = np.quantile(ysims[0][i, :, :, 0].exp().detach().numpy(), 0.05, axis=0)
        plt.plot(dates[i], lq, color=[0.5,0.5,0.5], linestyle='--')

        # 
        uq = np.quantile(ysims[1][i, :, :, 0].exp().detach().numpy(), 0.95, axis=0)
        plt.plot(dates[i], uq, color=[0.5,0.5,0.5], linestyle='--')


        lq = np.quantile(ysims[1][i, :, :, 0].exp().detach().numpy(), 0.05, axis=0)
        plt.plot(dates[i], lq, color=[0.5,0.5,0.5], linestyle='--')

        plt.title(counties[i])
        plt.xticks(rotation=30)

        if i%3==0:
            plt.ylabel('Daily Confirmed Cases')

        if i==1:
            plt.legend(custom_lines, ['Observed, 7-day average', 'Simulated (R-SEIRD)', 
                'Simulated (SEIRD)', 'Sim. 90% IQR'])


    plt.tight_layout()

    plt.savefig(opj('./figs', 'testcomparecemtrain.pdf'))

    if show > 0:
        plt.show()

    

if __name__ == '__main__':
    main()