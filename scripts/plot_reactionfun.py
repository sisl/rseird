#
# plot_reactionfun.py
#

"""
Vizualize the mapping between t, I/N and R0 for learned R-SEIRD model.
Produces Figure 4 in the paper.
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
from src.models import ReactiveTime
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
        Figure saved to figs/reactionfun.pdf
    """

    torch.manual_seed(1)

    # instantiate system
    system = ReactiveTime(betaE=0.3, mu=0.1)

    # load system
    system.load_state_dict(torch.load('./trained_models/reactive.th'))



    # setup t, I/N for query
    minD = pd.Timestamp(2020,3,11)
    maxD = pd.Timestamp(2020,9,19)
    query_dates = pd.date_range(minD, maxD) 
    T = len(query_dates)
    xx = np.arange(T)
    yy = np.linspace(0,1,100)
    X,Y = np.meshgrid(xx,yy)
    inp = torch.tensor(np.stack([X.flat, Y.flat],axis=-1))

    # query the learning function
    fun = (system.logbetaE.exp()*system._mlp(inp).sigmoid()).detach().numpy().flatten()

    # prepare df for plotting
    df = pd.DataFrame(dict(time = query_dates[X.flatten()], prevalence = Y.flatten(),
                        R0=fun/system.gamma))

    # viz reaction function
    plt.figure(figsize=(4.5,3))

    sns.heatmap(df.pivot('time', 'prevalence', 'R0'),  cbar_kws={'label': r'$R_0$'},
            linewidths=0.0, rasterized=True)
    plt.xticks([10,20,30,40,50,60,70,80,90],[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.7,0.8,0.9],
            rotation=0)
    plt.yticks((pd.Series([pd.Timestamp(2020,i,1) for i in range(4,10)])-minD).dt.days,
                ['Apr', 'May', 'June', 'July', 'Aug', 'Sep'])
    plt.xlabel(r'Prevalence ($I_t / N_t$)')
    plt.ylabel(None)

    plt.tight_layout()

    plt.savefig('figs/reactionfun.pdf')

    if show > 0:
        plt.show()

if __name__ == '__main__':
    main()