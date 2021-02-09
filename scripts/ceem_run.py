#
# ceem_run.py
#

"""
Run script for training SEIRD/R-SEIRD models.
"""

import matplotlib
matplotlib.use('Agg')

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
from time import time

opj = os.path.join

import logging

torch.set_default_dtype(torch.float64)
dtype=torch.get_default_dtype()


@click.command()
@click.option('--const', '-c', type=int, default=-1)
def main(const):
    """
    Args:
        const (int): if >= 0, trains SEIRD, else R-SEIRD
    Notes:
        R-SEIRD trained model can be found in 'data/reactive'
        SEIRD trained model can be found in 'data/const'
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

    mus = [] # for tracking death rates, to initialize

    for county_ in counties:
        dates_, dat = get_us_county(county_, start_date='2/22/20') 
        mus.append(dat[-1,1] / dat[-1,0]) # compute death rate
        dat = smooth_and_diff(dat) # moving average filter
        countydat.append(dat)
        dates.append(dates_)

        minT = min(minT, dat.shape[0])

    mu = np.mean(mus)

    # make all trajectories the same length
    countydat = [d[:minT,:] for d in countydat]
    dates = [d[:minT] for d in dates]

    # find largest confirmed cases for intializing x_{1:T}
    C0summax = np.stack(countydat)[:,:,0].sum(axis=1).max()

    # create t, y tensors
    countydat = np.stack(countydat)

    B = countydat.shape[0]
    T = countydat.shape[1]

    t = torch.arange(T, dtype=dtype).unsqueeze(0).expand(B,T).detach()
    first_d = pd.Series([d[0] for d in dates])
    last_d = pd.Series([d[-1] for d in dates])
    minD = first_d.min()
    maxD = last_d.max()
    print(minD, maxD)
    d_off = torch.tensor((first_d - first_d.min()).dt.days.values).reshape(B,1)
    t = t + d_off

    y = torch.tensor(countydat + 1, dtype=dtype).view(B,T,2).log().detach()


    # set up logdir and model
    if const < 0:
        logger.setup('data/reactive')
        # instantiate system
        system = ReactiveTime(betaE=0.3, mu=mu)

    else:
        logger.setup('data/const')
        # instantiate system
        system = SEIRDModel(betaE=0.3, mu=mu)


    # specify criteria for smoothing and learning
    wstd = 0.1
    ystd = 1.0

    Sig_v_inv = torch.tensor([1.0,0.5]) / (ystd ** 2)
    Sig_v_inv = torch.ones(2) / (ystd ** 2)
    Sig_w_inv = torch.ones(4) / (wstd ** 2)

    smoothing_criteria = []
    for b in range(B):
        obscrit = GaussianObservationCriterion(Sig_v_inv, t[b:b+1], y[b:b+1]) # pass in the inverse covariance
        dyncrit = GaussianDynamicsCriterion(Sig_w_inv, t[b:b+1]) 
        smoothing_criteria.append(GroupSOSCriterion([obscrit, dyncrit]))

    # specify solver kwargs
    smooth_solver_kwargs = {'verbose': 0, # to supress NLS printouts
                            'tr_rho': 0.5, # trust region for smoothing
                            'ftol':1e-5, # solve smoothing to coarse tolerance
                            'gtol':1e-5,
                            'xtol':1e-5
                           }

    # specify learning criteria
    # note: learning criteria do not need to be specified separately for each b in B
    dyncrit = GaussianDynamicsCriterion(Sig_w_inv, t) 
    obscrit = GaussianObservationCriterion(Sig_v_inv, t, y)
    learning_criteria = [GroupSOSCriterion([obscrit, dyncrit])]# since the observation objective doesnt depend on parameters
    learning_params = [list(system.parameters())] # the parameters we want optimized
    learning_opts = ['torch_minimize'] # the optimzer
    learner_opt_kwargs = {'method': 'Adam', # optimizer for learning
                          'tr_rho': 0.01, # trust region for learning
                          'lr': 5e-4,
                          'nepochs': 100,
                         }

    lr_sched = lambda k: 5e-4 * 100 / (100 + k) # updated in ecb

    # specify initial guess
    C0 = torch.tensor(countydat[:,:,0],dtype=dtype).view(B,T,1)
    S0 = torch.ones_like(C0) * C0summax * 5 # inital guess of population
    E0 = C0
    I0 = C0
    D0 = torch.tensor(countydat[:,:,1], dtype=dtype).view(B,T,1)
    R0 = 0.5 * (C0 + D0)
    xsm = torch.cat([S0,E0,I0,R0+D0],dim=-1).log().detach()
    tt = torch.arange(xsm.shape[1]).reshape(1,-1).expand(B,T)

    # define callback to save model
    start_time = time()
    def ecb(k):

        logger.logkv('train/epoch', k)
        logger.logkv('train/elapsed_time', time() - start_time)

        # update lr
        learner_opt_kwargs['lr'] = lr_sched(k)

        logger.logkv('train/lr', lr_sched(k))

        torch.save(system.state_dict(), os.path.join(
            logger.get_dir(), 'ckpts', 'best_model.th'))
        torch.save(xsm, os.path.join(
            logger.get_dir(), 'ckpts', 'best_xsm.th'))

        logger.logkv('train/mu', float(system.logmu.exp()))


    ecb(0)

    # instantiate CEEM and train
    ceem = CEEM(smoothing_criteria, learning_criteria, learning_params, learning_opts,
                    [ecb], lambda x: False, parallel=4 if B > 1 else 0)

    ceem.train(xs=xsm, sys=system, nepochs=1000, smooth_solver_kwargs=smooth_solver_kwargs,
               learner_opt_kwargs=learner_opt_kwargs)

    # save xsm for later
    torch.save(xsm, os.path.join(
            logger.get_dir(), 'xsm.th'))

    # save criteria for later
    torch.save(dict(smoothing_criteria=smoothing_criteria, 
                learning_criteria=learning_criteria,
                mu=mu,
                t=t,
                y=y),
                os.path.join(logger.get_dir(), 'criteria.th'))
    

if __name__ == '__main__':
    main()