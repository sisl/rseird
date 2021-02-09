#
# models.py
#

import torch
from torch import nn
import numpy as np
from ceem.dynamics import C2DSystem, ObsJacMixin, DynJacMixin

class SEIRDModel(C2DSystem, nn.Module, ObsJacMixin, DynJacMixin):
    """
    SEIRD system
    """
    
    def __init__(self, betaE, mu, method='midpoint',
        Activation=nn.Tanh):
        
        C2DSystem.__init__(self, dt=1.0, # integration time-step
                                 method=method # integration scheme
                          )
        nn.Module.__init__(self)

        self.logbetaE = nn.Parameter(torch.tensor(betaE).log())
        self.gamma = 1./5
        self.lam = 1./21
        self.logmu = nn.Parameter(torch.tensor(mu).log())
        
        self._xdim = 4 # log [S, E, I, RD]
        self._ydim = 2

        
    def step_derivs(self, t, x, u=None):
        
        betaE = self.logbetaE.exp()
        gamma = self.gamma

        x = x.exp()
        
        S = x[...,0:1]
        E = x[...,1:2]
        I = x[...,2:3]
        RD = x[...,3:4]
        
        N = S+E+I+RD
       
        
        dSdt = -betaE*S*E/N
        dEdt = -dSdt - gamma*E
        dIdt = gamma*E - self.lam*I
        dRDdt = self.lam * I
        
        dxdt_ = torch.cat([dSdt, dEdt, dIdt, dRDdt], dim=-1)
        dxdt = dxdt_ / x
        
        return dxdt
    
    def observe(self, t, x, u=None):

        betaE = self.logbetaE.exp()
        gamma = self.gamma
        mu = self.logmu.exp()
        lamD = mu * self.lam
        x = x.exp()
        
        S = x[...,0:1]
        E = x[...,1:2]
        I = x[...,2:3]
        RD = x[...,3:4]
        
        N = S+E+I+RD

        EtoI = gamma*E
        ItoD = lamD * I
        
        return (torch.cat([EtoI, ItoD], dim=-1)+1).log()



class ReactiveTime(C2DSystem, nn.Module, ObsJacMixin, DynJacMixin):
    """
    R-SEIRD system
    """
    
    def __init__(self, betaE, mu, method='midpoint',
        Activation=nn.Tanh):
        
        C2DSystem.__init__(self, dt=1.0, # integration time-step
                                 method=method # integration scheme
                          )
        nn.Module.__init__(self)

        self.logbetaE = nn.Parameter(torch.tensor(betaE).log())
        self.gamma = 1./5
        self.lam = 1./21
        self.logmu = nn.Parameter(torch.tensor(mu).log())
        
        self._mlp = LNMLP(2, [32]*3, 1, Activation=Activation)
        
        
        self._xdim = 4 # log [S, E, I, RD]
        self._ydim = 2

        
    def step_derivs(self, t, x, u=None):
        
        betaE = self.logbetaE.exp()
        gamma = self.gamma

        x = x.exp()
        
        S = x[...,0:1]
        E = x[...,1:2]
        I = x[...,2:3]
        RD = x[...,3:4]
        
        N = S+E+I+RD

        mlpinp = torch.cat([t.reshape(*I.shape), I/N],dim=-1)
        betaE_bar = betaE * self._mlp(mlpinp).sigmoid()
        
        dSdt = -betaE_bar*S*E/N
        dEdt = -dSdt - gamma*E
        dIdt = gamma*E - self.lam*I
        dRDdt = self.lam * I
        
        dxdt_ = torch.cat([dSdt, dEdt, dIdt, dRDdt], dim=-1)
        dxdt = dxdt_ / (x+1e-5) # handle instability
        
        return dxdt
    
    def observe(self, t, x, u=None):

        betaE = self.logbetaE.exp()
        gamma = self.gamma
        mu = self.logmu.exp()
        lamD = mu * self.lam
        x = x.exp()
        
        S = x[...,0:1]
        E = x[...,1:2]
        I = x[...,2:3]
        RD = x[...,3:4]
        
        N = S+E+I+RD

        EtoI = gamma*E
        ItoD = lamD * I
        
        return (torch.cat([EtoI, ItoD], dim=-1)+1).log()



class LNMLP(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size, gain=1.0,
                 ln=False, Activation = nn.Tanh):
        super().__init__()
        if len(hidden_sizes) > 0:
            layers = [nn.Linear(input_size, hidden_sizes[0])]
            layers.append(Activation())
            if ln:
                layers.append(nn.LayerNorm(hidden_sizes[0]))
            for i in range(len(hidden_sizes) - 1):
                layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                layers.append(Activation())
                if ln:
                    layers.append(nn.LayerNorm(hidden_sizes[i + 1]))

            layers.append(nn.Linear(hidden_sizes[-1], output_size))
        else:
            layers = [nn.Linear(input_size, output_size)]
        self._layers = layers
        self._mlp = nn.Sequential(*layers)
        self.reset_params(gain=gain)

    def forward(self, input_):
        return self._mlp(input_)

    def reset_params(self, gain=1.0):
        self.apply(lambda x: weights_init_mlp(x, gain=gain))


def weights_init_mlp(m, gain=1.0):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init_normc_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0)


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


    