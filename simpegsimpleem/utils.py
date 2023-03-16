import numpy as np
import os
from matplotlib import pyplot as plt
from discretize import TensorMesh

from SimPEG import maps
from SimPEG.electromagnetics import time_domain as tdem
from SimPEG.electromagnetics.utils.em1d_utils import plot_layer
import libaarhusxyz
import pandas as pd

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import os, tarfile
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from discretize import TensorMesh, SimplexMesh
#from pymatsolver import PardisoSolver

from SimPEG.utils import mkvc
from SimPEG import (
    maps, data, data_misfit, inverse_problem, regularization, optimization,
    directives, inversion, utils
    )

from SimPEG.utils import mkvc
import SimPEG.electromagnetics.time_domain as tdem
import SimPEG.electromagnetics.utils.em1d_utils
from SimPEG.electromagnetics.utils.em1d_utils import get_2d_mesh,plot_layer, get_vertical_discretization_time
from SimPEG.regularization import LaterallyConstrained, RegularizationMesh

import scipy.stats


def nadeau_bengio(ascores, bscores, train_size, test_size):
    """See https://github.com/emerald-geomodelling/paper-raw-em-interpretation/issues/3

    [Nadeau and
    Bengio](https://link.springer.com/article/10.1023/A:1024068626366)
    showed how the standard Student's t-test can not be used for MCCV,
    as the test sets are not (unlikely to be) independent. The
    proposed a modified version that accounts for this.

    Here is a python implementation. ascores and bscores are your test
    scores from your cross validation runs (e.g. RMSE or average error
    for each run or similar) for two algorithms (or feature
    selections).
    """
    diff = [a - b for a, b in zip(ascores, bscores)]
    d_bar = np.mean(diff)
    sigma2 = np.var(diff)

    n1 = train_size
    n2 = test_size
    n = len(ascores)
    sigma2_mod = sigma2 * (1/n + n1/n2)
    t_static =  d_bar / np.sqrt(sigma2_mod)
    
    pvalue = (1.0 - scipy.stats.t.cdf(abs(t_static),  n-1)) * 2.0
    
    return {"t": t_static, "p": pvalue}

def make_2layer(xdist, dtb, layers, res_upper=30, res_lower=300, x=None, y=None):
    gxdist, gz = np.meshgrid(xdist, layers)
    gdtb, dummy = np.meshgrid(dtb, layers)
    
    gdtb = gdtb[:-1,:]
    gzt = gz[:-1,:]
    gzb = gz[1:,:]
        
    bedrockoverlap = np.clip((gdtb - gzb) / (gzt - np.fmin(gzb, gzb[np.isfinite(gzb)].max())), 0, 1)
    res = bedrockoverlap * res_lower + (1 - bedrockoverlap) * res_upper
    
    if x is None:
        x = xdist
    if y is None:
        y = 0 * xdist
    
    flightlines = pd.DataFrame({"xdist": xdist, "x": x, "y": y, "interface_depth": dtb}).assign(line_no=0)
    resistivity = pd.DataFrame(res.T)
    
    xyz = libaarhusxyz.XYZ()

    xyz.flightlines = flightlines
    xyz.layer_data["resistivity"] = resistivity
    xyz.layer_data["dep_top"] = pd.DataFrame(gzt.T)
    xyz.layer_data["dep_bot"] = pd.DataFrame(gzb.T)
    
    return xyz

def add_noise(xyz, rel_uncertainty=0.01):
    dpred = xyz.dbdt_ch1gt.values.flatten()
    noise = rel_uncertainty*np.abs(dpred)*np.random.rand(len(dpred))
    xyz.layer_data["dbdt_ch1gt"] += noise.reshape(xyz.dbdt_ch1gt.shape)

def add_uncertainty_normal(xyz, rel_uncertainty):
    xyz.layer_data["dbdt_std_ch1gt"] = np.abs(rel_uncertainty
                                              * xyz.layer_data["dbdt_ch1gt"]
                                              * np.random.randn(*xyz.layer_data["dbdt_ch1gt"].shape))

def add_uncertainty(xyz, rel_uncertainty):
    xyz.layer_data["dbdt_std_ch1gt"] = rel_uncertainty*np.abs(xyz.dbdt_ch1gt)
