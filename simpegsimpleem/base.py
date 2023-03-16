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


class XYZSystem(object):
    n_layer=30
    start_res=100
    
    parallel = True
    n_cpu=3
    
    def __init__(self, xyz, **kw):
        self.xyz = xyz
        self.options = kw
            
    def __getattribute__(self, name):
        options = object.__getattribute__(self, "options")
        if name in options: return options[name]
        return object.__getattribute__(self, name)
        
    def make_system(self, idx, location, times):
        raise NotImplementedError("You must subclass XYZInversion and override make_system() with your own method!")

    @property
    def times(self):
        return np.array(self.xyz.model_info['gate times for channel 1'])
    
    def make_thicknesses(self):
        if "dep_top" in self.xyz.layer_params:
            return np.diff(self.xyz.layer_params["dep_top"].values)
        return get_vertical_discretization_time(
            self.times,
            sigma_background=0.1,
            n_layer=self.n_layer-1
        )
        
    def make_survey(self):
        times = self.times
        systems = [
            self.make_system(
                idx,
                self.xyz.flightlines.loc[
                    idx, [self.xyz.x_column, self.xyz.y_column, self.xyz.alt_column]
                ].astype(float).values,
                times)
            for idx in range(0, len(self.xyz.flightlines))]
        return tdem.Survey([
            source
            for sources in systems
            for source in sources])

    def n_param(self, thicknesses):
        return (len(thicknesses)+1)*len(self.xyz.flightlines)
    
    def make_simulation(self, survey, thicknesses):
        return tdem.Simulation1DLayeredStitched(
            survey=survey,
            thicknesses=thicknesses,
            sigmaMap=maps.ExpMap(nP=self.n_param(thicknesses)), 
            parallel=self.parallel,
            n_cpu=self.n_cpu)    
    
    def make_data(self, survey):
        return data.Data(
            survey,
            dobs=self.xyz.dbdt_ch1gt.values.flatten()),
#            standard_deviation=self.xyz.dbdt_std_ch1gt.values.flatten())
    
    def make_misfit_weights(self, thicknesses):
        return 1./self.xyz.dbdt_std_ch1gt.values.flatten()
    
    def make_misfit(self, thicknesses):
        survey = self.make_survey()

        dmis = data_misfit.L2DataMisfit(
            simulation=self.make_simulation(survey, thicknesses),
            data=self.make_data(survey))
        dmis.W = self.make_misfit_weights(thicknesses)
        return dmis
    
    alpha_s = 1e-10
    alpha_r = 1.
    alpha_z = 1.
    def make_regularization(self, thicknesses):
        if False:
            hz = np.r_[thicknesses, thicknesses[-1]]
            reg = LaterallyConstrained(
                get_2d_mesh(len(self.xyz.flightlines), hz),
                mapping=maps.IdentityMap(nP=self.n_param(thicknesses)),
                alpha_s = 0.01,
                alpha_r = 1.,
                alpha_z = 1.)
            # reg.get_grad_horizontal(self.xyz.flightlines[["x", "y"]], hz, dim=2, use_cell_weights=True)
            # ps, px, py = 0, 0, 0
            # reg.norms = np.c_[ps, px, py, 0]
            reg.mref = np.log(np.ones(self.n_param(thicknesses)) * 1/self.start_res)
            # reg.mrefInSmooth = False
            return reg
        else:
            coords = self.xyz.flightlines[[self.xyz.x_column, self.xyz.y_column]].astype(float).values
            # FIXME: Triangulation fails if all coords are on a line, as in a typical synthetic case...
            coords[:,1] += np.random.randn(len(coords)) * 5 #1e-10
            tri = Delaunay(coords)
            hz = np.r_[thicknesses, thicknesses[-1]]

            mesh_radial = SimplexMesh(tri.points, tri.simplices)
            mesh_vertical = SimPEG.electromagnetics.utils.em1d_utils.set_mesh_1d(hz)
            mesh_reg = [mesh_radial, mesh_vertical]
            n_param = int(mesh_radial.n_nodes * mesh_vertical.nC)
            reg_map = SimPEG.maps.IdentityMap(nP=n_param)    # Mapping between the model and regularization
            reg = SimPEG.regularization.LaterallyConstrained(
                mesh_reg, mapping=reg_map,
                alpha_s = self.alpha_s,
                alpha_r = self.alpha_r,
                alpha_z = self.alpha_z,
            )
            reg.mref = np.log(np.ones(self.n_param(thicknesses)) * 1/self.start_res)
            return reg
            
    def make_directives(self):
        return [
            directives.BetaEstimate_ByEig(beta0_ratio=10),
            SimPEG.directives.BetaSchedule(coolingFactor=2, coolingRate=1),
            SimPEG.directives.TargetMisfit()

#            directives.SaveOutputEveryIteration(save_txt=False),
            # directives.Update_IRLS(
            #     max_irls_iterations=30,
            #     minGNiter=1,
            #     fix_Jmatrix=True,
            #     f_min_change = 1e-3,
            #     coolingRate=1),
            # directives.UpdatePreconditioner()

        ]

    def make_optimizer(self):
        return optimization.InexactGaussNewton(maxIter = 40, maxIterCG=20)
    
    def make_inversion(self):
        thicknesses = self.make_thicknesses()

        return inversion.BaseInversion(
            inverse_problem.BaseInvProblem(
                self.make_misfit(thicknesses),
                self.make_regularization(thicknesses),
                self.make_optimizer()),
            self.make_directives())

    def make_forward(self):
        return self.make_simulation(self.make_survey(), self.make_thicknesses())
        
    def inverted_model_to_xyz(self, model, thicknesses):
        xyzsparse = libaarhusxyz.XYZ()
        xyzsparse.flightlines = self.xyz.flightlines
        xyzsparse.layer_data["resistivity"] = 1 / np.exp(pd.DataFrame(
            model.reshape((len(self.xyz.flightlines),
                           len(model) // len(self.xyz.flightlines)))))

        dep_top = np.cumsum(np.concatenate(([0], thicknesses)))
        dep_bot = np.concatenate((dep_top[1:], [np.inf]))

        xyzsparse.layer_data["dep_top"] = pd.DataFrame(np.meshgrid(dep_top, self.xyz.flightlines.index)[0])
        xyzsparse.layer_data["dep_bot"] = pd.DataFrame(np.meshgrid(dep_bot, self.xyz.flightlines.index)[0])

        return xyzsparse
    
    def invert(self):
        self.inv = self.make_inversion()
        
        recovered_model = self.inv.run(self.inv.invProb.reg.mref)
        
        thicknesses = self.inv.invProb.dmisfit.simulation.thicknesses
        
        self.sparse = self.inverted_model_to_xyz(recovered_model, thicknesses)
        self.l2 = None
        if hasattr(self.inv.invProb, "l2model"):
            self.l2 = self.inverted_model_to_xyz(self.inv.invProb.l2model, thicknesses)
        
        return self.sparse, self.l2

    def forward(self):
        # self.inv.invProb.dmisfit.simulation
        self.sim = self.make_forward()

        model_cond=np.log(1/self.xyz.resistivity.values)
        resp = self.sim.dpred(model_cond.flatten())

        resp = resp.reshape((len(self.xyz.flightlines), len(resp) // len(self.xyz.flightlines)))

        xyzresp = libaarhusxyz.XYZ()
        xyzresp.flightlines = self.xyz.flightlines
        xyzresp.layer_data = {
            "dbdt_ch1gt": pd.DataFrame(resp)
        }

        # XYZ assumes all receivers have the same times
        xyzresp.model_info["gate times for channel 1"] = list(self.times)

        return xyzresp
