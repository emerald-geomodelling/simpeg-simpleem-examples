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
from . import base

class DualMomentTEMXYZSystem(base.XYZSystem):
    gate_start_lm=5
    gate_end_lm=11
    gate_start_hm=12
    gate_end_hm=26

    rx_orientation = 'z'
    tx_orientation = 'z'
    
    @classmethod
    def load_gex(cls, gex):
        class GexSystem(cls):
            pass
        GexSystem.gex = gex
        return GexSystem   
    
    @property
    def area(self):
        return self.gex['General']['TxLoopArea']
    
    @property
    def waveform_hm(self):
        return self.gex['General']['WaveformHMPoint']
    
    @property
    def waveform_lm(self):
        return self.gex['General']['WaveformLMPoint']

    @property
    def lm_data(self):
        return -(self.xyz.dbdt_ch1gt.values*self.gex['Channel1']['GateFactor'])[:,self.gate_start_lm:self.gate_end_lm]
    @property
    def hm_data(self):
        return -(self.xyz.dbdt_ch2gt.values*self.gex['Channel2']['GateFactor'])[:,self.gate_start_hm:self.gate_end_hm]
    @property
    def lm_std(self):
        return (self.xyz.dbdt_std_ch1gt.values*self.gex['Channel1']['GateFactor'])[:,self.gate_start_lm:self.gate_end_lm]
    @property
    def hm_std(self):
        return (self.xyz.dbdt_std_ch2gt.values*self.gex['Channel2']['GateFactor'])[:,self.gate_start_hm:self.gate_end_hm]
    
    @property
    def times(self):
        import emeraldprocessing.tem
        lmtimes = emeraldprocessing.tem.getGateTimesFromGEX(self.gex, 'Channel1')[:,0]
        hmtimes = emeraldprocessing.tem.getGateTimesFromGEX(self.gex, 'Channel2')[:,0]
        
        return (np.array(lmtimes[self.gate_start_lm:self.gate_end_lm]),
                np.array(hmtimes[self.gate_start_hm:self.gate_end_hm]))    
    
    def make_waveforms(self):
        time_input_currents_hm = self.waveform_hm[:,0]
        input_currents_hm = self.waveform_hm[:,1]
        time_input_currents_lm = self.waveform_lm[:,0]
        input_currents_lm = self.waveform_lm[:,1]

        waveform_hm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_hm, input_currents_hm)
        waveform_lm = tdem.sources.PiecewiseLinearWaveform(time_input_currents_lm, input_currents_lm)
        return waveform_lm, waveform_hm
    
    def make_system(self, idx, location, times):
        # FIXME: Martin says set z to altitude, not z (subtract topo), original code from seogi doesn't work!
        # Note: location[2] is already == altitude
        receiver_location = (location[0] + self.gex['General']['RxCoilPosition'][0],
                             location[1],
                             location[2] + np.abs(self.gex['General']['RxCoilPosition'][2]))
        waveform_lm, waveform_hm = self.make_waveforms()        

        return [
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[0], self.rx_orientation)],
                location=location,
                waveform=waveform_lm,
                orientation=self.tx_orientation,
                i_sounding=idx),
            tdem.sources.MagDipole(
                [tdem.receivers.PointMagneticFluxTimeDerivative(
                    receiver_location, times[1], self.rx_orientation)],
                location=location,
                waveform=waveform_hm,
                orientation=self.tx_orientation,
                i_sounding=idx)]
    
    def make_data_uncert_array(self):
        dobs = np.hstack((self.lm_data, self.hm_data)).flatten()
        #uncertainties = np.hstack((self.lm_std, self.hm_std)).flatten()
        #uncertainties = uncertainties * dobs + 1e-13
        uncertainties = 0.05*np.abs(dobs) + 1e-13
        #uncertainties = 0.05*np.abs(dobs) + 1e-13
        
        inds_inactive_dobs = np.isnan(dobs)
        dobs[inds_inactive_dobs] = 9999.
        uncertainties[inds_inactive_dobs] = np.Inf        

        return dobs, uncertainties
        
    def make_data(self, survey):
        dobs, uncertainties = self.make_data_uncert_array()
        return SimPEG.data.Data(
            survey,
            dobs=dobs,
            standard_deviation=uncertainties)

    def make_thicknesses(self):
        # HACK FOR NOW
        n_layer = 30
        return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization(n_layer-1, 3, 1.07)
        
        if "dep_top" in self.xyz.layer_params:
            return np.diff(self.xyz.layer_params["dep_top"].values)
        return SimPEG.electromagnetics.utils.em1d_utils.get_vertical_discretization_time(
            np.sort(np.concatenate(inv.times)),
            sigma_background=0.1,
            n_layer=self.n_layer-1
        )

    def make_misfit_weights(self, thicknesses):
        dobs, uncertainties = self.make_data_uncert_array()
        print("UNCERT", uncertainties)
        return 1./uncertainties

    n_cpu=6
    parallel = True
