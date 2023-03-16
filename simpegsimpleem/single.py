import SimPEG.electromagnetics.time_domain as tdem
import numpy as np

from . import base

class SingleMomentTEMXYZSystem(base.XYZSystem):
    area = 340
    i_max = 1
    rx_orientation = 'z'

    #turns = 1
    #rx_area = 1
    #alt = 30
    
    def make_system(self, idx, location, times):
        return [tdem.sources.CircularLoop(
            location = location,
            receiver_list = [
                tdem.receivers.PointMagneticFluxTimeDerivative(
                    location,
                    times,
                    orientation = self.rx_orientation)],
            waveform = tdem.sources.StepOffWaveform(),
            radius = np.sqrt(self.area/np.pi), 
            current = self.i_max, 
            i_sounding = idx)]
