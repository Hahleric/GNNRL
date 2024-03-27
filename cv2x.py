import numpy as np
import time
import random
import math


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.BS_position = [0, 1000, 2000]
        self.shadow_std = 8
        self.Decorrelation_distance = 50

    def get_path_loss(self, position):
        distance = 0
        if self.BS_position[0] < position < self.BS_position[1]:
            distance = position - self.BS_position[0]
        if self.BS_position[1] < position < self.BS_position[2]:
            distance = position - self.BS_position[1]
        if position > self.BS_position[2]:
            distance = position - self.BS_position[2]
        # 128.1+37.6log10(d)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_path_loss_mbs(self, position):
        distance = 0
        if self.BS_position[0] < position < self.BS_position[1]:
            distance = position - self.BS_position[0]
        if self.BS_position[1] < position < self.BS_position[2]:
            distance = position - self.BS_position[1]
        if position > self.BS_position[2]:
            distance = position - self.BS_position[2]
        # 128.1+37.6log10(d)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(
                (4 * distance) ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
            + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Environ:
    def __init__(self, n_veh, V2I_min, BW, BW_MBS):
        self.V2Ichannels = V2Ichannels()

        self.V2I_Shadowing = np.random.normal(0, 4, n_veh)
        self.V2I_min = V2I_min
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.vehAntGain = 3
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.n_Veh = n_veh
        self.bandwidth = BW  # Bandwidth for SBS
        self.bandwidth_mbs = BW_MBS  # Bandwidth for MBS

    def Compute_Performance_Static(self, veh_dis):
        # For SBS
        V2I_pathloss = np.array([self.V2Ichannels.get_path_loss(d) for d in veh_dis])
        V2I_channels_abs = V2I_pathloss + self.V2I_Shadowing
        V2I_channels_with_fastfading = V2I_channels_abs[:, np.newaxis] - 20 * np.log10(
            np.abs(np.random.normal(0, 1, (len(veh_dis), 1)) +
                   1j * np.random.normal(0, 1, (len(veh_dis), 1))) / math.sqrt(2))

        platoon_V2I_Signal = 10 ** ((30 - V2I_channels_with_fastfading[:, 0] +
                                     self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(platoon_V2I_Signal, self.sig2))
        interplatoon_rate = V2I_Rate * self.bandwidth

        # For MBS
        V2I_pathloss_mbs = np.array([self.V2Ichannels.get_path_loss_mbs(d) for d in veh_dis])
        V2I_channels_abs_mbs = V2I_pathloss_mbs + self.V2I_Shadowing
        V2I_channels_with_fastfading_mbs = V2I_channels_abs_mbs[:, np.newaxis] - 20 * np.log10(
            np.abs(np.random.normal(0, 1, (len(veh_dis), 1)) +
                   1j * np.random.normal(0, 1, (len(veh_dis), 1))) / math.sqrt(2))

        platoon_V2I_Signal_mbs = 10 ** ((30 - V2I_channels_with_fastfading_mbs[:, 0] +
                                         self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate_mbs = np.log2(1 + np.divide(platoon_V2I_Signal_mbs, self.sig2))
        interplatoon_rate_mbs = V2I_Rate_mbs * self.bandwidth_mbs

        return interplatoon_rate, interplatoon_rate_mbs


if __name__ == '__main__':
    n_veh = 69878  # Assuming this is how you get the number of vehicles
    V2I_min = 300  # Minimum required data rate for V2I Communication
    bandwidth = 180000  # Bandwidth per RB for SBS
    bandwidth_mbs = 1000000  # Bandwidth per RB for MBS
    vehicle_dis = np.zeros(n_veh)
    env = Environ(n_veh, V2I_min, bandwidth, bandwidth_mbs)
    v2i_rate, v2i_rate_mbs = env.Compute_Performance_Static(vehicle_dis)
    print(v2i_rate.shape, v2i_rate_mbs.shape)
