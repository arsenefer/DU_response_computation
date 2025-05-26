import numpy as np
import scipy as sp
from apply_rfchain import open_gp300, smap_2_tf
module_set=set(dir())

files_location = "/volatile/home/af274537/Documents/WorkingDir/new_rfchain"
electronics_path = f"{files_location}/electronics"
l_eff_path = f"{files_location}/l_eff_maps_2"

latitude = (90-(40.965682)) * np.pi / 180 # Latitude of the site in radians

def get_data(electronics_path=electronics_path, l_eff_path=l_eff_path, duration_mus=4.096, input_sampling_freq=2e9, out_sampling_freq=2e9):

    #Input traces info
    duration = duration_mus*1e-6

    N_samples = int(np.round(duration * input_sampling_freq))
    sampling_period = 1/input_sampling_freq
    freqs = sp.fft.rfftfreq(N_samples, sampling_period)

    #Output traces info
    out_N_samples = int(np.round(duration * out_sampling_freq))
    out_sampling_period = 1/out_sampling_freq
    out_freqs = sp.fft.rfftfreq(out_N_samples, out_sampling_period)

    #Input noise
    All_lst_hours = np.arange(0,24,0.1)
    LST_radians = All_lst_hours * 15 * np.pi / 180

    balun1      = np.loadtxt(f"{electronics_path}/balun_in_nut.s2p", comments=['#', '!']).astype(np.float64)
    matchnet_sn = np.loadtxt(f"{electronics_path}/MatchingNetworkX.s2p", comments=['#', '!']).astype(np.float64)
    # matchnet_sn = np.loadtxt(f"{electronics_path}/XY_matching_network.s2p", comments=['#', '!']).astype(np.float64)
    matchnet_ew = np.loadtxt(f"{electronics_path}/MatchingNetworkY.s2p", comments=['#', '!']).astype(np.float64)
    # matchnet_ew = np.loadtxt(f"{electronics_path}/XY_matching_network.s2p", comments=['#', '!']).astype(np.float64)
    matchnet_z  = np.loadtxt(f"{electronics_path}/MatchingNetworkZ.s2p", comments=['#', '!']).astype(np.float64)
    LNA_sn      = np.loadtxt(f"{electronics_path}/LNA-X.s2p", comments=['#', '!']).astype(np.float64)
    LNA_ew      = np.loadtxt(f"{electronics_path}/LNA-Y.s2p", comments=['#', '!']).astype(np.float64)
    LNA_z       = np.loadtxt(f"{electronics_path}/LNA-Z.s2p", comments=['#', '!']).astype(np.float64)
    cable       = np.loadtxt(f"{electronics_path}/cable+Connector.s2p", comments=['#', '!']).astype(np.float64)
    vga         = np.loadtxt(f"{electronics_path}/feb+amfitler+biast.s2p", comments=['#', '!']).astype(np.float64)
    balun2      = np.loadtxt(f"{electronics_path}/balun_before_ad.s2p", comments=['#', '!']).astype(np.float64)
    zload_map   = np.loadtxt(f"{electronics_path}/S_balun_AD.s1p", comments=['#', '!']).astype(np.float64)
    zant_map    = np.loadtxt(f"{electronics_path}/Z_ant_3.2m.csv", delimiter=",", comments=['#', '!'], skiprows=1).astype(np.float64)

    list_s_maps_sn = [balun1, matchnet_sn, LNA_sn, cable, vga]
    list_s_maps_ew = [balun1, matchnet_ew, LNA_ew, cable, vga]
    list_s_maps_z = [balun1, matchnet_z, LNA_z, cable, vga]
    is_db = [False, False, True, True, True]

    tf_sn = smap_2_tf(list_s_maps_sn, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=0)
    tf_ew = smap_2_tf(list_s_maps_ew, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=1)
    tf_z = smap_2_tf(list_s_maps_z, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=2)
    tf = np.stack([tf_sn, tf_ew, tf_z])


    path_to_GP300_EW = f"{l_eff_path}/Light_GP300Antenna_EWarm_leff.npz"
    path_to_GP300_SN = f"{l_eff_path}/Light_GP300Antenna_SNarm_leff.npz"
    path_to_GP300_Z = f"{l_eff_path}/Light_GP300Antenna_Zarm_leff.npz"
    t_EW = open_gp300(path_to_GP300_EW)
    t_SN = open_gp300(path_to_GP300_SN)
    t_Z = open_gp300(path_to_GP300_Z)
    return duration, input_sampling_freq, out_sampling_freq, \
           N_samples, sampling_period, freqs, \
           out_N_samples, out_sampling_period, out_freqs, \
           LST_radians, tf, t_SN, t_EW, t_Z

duration, sampling_freq, out_sampling_freq, \
    N_samples, sampling_period, freqs, \
    out_N_samples, out_sampling_period, out_freqs, \
    LST_radians, tf, t_SN, t_EW, t_Z = get_data()

all_objects=set(dir())
__all__ = list(all_objects-module_set-{'module_set'})