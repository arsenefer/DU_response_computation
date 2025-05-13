import numpy as np
import scipy as sp
from apply_rfchain import open_gp300, smap_2_tf
module_set=set(dir())

latitude = (90-(42.2281)) * np.pi / 180

#Input traces info
duration = 4.096*1e-6
sampling_freq = 2e9
out_sampling_freq = 2e9

N_samples = int(np.round(duration * sampling_freq))
sampling_period = 1/sampling_freq
freqs = sp.fft.rfftfreq(N_samples, sampling_period)

#Output traces info
out_N_samples = int(np.round(duration * out_sampling_freq))
out_sampling_period = 1/out_sampling_freq
out_freqs = sp.fft.rfftfreq(out_N_samples, out_sampling_period)

#Input noise
All_lst_hours = np.arange(0,24,0.1)
LST_radians = All_lst_hours * 15 * np.pi / 180


balun1      = np.loadtxt("./electronics/balun_in_nut.s2p", comments=['#', '!']).astype(np.float64)
matchnet_sn = np.loadtxt("./electronics/MatchingNetworkX.s2p", comments=['#', '!']).astype(np.float64)
matchnet_ew = np.loadtxt("./electronics/MatchingNetworkY.s2p", comments=['#', '!']).astype(np.float64)
matchnet_z  = np.loadtxt("./electronics/MatchingNetworkZ.s2p", comments=['#', '!']).astype(np.float64)
LNA_sn      = np.loadtxt("./electronics/LNA-X.s2p", comments=['#', '!']).astype(np.float64)
LNA_ew      = np.loadtxt("./electronics/LNA-Y.s2p", comments=['#', '!']).astype(np.float64)
LNA_z       = np.loadtxt("./electronics/LNA-Z.s2p", comments=['#', '!']).astype(np.float64)
cable       = np.loadtxt("./electronics/cable+Connector.s2p", comments=['#', '!']).astype(np.float64)
vga         = np.loadtxt("./electronics/feb+amfitler+biast.s2p", comments=['#', '!']).astype(np.float64)
balun2      = np.loadtxt("./electronics/balun_before_ad.s2p", comments=['#', '!']).astype(np.float64)
zload_map   = np.loadtxt("./electronics/S_balun_AD.s1p", comments=['#', '!']).astype(np.float64)
zant_map    = np.loadtxt("./electronics/Z_ant_3.2m.csv", delimiter=",", comments=['#', '!'], skiprows=1).astype(np.float64)

list_s_maps_sn = [balun1, matchnet_sn, LNA_sn, cable, vga]
list_s_maps_ew = [balun1, matchnet_ew, LNA_ew, cable, vga]
list_s_maps_z = [balun1, matchnet_z, LNA_z, cable, vga]
is_db = [False, False, True, True, True]

tf_sn = smap_2_tf(list_s_maps_sn, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=0)
tf_ew = smap_2_tf(list_s_maps_ew, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=1)
tf_z = smap_2_tf(list_s_maps_z, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=2)
tf = np.stack([tf_sn, tf_ew, tf_z])


path_to_GP300_EW = "./l_eff_maps/Light_GP300Antenna_EWarm_leff.npz"
path_to_GP300_SN = "./l_eff_maps/Light_GP300Antenna_SNarm_leff.npz"
path_to_GP300_Z = "./l_eff_maps/Light_GP300Antenna_Zarm_leff.npz"
t_EW = open_gp300(path_to_GP300_EW)
t_SN = open_gp300(path_to_GP300_SN)
t_Z = open_gp300(path_to_GP300_Z)

all_objects=set(dir())
__all__ = list(all_objects-module_set-{'module_set'})