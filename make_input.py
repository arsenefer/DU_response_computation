import numpy as np
import scipy as sp
from apply_rfchain import open_gp300, smap_2_tf, interp
import os
import json

module_set=set(dir())

files_location = "/volatile/home/af274537/Documents/WorkingDir/new_rfchain"
electronics_path = f"{files_location}/electronics"
l_eff_path = f"{files_location}/l_eff_maps_2"

def load_input_params_from_dict(params_RF):
    latitude = (90-(params_RF['latitude'])) * np.pi / 180
    altitude = params_RF['altitude']
    #Input traces info
    duration = params_RF['duration']
    input_sampling_freq = params_RF['input_sampling_freq']
    out_sampling_freq = params_RF['out_sampling_freq']

    N_samples = int(np.round(duration * input_sampling_freq))
    sampling_period = 1/input_sampling_freq
    in_freqs = sp.fft.rfftfreq(N_samples, sampling_period)

    #Output traces info
    out_N_samples = int(np.round(duration * out_sampling_freq))
    out_sampling_period = 1/out_sampling_freq
    out_freqs = sp.fft.rfftfreq(out_N_samples, out_sampling_period)

    #Input noise
    All_lst_hours = np.arange(0, 24, 0.1)
    LST_radians = All_lst_hours * 15 * np.pi / 180

    if "transfer_function" in params_RF and params_RF["transfer_function"] is not None:
        tf_dict = np.load(os.path.join(s_parameters_path, params_RF["transfer_function"]))
        tf = tf_dict['tf'].astype(np.float64)
        base_freqs = tf_dict['freqs'].astype(np.float64)
        tf = interp.interp1d(base_freqs, tf, kind='linear', axis=0, bounds_error=False, fill_value=0.0)(in_freqs)
    else: 
        s_parameters_path = params_RF['s_parameters_path']

        balun1      = np.loadtxt(os.path.join(s_parameters_path, params_RF["balun1_filename"]), comments=['#', '!']).astype(np.float64)
        matchnet_sn = np.loadtxt(os.path.join(s_parameters_path, params_RF["matchnet_sn_filename"]), comments=['#', '!']).astype(np.float64)
        matchnet_ew = np.loadtxt(os.path.join(s_parameters_path, params_RF["matchnet_ew_filename"]), comments=['#', '!']).astype(np.float64)
        matchnet_z  = np.loadtxt(os.path.join(s_parameters_path, params_RF["matchnet_z_filename"]), comments=['#', '!']).astype(np.float64)
        LNA_sn      = np.loadtxt(os.path.join(s_parameters_path, params_RF["LNA_sn_filename"]), comments=['#', '!']).astype(np.float64)
        LNA_ew      = np.loadtxt(os.path.join(s_parameters_path, params_RF["LNA_ew_filename"]), comments=['#', '!']).astype(np.float64)
        LNA_z       = np.loadtxt(os.path.join(s_parameters_path, params_RF["LNA_z_filename"]), comments=['#', '!']).astype(np.float64)
        cable       = np.loadtxt(os.path.join(s_parameters_path, params_RF["cable_filename"]), comments=['#', '!']).astype(np.float64)
        vga         = np.loadtxt(os.path.join(s_parameters_path, params_RF["vga_filename"]), comments=['#', '!']).astype(np.float64)
        balun2      = np.loadtxt(os.path.join(s_parameters_path, params_RF["balun2_filename"]), comments=['#', '!']).astype(np.float64)
        zload_map   = np.loadtxt(os.path.join(s_parameters_path, params_RF["zload_map_filename"]), comments=['#', '!']).astype(np.float64)
        zant_map    = np.loadtxt(os.path.join(s_parameters_path, params_RF["zant_map_filename"]), delimiter=",", comments=['#', '!'], skiprows=1).astype(np.float64)

        list_s_maps_sn = [balun1, matchnet_sn, LNA_sn, cable, vga]
        list_s_maps_ew = [balun1, matchnet_ew, LNA_ew, cable, vga]
        list_s_maps_z = [balun1, matchnet_z, LNA_z, cable, vga]
        is_db = [False, False, True, True, True]

        tf_sn = smap_2_tf(list_s_maps_sn, zload_map, zant_map, in_freqs, is_db=is_db, balun_2_map=balun2, axis=0)
        tf_ew = smap_2_tf(list_s_maps_ew, zload_map, zant_map, in_freqs, is_db=is_db, balun_2_map=balun2, axis=1)
        tf_z = smap_2_tf(list_s_maps_z, zload_map, zant_map, in_freqs, is_db=is_db, balun_2_map=balun2, axis=2)
        tf = np.stack([tf_sn, tf_ew, tf_z])

    t_SN = open_gp300(params_RF["path_to_GP300_SN"])
    t_EW = open_gp300(params_RF["path_to_GP300_EW"])
    t_Z = open_gp300(params_RF["path_to_GP300_Z"])


    return duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
           N_samples, sampling_period, in_freqs, \
           out_N_samples, out_sampling_period, out_freqs, \
           LST_radians, tf, t_SN, t_EW, t_Z

all_objects=set(dir())
__all__ = list(all_objects-module_set-{'module_set'})