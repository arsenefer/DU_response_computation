# Here is the file to convert efield to voltage abd to generate noise traces
import numpy as np 
import matplotlib.pyplot as plt
import uproot
import scipy as sp
import apply_rfchain as rfc
#from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, compute_noise
#from input_script import *
import json
import os

from grand_psu_lib.utils import filtering as filt
from grand_psu_lib.utils import utils as utils



SMALL_SIZE = 10;MEDIUM_SIZE = 12;BIGGER_SIZE = 14
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE);plt.rc('xtick', labelsize=MEDIUM_SIZE);plt.rc('ytick', labelsize=MEDIUM_SIZE);plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE)


params_file = 'RF_params_new_leffs.json'


with open(params_file, 'r') as f:
    params_RF = json.load(f)


def load_parameters_and_compute_stuff(params_RF):
    latitude = (90-(params_RF['latitude'])) * np.pi / 180

    #Input traces info
    duration = params_RF['duration']
    sampling_freq = params_RF['sampling_freq']
    out_sampling_freq = params_RF['out_sampling_freq']

    N_samples = int(np.round(duration * sampling_freq))
    sampling_period = 1/sampling_freq
    freqs = sp.fft.rfftfreq(N_samples, sampling_period)

    #Output traces info
    out_N_samples = int(np.round(duration * out_sampling_freq))
    out_sampling_period = 1/out_sampling_freq
    out_freqs = sp.fft.rfftfreq(out_N_samples, out_sampling_period)

    #Input noise
    All_lst_hours = np.arange(0, 24, 0.1)
    LST_radians = All_lst_hours * 15 * np.pi / 180

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

    tf_sn = rfc.smap_2_tf(list_s_maps_sn, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=0)
    tf_ew = rfc.smap_2_tf(list_s_maps_ew, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=1)
    tf_z = rfc.smap_2_tf(list_s_maps_z, zload_map, zant_map, out_freqs, is_db=is_db, balun_2_map=balun2, axis=2)
    tf = np.stack([tf_sn, tf_ew, tf_z])

    t_SN = rfc.open_gp300(params_RF["path_to_GP300_SN"])
    t_EW = rfc.open_gp300(params_RF["path_to_GP300_EW"])
    t_Z = rfc.open_gp300(params_RF["path_to_GP300_Z"])

    l_eff = [t_SN, t_EW, t_Z]

    return l_eff, tf, latitude, out_freqs

    #return latitude, duration, sampling_freq, out_sampling_freq, N_samples, sampling_period, freqs, out_N_samples, out_sampling_period, out_freqs, LST_radians, tf, t_SN, t_EW, t_Z


a =  load_parameters_and_compute_stuff(params_RF)

l_eff = a[0]
tf = a[1]
latitude = a[2]
out_freqs = a[3]


#latitude_new, duration_new, sampling_freq_new, out_sampling_freq_new, N_samples_new, sampling_period_new, freqs_new, out_N_samples_new, out_sampling_period_new, out_freqs_new, LST_radians_new, tf_new, t_SN_new, t_EW_new, t_Z_new = load_parameters_and_compute_stuff(params_RF_new)


noise_computer = rfc.compute_noise(1, latitude,
                              [f"EXPLORATION/LFmap/LFmapshort{i}.npy" for i in range(20, 251)],
                              np.arange(20,251)*1e6,
                              out_freqs,
                              tf, leff_x=l_eff[0], leff_y=l_eff[1], leff_z=l_eff[2])

noise_computer.P_nu
noise_computer.noise_rms_traces()


samples, samples_fft = noise_computer.noise_samples(3, 50, micro=False)  # THese are 8192 long, samples at 2GHz
samples_1024 = samples[:, :, ::4][:, :, 0:1024]   # these are 1024 long, sampled at 500 MHz
psd, f = filt.return_psd(samples, params_RF['out_sampling_freq'], freq_out=True)
psd_1024, f_1024 = filt.return_psd(samples_1024, params_RF['out_sampling_freq']/4, freq_out=True)


plt.figure()
plt.clf()
plt.plot(f/1e6, psd.mean(axis=0)[0], label='mean  PSD of 8192bin long traces')
plt.plot(f_1024/1e6, psd_1024.mean(axis=0)[0], label='mean PSD of 1024bin long traces')
plt.plot(noise_computer.target_freqs/1e6, noise_computer.noise_variance[3, 0], label='Theoretical  Galactic noise')
plt.title('X-axis Galactic contribution PSD')
plt.xlabel('Frequency [MHz]')
plt.ylabel('PSD [V^2/Hz] ')
plt.yscale('log')
plt.legend
plt.ylim(1e-17, 1e-13)
plt.xlim(0, 250)
plt.tight_layout()


if False:


    all_antenna_pos, meta_data, efield_data = open_event_root(root_dir)
    for ev_number in range(0, 10):
        event_traces = efield_data['traces'][ev_number].to_numpy().astype(np.float64)

        # event_traces = event_traces[...,500:4096+500]

        event_trace_fft = sp.fft.rfft(event_traces)
        antenna_pos = all_antenna_pos[efield_data['du_id'][ev_number]]
        xmax_pos = meta_data['xmax_pos'][ev_number]
        shower_core_pos = meta_data['core_pos'][ev_number]


        theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error
        # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
        l_eff_sn = get_leff(t_SN, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff_ew = get_leff(t_EW, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff_z = get_leff(t_Z, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff = np.stack([l_eff_sn, l_eff_ew, l_eff_z], axis=2)

        full_response = l_eff * tf[None,None,...]
        
        
            
        vout, vout_f = efield_2_voltage(event_trace_fft, 
                                        full_response, 
                                        current_rate=2e9, target_rate=2e9)


        print(vout.shape,   vout_f.shape)

        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
        ####################################################################################################
