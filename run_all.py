# Here is the file to convert efield to voltage
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
SMALL_SIZE = 10;MEDIUM_SIZE = 12;BIGGER_SIZE = 14
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE);plt.rc('xtick', labelsize=MEDIUM_SIZE);plt.rc('ytick', labelsize=MEDIUM_SIZE);plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE)
import uproot 
import scipy as sp
from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, compute_noise
from input_script import *

## Input section
root_dir = f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/only_0_NJ/"


noise_computer = compute_noise(1, latitude, 
                              [f"LFmap/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              out_freqs, 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z)

samples_fft, samples = noise_computer.noise_samples(3, 1000)


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


    #The code ends here, after are only plots

    ant_n = 2



    with uproot.open("/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/sim_Xiaodushan_20221026_030000_RUN0_CD_ZHAireS-NJ_0000/" + "voltage_13020-23098_L0_0000.root") as f:
        mat_trace = f["tvoltage"]['trace'].array()[ev_number].to_numpy().astype(np.float64)
        mat_trace = mat_trace[...,500:4096+500]
        mat_fft = sp.fft.rfft(mat_trace, axis=-1)

    times = np.linspace(0,duration*1e6, N_samples)
    window = (times > 0.-1) & (times < 5)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6 ))
    psd = np.abs(vout_f)**2/(N_samples*sampling_freq) * 1e6
    psd_mat = np.abs(mat_fft)**2/(N_samples*sampling_freq) * 1e6
    labels = ['North', 'West', 'Z']
    for i in range(3):
        ax.plot(freqs[(freqs>30*1e6)&(freqs<249*1e6)]/1e6, psd[2, i, (freqs>30*1e6)&(freqs<249*1e6)], label=f"Original trace {labels[i]}")
        ax.plot(freqs[(freqs>30*1e6)&(freqs<249*1e6)]/1e6, psd_mat[2, i, (freqs>30*1e6)&(freqs<249*1e6)], label=f"Target trace {labels[i]}", ls=':')
    ax.legend()
    ax.set_title(f"Comparison PSD - Voltage L0 - ev.:{ev_number}, ant.:{ant_n}")
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("PSD [µV²/MHz]")
    ax.set_yscale("log")
    plt.tight_layout()
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6 ))
    labels = ['North', 'West', 'Z']
    for i in range(0,3):
        # ax.plot(times[window], 100*2*(vout[2, i, window]-mat_trace[2, i, window])/(mat_trace[2, i, window]), label=f"Original trace {labels[i]}")
        ax.plot(times[window], vout[2, i, window], label=f"Remade trace {labels[i]}")
        ax.plot(times[window], mat_trace[2, i, window], label=f"Target trace {labels[i]}", ls=':')
    ax.legend()
    ax.set_title(f"Comparison traces - Voltage L0 - ev.:{ev_number}, ant.:{ant_n}")
    ax.set_xlabel("Time [µs]")
    ax.set_ylabel("Voltage [µV]")
    plt.tight_layout()
    plt.show()