# Here is the file to convert efield to voltage
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
SMALL_SIZE = 10;MEDIUM_SIZE = 12;BIGGER_SIZE = 14
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE);plt.rc('xtick', labelsize=MEDIUM_SIZE);plt.rc('ytick', labelsize=MEDIUM_SIZE);plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE)
import uproot 
import json
import scipy as sp
from apply_rfchain import open_event_root, percieved_theta_phi, get_leff, make_full_response_matrix, efield_2_voltage, voltage_to_adc
from noise import compute_noise
from make_input import load_input_params_from_dict

## Input section
root_dir = "/volatile/home/af274537/Documents/DATA/ROOT_AND_TRACES/GROOT_DS/DC2.1rc4/ZHaireS-NJ/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-NJ_0004/"
# root_dir = f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/only_0_NJ/"
antenna_params_file = 'antenna_configs/RF_params_new_leffs.json'



with open(antenna_params_file, 'r') as f:
    params_RF = json.load(f)

duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
N_samples, sampling_period, freqs, \
out_N_samples, out_sampling_period, out_freqs, \
LST_radians, tf, t_SN, t_EW, t_Z = load_input_params_from_dict(params_RF)

noise_computer = compute_noise(12, latitude, 
                              [f"files/LFmap/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              out_freqs, 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z, duration=duration)

samples, samples_fft = noise_computer.noise_samples(18, 1000)
samples = voltage_to_adc(samples)
all_antenna_pos, meta_data, efield_data = open_event_root(root_dir, start=0, stop=10)
for ev_number in range(0, 10):
    event_traces = efield_data['traces'][ev_number].astype(np.float64)

    # event_traces = event_traces[...,500:4096+500]

    event_trace_fft = sp.fft.rfft(event_traces)
    antenna_pos = all_antenna_pos[efield_data['du_id'][ev_number]]
    xmax_pos = meta_data['xmax_pos'][ev_number]
    shower_core_pos = meta_data['core_pos'][ev_number]
    index = meta_data['event_index'][ev_number]


    # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error
    theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
    full_response_matrix = make_full_response_matrix(t_SN, t_EW, t_Z, 
                                                     theta_du, phi_du, tf, 
                                                     input_sampling_freq=input_sampling_freq, 
                                                     duration=duration)
        
    vout, vout_f = efield_2_voltage(event_trace_fft, 
                                    full_response_matrix, 
                                    current_rate=input_sampling_freq, target_rate=out_sampling_freq)


    print(vout.shape,   vout_f.shape)
    # Clean, grouped plots using subplots
    comp_labels = ['SN', 'EW', 'Z']
    colors = ['C0', 'C1', 'C2']

    # Frequency axis (MHz)
    freq_mhz = out_freqs / 1e6

    # Time axis (µs) and window
    times = np.arange(vout.shape[-1]) * 1e6 / out_sampling_freq
    window = (times > 0.5) & (times < 1.8)

    sample_idx = 0  # choose noise sample

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # Frequency-domain |V(f)|
    ax = axs[0]
    for i, lab in enumerate(comp_labels):
        ax.plot(freq_mhz, np.abs(vout_f[2, i]), label=lab, color=colors[i])
    ax.set_title('Frequency-domain amplitude')
    ax.set_xlabel('Frequency [MHz]')
    ax.set_ylabel('|V(f)| [a.u.]')
    ax.set_yscale('log')
    ax.legend(title='Component')
    ax.grid(True, alpha=0.3)

    # Time-domain voltage
    ax = axs[1]
    for i, lab in enumerate(comp_labels):
        ax.plot(times[window], vout[2, i, window], label=lab, color=colors[i])
    ax.set_title('Time-domain voltage (windowed)')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Voltage [µV]')
    ax.legend(title='Component')
    ax.grid(True, alpha=0.3)

    # Time-domain voltage + noise
    ax = axs[2]
    for i, lab in enumerate(comp_labels):
        ax.plot(times[window], vout[2, i, window] + samples[sample_idx, i, window],
                label=f'{lab} + noise', color=colors[i])
    ax.set_title('Time-domain voltage + noise (windowed)')
    ax.set_xlabel('Time [µs]')
    ax.set_ylabel('Voltage [µV]')
    ax.legend(title='Component')
    ax.grid(True, alpha=0.3)

    plt.show()
    continue
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################
    ####################################################################################################


    #The code ends here, after are only plots

    ant_n = 2



    with uproot.open("/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/sim_Xiaodushan_20221026_030000_RUN0_CD_ZHAireS-NJ_0000/" + "voltage_13020-23098_L0_0000.root") as f:
        mat_trace = f["tvoltage"]['trace'].array()[ev_number].to_numpy().astype(np.float64)
        # mat_trace = mat_trace[...,500:4096+500]
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