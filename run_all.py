# Here is the file to convert efield to voltage
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
SMALL_SIZE = 10;MEDIUM_SIZE = 12;BIGGER_SIZE = 14
plt.rc('font', size=BIGGER_SIZE);plt.rc('axes', titlesize=BIGGER_SIZE);plt.rc('axes', labelsize=BIGGER_SIZE);plt.rc('xtick', labelsize=MEDIUM_SIZE);plt.rc('ytick', labelsize=MEDIUM_SIZE);plt.rc('legend', fontsize=BIGGER_SIZE);plt.rc('figure', titlesize=BIGGER_SIZE)
import uproot 
import scipy as sp

from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, compute_noise


root_dir = f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/only_0_NJ/"
all_antenna_pos, meta_data, efield_data = open_event_root(root_dir)

duration = 4.096*1e-6
N_samples = efield_data['traces'][0].to_numpy().astype(np.float64).shape[-1]
sampling_period = duration/N_samples
sampling_freq = 1/sampling_period
min_freq, max_freq = 30e6, 250e6 
freqs = sp.fft.rfftfreq(N_samples, sampling_period)
in_antenna_band = (freqs > min_freq) & (freqs < max_freq)
target_freqs = freqs[in_antenna_band]

ant_n = 2

#Loading maps

### S-parameters
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

tf_sn = smap_2_tf(list_s_maps_sn, zload_map, zant_map, freqs, is_db=is_db, balun_2_map=balun2, axis=0)
tf_ew = smap_2_tf(list_s_maps_ew, zload_map, zant_map, freqs, is_db=is_db, balun_2_map=balun2, axis=1)
tf_z = smap_2_tf(list_s_maps_z, zload_map, zant_map, freqs, is_db=is_db, balun_2_map=balun2, axis=2)
tf = np.stack([tf_sn, tf_ew, tf_z])


path_to_GP300_EW = "/volatile/home/af274537/Documents/WorkingDir/grand/data/detector/Light_GP300Antenna_EWarm_leff.npz"
path_to_GP300_SN = "/volatile/home/af274537/Documents/WorkingDir/grand/data/detector/Light_GP300Antenna_SNarm_leff.npz"
path_to_GP300_Z = "/volatile/home/af274537/Documents/WorkingDir/grand/data/detector/Light_GP300Antenna_Zarm_leff.npz"
t_EW = open_gp300(path_to_GP300_EW)
t_SN = open_gp300(path_to_GP300_SN)
t_Z = open_gp300(path_to_GP300_Z)

latitude = (90-(42.2281)) * np.pi / 180
LST_hours = np.arange(0,24,0.1)
LST_radians = LST_hours * 15 * np.pi / 180

noise_computer = compute_noise(LST_radians[::15], latitude, 
                              [f"EXPLORATION/LFmap/LFmapshort{i}.npy" for i in range(20, 251)], np.arange(20,251)*1e6, freqs, tf,
                              leff_x=t_SN, leff_y=t_EW, leff_z=t_Z)

samples_fft, samples = noise_computer.noise_samples(15, 1000)


for ev_number in range(0, 10):
    event_traces = efield_data['traces'][ev_number].to_numpy().astype(np.float64)
    print(event_traces.shape)
    print('Et la la')
    event_trace_fft = sp.fft.rfft(event_traces)
    antenna_pos = all_antenna_pos[efield_data['du_id'][ev_number]]
    xmax_pos = meta_data['xmax_pos'][ev_number]
    shower_core_pos = meta_data['core_pos'][ev_number]


    theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error
    # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
    l_eff_sn = get_leff(t_SN, theta_du, phi_du, N_sample=8192, duration=4.096e-6)
    l_eff_ew = get_leff(t_EW, theta_du, phi_du, N_sample=8192, duration=4.096e-6)
    l_eff_z = get_leff(t_Z, theta_du, phi_du, N_sample=8192, duration=4.096e-6)
    l_eff = np.stack([l_eff_sn, l_eff_ew, l_eff_z], axis=2)
    l_eff_fullband = np.zeros((*l_eff.shape[:-1], 4097), dtype=np.complex128)
    l_eff_fullband[..., in_antenna_band] = l_eff

    full_response_all_freqs = l_eff_fullband * tf[None,None,...]
    
    
    print(full_response_all_freqs.shape)
        
    vout, vout_f = efield_2_voltage(event_trace_fft, in_antenna_band, full_response_all_freqs, target_rate=2e9, current_rate=2e9)



    print(vout.shape)
    print('On est la')
    #The code ends here, after are only plots




    with uproot.open("/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/sim_Xiaodushan_20221026_030000_RUN0_CD_ZHAireS-NJ_0000/" + "voltage_13020-23098_L0_0000.root") as f:
        mat_trace = f["tvoltage"]['trace'].array()[ev_number].to_numpy().astype(np.float64)
        mat_fft = sp.fft.rfft(mat_trace, axis=-1)

    times = np.linspace(0,duration*1e6, N_samples)
    window = (times > 0.60) & (times < 1.5)

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