# Here is the file to convert efield to voltage
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import uproot 
import scipy as sp

from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)
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
balun1      = np.loadtxt("electronics/balun_in_nut.s2p", comments=['#', '!']).astype(np.float64)
matchnet_sn = np.loadtxt("electronics/MatchingNetworkX.s2p", comments=['#', '!']).astype(np.float64)
matchnet_ew = np.loadtxt("electronics/MatchingNetworkY.s2p", comments=['#', '!']).astype(np.float64)
matchnet_z  = np.loadtxt("electronics/MatchingNetworkZ.s2p", comments=['#', '!']).astype(np.float64)
LNA_sn      = np.loadtxt("electronics/LNA-X.s2p", comments=['#', '!']).astype(np.float64)
LNA_ew      = np.loadtxt("electronics/LNA-Y.s2p", comments=['#', '!']).astype(np.float64)
LNA_z       = np.loadtxt("electronics/LNA-Z.s2p", comments=['#', '!']).astype(np.float64)
cable       = np.loadtxt("electronics/cable+Connector.s2p", comments=['#', '!']).astype(np.float64)
vga         = np.loadtxt("electronics/feb+amfitler+biast.s2p", comments=['#', '!']).astype(np.float64)
balun2      = np.loadtxt("electronics/balun_before_ad.s2p", comments=['#', '!']).astype(np.float64)
zload_map   = np.loadtxt("electronics/S_balun_AD.s1p", comments=['#', '!']).astype(np.float64)
zant_map    = np.loadtxt("electronics/Z_ant_3.2m.csv", delimiter=",", comments=['#', '!'], skiprows=1).astype(np.float64)

list_s_maps_sn = [balun1, matchnet_sn, LNA_sn, cable, vga]
list_s_maps_ew = [balun1, matchnet_ew, LNA_ew, cable, vga]
list_s_maps_z = [balun1, matchnet_z, LNA_z, cable, vga]
is_db = [False, False, True, True, True]

tf_sn = smap_2_tf(list_s_maps_sn, zload_map, zant_map, target_freqs, is_db=is_db, balun_2_map=balun2, axis=0)
tf_ew = smap_2_tf(list_s_maps_ew, zload_map, zant_map, target_freqs, is_db=is_db, balun_2_map=balun2, axis=1)
tf_z = smap_2_tf(list_s_maps_z, zload_map, zant_map, target_freqs, is_db=is_db, balun_2_map=balun2, axis=2)
tf = np.stack([tf_sn, tf_ew, tf_z])


# fig, ax = plt.subplots(2, 1, figsize=(7,7), sharex=True)
# mat_tf = np.load("EXPLORATION/from grandlib/rf_chainRF2/rfc_tf.npy").astype(np.complex128)
# labels = ['North', 'West', 'Z']
# for i in range(3):
#     ax[0].plot(freqs[(freqs>=30*1e6)&(freqs<=250*1e6)]/1e6, np.abs(tf[i]), label=f"Recomputed {labels[i]}")
#     ax[0].plot(freqs[(freqs>=30*1e6)&(freqs<=250*1e6)]/1e6, np.abs(mat_tf[i,(freqs>=30*1e6)&(freqs<=250*1e6)]), label=f"Target {labels[i]}", ls=':')
# for i in range(3):
#     ax[1].plot(freqs[(freqs>=30*1e6)&(freqs<=250*1e6)]/1e6, np.angle(tf[i]), label=f"Recomputed {labels[i]}")
#     ax[1].plot(freqs[(freqs>=30*1e6)&(freqs<=250*1e6)]/1e6, np.angle(mat_tf[i,(freqs>=30*1e6)&(freqs<=250*1e6)]), label=f"Target {labels[i]}", ls=':')
#     # ax.plot(freqs[(freqs>30*1e6)&(freqs<249*1e6)]/1e6, psd_mat[i, (freqs>30*1e6)&(freqs<249*1e6)], label=f"Target trace {labels[i]}", ls=':')
# ax[0].legend()
# ax[0].set_ylabel("|H(f)|")
# ax[0].set_yscale("log")
# ##ax[
# ax[1].set_xlabel("Frequency [MHz]")
# ax[1].set_ylabel("Phase [rad]")
# fig.suptitle(f"Trasnfer function H(f)")
# plt.tight_layout()
# plt.show()
# exit()
# compute_noise_voltage()
### Effective length
leff_map_EW = open_gp300("l_eff_maps/Light_GP300Antenna_EWarm_leff.npz")
leff_map_SN = open_gp300("l_eff_maps/Light_GP300Antenna_SNarm_leff.npz")
leff_map_Z = open_gp300("l_eff_maps/Light_GP300Antenna_Zarm_leff.npz")

for ev_number in range(0, 10):
    event_traces = efield_data['traces'][ev_number].to_numpy().astype(np.float64)
    event_trace_fft = sp.fft.rfft(event_traces)
    antenna_pos = all_antenna_pos[efield_data['du_id'][ev_number]]
    xmax_pos = meta_data['xmax_pos'][ev_number]
    shower_core_pos = meta_data['core_pos'][ev_number]


    theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error
    # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
    l_eff_sn = get_leff(leff_map_SN, theta_du, phi_du)
    l_eff_ew = get_leff(leff_map_EW, theta_du, phi_du)
    l_eff_z = get_leff(leff_map_Z, theta_du, phi_du)
    l_eff = np.stack([l_eff_sn, l_eff_ew, l_eff_z], axis=2)


    full_response = l_eff * tf[None,None,...]
    
    
    print(full_response.shape)
    
    full_response_all_freqs = np.zeros((*full_response.shape[:-1], len(freqs)), dtype=np.complex128)
    full_response_all_freqs[..., in_antenna_band] = full_response
    
    vout, vout_f = efield_2_voltage(event_trace_fft, in_antenna_band, full_response, target_rate=2e9, current_rate=2e9)

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