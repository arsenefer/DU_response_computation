"""
This script processes electric field data and converts it into voltage traces, incorporating noise and system response. 

1. Load and process S-parameters for various RF chain components.
2. Compute transfer functions for the RF chain.
3. Load effective length maps for antennas.
4. Simulate noise traces based on LST and latitude.
5. Convert electric field data to voltage traces, including noise.

It does that for multiple ROOT files containing electric field data, and outputs the voltage traces for each event as a list.
Each component of the list stands for one root file
Each component is a list of array whose element are the trace for the corresponding event.

Inputs:
- ROOT directories containing electric field data.
- S-parameter files for RF components.
- Effective length maps for antennas.
- Noise maps for LST-based noise computation.
Usage:
- Ensure all required input files (S-parameters, effective length maps, noise maps) are available in the specified paths.
- Adjust input parameters such as sampling frequency, duration, and noise level as needed.
- Run the script to generate voltage traces for the given electric field data.
"""



# Here is the file to convert efield to voltage
import numpy as np 
import uproot 
import scipy as sp

from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, voltage_to_adc, compute_noise

## Input section
all_root_dirs = [f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2RF2Test/only_0_NJ/"]
latitude = (90-(42.2281)) * np.pi / 180

#Input traces info
duration = 4.096*1e-6
sampling_freq = 2e9
out_sampling_freq = 2e9

#Input noise
All_lst_hours = np.arange(0,24,0.1)

###################



N_samples = int(np.round(duration * sampling_freq))
sampling_period = 1/sampling_freq
freqs = sp.fft.rfftfreq(N_samples, sampling_period)

#Output traces info
out_N_samples = int(np.round(duration * out_sampling_freq))
out_sampling_period = 1/out_sampling_freq
out_freqs = sp.fft.rfftfreq(out_N_samples, out_sampling_period)

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

LST_radians = All_lst_hours * 15 * np.pi / 180
noise_computer = compute_noise(10., latitude, 
                              [f"EXPLORATION/LFmap/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              out_freqs, 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z)
noise_computer.noise_rms_traces()
All_vout = []
for root_dir in all_root_dirs:
    all_antenna_pos, meta_data, efield_data = open_event_root(root_dir)
    file_Vout = []
    for ev_number in range(len(efield_data['traces'])):
        event_traces = efield_data['traces'][ev_number].to_numpy().astype(np.float64)

        event_trace_fft = sp.fft.rfft(event_traces)
        antenna_pos = all_antenna_pos[efield_data['du_id'][ev_number]]
        xmax_pos = meta_data['xmax_pos'][ev_number]
        shower_core_pos = meta_data['core_pos'][ev_number]


        # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error
        theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
        l_eff_sn = get_leff(t_SN, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff_ew = get_leff(t_EW, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff_z = get_leff(t_Z, theta_du, phi_du, input_sampling_freq=sampling_freq, duration=duration)
        l_eff = np.stack([l_eff_sn, l_eff_ew, l_eff_z], axis=2)

        full_response = l_eff * tf[None,None,...]
        
        
            
        vout, vout_f = efield_2_voltage(event_trace_fft, 
                                        full_response, 
                                        current_rate=2e9, target_rate=2e9)

        samples, samples_fft = noise_computer.noise_samples(10, len(vout_f))
        file_Vout.append(vout+samples)
        
    All_vout.append(file_Vout)

print(len(All_vout), type(All_vout))
print(len(All_vout[0]), type(All_vout[0]))
print(len(All_vout[0][0]), type(All_vout[0][0]))
print(All_vout[0][0].shape)
print(All_vout[0][0][0].shape)
print(np.std(voltage_to_adc(All_vout[0][0][0]), axis=-1))
import matplotlib.pyplot as plt
for i in range(10):
    plt.figure()
    for J in range(3):
        plt.plot(np.linspace(0,duration, N_samples), voltage_to_adc(All_vout[0][i][2][J]))
plt.show()