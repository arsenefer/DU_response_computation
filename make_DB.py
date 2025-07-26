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
from glob import glob
import os
import h5py
import json

from apply_rfchain import open_gp300, open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, voltage_to_adc, make_full_response_matrix
from noise import compute_noise

from make_input import load_input_params_from_dict

params_file = 'antenna_configs/RF_params_dummy.json'
with open(params_file, 'r') as f:
    params_RF = json.load(f)

duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
N_samples, sampling_period, freqs, \
out_N_samples, out_sampling_period, out_freqs, \
LST_radians, tf, t_SN, t_EW, t_Z = load_input_params_from_dict(params_RF)
all_root_dirs = sorted(glob(f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2Training/sim_Xiaodushan_*", ))

noise_computer = compute_noise(10., latitude, 
                              [f"files/LFmap/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              out_freqs, 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z)
noise_computer.noise_fourrier_spectrum

output_dir_base = "/volatile/home/af274537/Documents/Data/GNN_forICRC/hdf5data_Nleff_dummy/"
big_list= []
for root_dir in all_root_dirs:
    output_dir = output_dir_base + root_dir.rstrip('/').split('/')[-1]
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    file_Vout = []
    step = 20
    existing_files = set(glob(f"{output_dir}/*.hdf5"))
    for upper_bound in np.arange(0, 1000, step)+step:
        start = upper_bound - step
        stop = upper_bound
        for ev_idx in range(start, stop):
            if f"{output_dir}/{ev_idx}.hdf5" in existing_files:
                start = ev_idx + 1
                print(f"Skipping {ev_idx} in {root_dir} as it already exists.")
        if start == stop:
            continue
        print(f"Processing events {start} to {stop} in {root_dir}")
        all_antenna_pos, meta_data, efield_data = open_event_root(root_dir, start=start, stop=stop)
        for ev_idx in range(len(efield_data['traces'])):
            event_traces = efield_data['traces'][ev_idx].to_numpy().astype(np.float64)

            event_trace_fft = sp.fft.rfft(event_traces)
            antenna_pos = all_antenna_pos[efield_data['du_id'][ev_idx]]
            xmax_pos = meta_data['xmax_pos'][ev_idx]
            shower_core_pos = meta_data['core_pos'][ev_idx]


            # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error



            theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
            full_response_matrix = make_full_response_matrix(t_SN, t_EW, t_Z, theta_du, phi_du, tf, input_sampling_freq=input_sampling_freq, duration=duration)

            vout, vout_f = efield_2_voltage(event_trace_fft, 
                                            full_response_matrix, 
                                            current_rate=2e9, target_rate=2e9)

            # vout = voltage_to_adc(vout)
            with h5py.File(f"{output_dir}/{start+ev_idx}.hdf5", "w") as f:
                dset = f.create_dataset("v_out_L0", vout.shape, dtype=np.float16)
                dset[:] = vout

                vout_down = vout[...,::4]  #Downsampling to 500MHz
                print(f"vout shape: {vout.shape}")
                print(f"vout_down shape: {vout_down.shape}")
                dset = f.create_dataset("v_out_L1", vout_down.shape, dtype=np.float16)
                dset[:] = vout_down

                vout_down_alpha = vout[...,500:4096+500]  #Downsampling to keeping 1024 samples
                vout_down_alpha = vout_down_alpha[...,::4]  #Downsampling to 500MHz
                dset = f.create_dataset("v_out_L1_alpha", vout_down_alpha.shape, dtype=np.float16)
                dset[:] = vout_down_alpha

                du_s = efield_data['du_s'][ev_idx].to_numpy()
                dset = f.create_dataset("du_s", len(du_s), dtype=du_s.dtype)
                dset[:] = efield_data['du_s'][ev_idx].to_numpy()

                du_ns = efield_data['du_ns'][ev_idx].to_numpy()
                dset = f.create_dataset("du_ns", len(du_ns), dtype=du_ns.dtype)
                dset[:] = efield_data['du_ns'][ev_idx].to_numpy()

                du_id = efield_data['du_id'][ev_idx].to_numpy()
                dset = f.create_dataset("du_id", len(du_id), dtype=du_id.dtype)
                dset[:] = efield_data['du_id'][ev_idx].to_numpy()

                dset = f.create_dataset("du_pos", antenna_pos.shape, dtype=antenna_pos.dtype)
                dset[:] = antenna_pos

                f.attrs['event_idx'] = ev_idx
                f.attrs['event_number'] = meta_data['event_numbers'][ev_idx]
                f.attrs['shower_core_pos'] = meta_data['core_pos'][ev_idx]
                f.attrs['xmax_pos'] = meta_data['xmax_pos'][ev_idx]
                f.attrs['xmax_grams'] = meta_data['xmax_grams'][ev_idx]
                f.attrs['energy_primary'] = meta_data['energy_primary'][ev_idx]
                f.attrs['p_types'] = str(meta_data['p_types'][ev_idx])
                f.attrs['zenith'] = meta_data['zenith'][ev_idx]
                f.attrs['azimuth'] = meta_data['azimuth'][ev_idx]
            big_list.append([root_dir.split('/')[-1], ev_idx, meta_data['event_numbers'][ev_idx], meta_data['core_pos'][ev_idx],
                                meta_data['xmax_pos'][ev_idx], meta_data['xmax_grams'][ev_idx], 
                                meta_data['energy_primary'][ev_idx], meta_data['p_types'][ev_idx],
                                meta_data['zenith'][ev_idx], meta_data['azimuth'][ev_idx]])
            
import pandas as pd
pd.DataFrame({
    'root_name': [x[0] for x in big_list],
    'event_idx': [x[1] for x in big_list],
    'event_number': [x[2] for x in big_list],
    'core_pos': [x[3] for x in big_list],
    'xmax_pos': [x[4] for x in big_list],
    'xmax_grams': [x[5] for x in big_list],
    'energy_primary': [x[6] for x in big_list],
    'p_types': [x[7] for x in big_list],
    'zenith': [x[8] for x in big_list],
    'azimuth': [x[9] for x in big_list]
}).to_csv(f"{output_dir_base}/metadata.csv", index=False)


## Faire CSV avec 