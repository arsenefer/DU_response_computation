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
import pandas as pd

from apply_rfchain import open_event_root, percieved_theta_phi, get_leff, smap_2_tf, efield_2_voltage, voltage_to_adc, make_full_response_matrix
from noise import compute_noise

from make_input import load_input_params_from_dict, open_gp300
import matplotlib.pyplot as plt
all_root_dirs = sorted(glob(f"/volatile/home/af274537/Documents/Data/GROOT_DS/DC2.1rc4/ZHaireS-NJ/sim_Xiaodushan_*", ))
# all_root_dirs = sorted(glob(f"/sps/grand/DC2.1rc4/GP300ZHAireS-NJ/sim_Xiaodushan_*", ))

output_dir_base = "/volatile/home/af274537/Documents/Data/GNN_forICRC/hdf5data_Nleff_dummy_1rc4_bollo_testmeta/"
# output_dir_base = "/sps/grand/aferrier/DC2_dummy/"

params_file = 'antenna_configs/RF_params_new_leffs.json'


with open(params_file, 'r') as f:
    params_RF = json.load(f)

duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
N_samples, sampling_period, freqs, \
out_N_samples, out_sampling_period, out_freqs, \
LST_radians, tf, t_SN, t_EW, t_Z = load_input_params_from_dict(params_RF)
print(out_freqs.max(), np.fft.rfftfreq(1024, 1/500e6).max())
noise_computer = compute_noise(10., latitude, 
                              [f"files/LFmap/LFmapshort{i}.npy" for i in range(20, 251)], 
                              np.arange(20,251)*1e6, 
                              np.fft.rfftfreq(1024, 1/500e6), 
                              tf, leff_x=t_SN, leff_y=t_EW, leff_z=t_Z)



big_df = pd.DataFrame({})
for root_dir in all_root_dirs:
    root_dir_name = root_dir.rstrip('/').split('/')[-1]
    output_dir = output_dir_base + root_dir_name
    os.makedirs(output_dir, exist_ok=True)
    file_Vout = []
    step = 200
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
            if meta_data['energy_primary'][ev_idx]*1e-9 <1:
                continue
            event_traces = efield_data['traces'][ev_idx].astype(np.float64)

            event_trace_fft = sp.fft.rfft(event_traces)
            antenna_pos = all_antenna_pos[efield_data['du_id'][ev_idx]]
            xmax_pos = meta_data['xmax_pos'][ev_idx]
            shower_core_pos = meta_data['core_pos'][ev_idx]
            index = meta_data['event_index'][ev_idx]

            # theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos+np.array([0,0,1264])) #To reproduce error


            theta_du, phi_du = percieved_theta_phi(antenna_pos, xmax_pos)
            full_response_matrix = make_full_response_matrix(t_SN, t_EW, t_Z, theta_du, phi_du, tf, input_sampling_freq=input_sampling_freq, duration=duration)

            vout, vout_f = efield_2_voltage(event_trace_fft, 
                                            full_response_matrix, 
                                            current_rate=2e9, target_rate=2e9)

            # vout = voltage_to_adc(vout)
            efield_file_name = meta_data['files'][ev_idx].rstrip('/').split('/')[-1]


            
            noise,_ = noise_computer.noise_samples(18, len(vout_down))
            fig, ax = plt.subplots(3, 1, figsize=(10, 15))

            os.makedirs(f"{output_dir}/{efield_file_name}", exist_ok=True)
            with h5py.File(f"{output_dir}/{efield_file_name}/{index}.hdf5", "w") as f:
                dset = f.create_dataset("v_out_L0", vout.shape, dtype=np.float16)
                dset[:] = vout

                # print(f"vout shape: {vout.shape}")
                # print(f"vout_down shape: {vout_down.shape}")
                vout_down = vout[...,::4]  #Downsampling to 500MHz
                dset = f.create_dataset("v_out_L1", vout_down.shape, dtype=np.float16)
                dset[:] = vout_down

                du_s = efield_data['du_s'][ev_idx]
                dset = f.create_dataset("du_s", len(du_s), dtype=du_s.dtype)
                dset[:] = efield_data['du_s'][ev_idx]

                du_ns = efield_data['du_ns'][ev_idx]
                dset = f.create_dataset("du_ns", len(du_ns), dtype=du_ns.dtype)
                dset[:] = efield_data['du_ns'][ev_idx]

                du_id = efield_data['du_id'][ev_idx]
                dset = f.create_dataset("du_id", len(du_id), dtype=du_id.dtype)
                dset[:] = efield_data['du_id'][ev_idx]

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

            big_df = pd.concat([big_df, pd.DataFrame({
                'root_dir_name': [root_dir_name],
                'root_file_name': [efield_file_name],
                'event_idx': [index],
                'event_number': [meta_data['event_numbers'][ev_idx]],
                'core_pos_x': [meta_data['core_pos'][ev_idx][0]],
                'core_pos_y': [meta_data['core_pos'][ev_idx][1]],
                'core_pos_z': [meta_data['core_pos'][ev_idx][2]],
                'xmax_pos_x': [meta_data['xmax_pos'][ev_idx][0]],
                'xmax_pos_y': [meta_data['xmax_pos'][ev_idx][1]],
                'xmax_pos_z': [meta_data['xmax_pos'][ev_idx][2]],
                'xmax_grams': [meta_data['xmax_grams'][ev_idx]],
                'energy_primary': [meta_data['energy_primary'][ev_idx]],
                'p_types': [meta_data['p_types'][ev_idx]],
                'zenith': [meta_data['zenith'][ev_idx]],
                'azimuth': [meta_data['azimuth'][ev_idx]]
            })], ignore_index=True)
# big_df.to_csv(f"{output_dir_base}/metadata.csv", index=False)



## Faire CSV avec 