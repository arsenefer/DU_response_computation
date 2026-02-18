# RFchain_computation

Converts simulated electric-field (E-field) traces from cosmic-ray air-shower simulations into realistic voltage traces as seen after the full RF chain of the GRAND detector, including galactic noise.

## Overview

The pipeline takes ROOT files containing per-antenna E-field time series and:

1. **Loads the antenna and electronics description** from a JSON configuration file (S-parameters of each RF component, antenna impedance, effective length maps).
2. **Builds a transfer function** for each polarisation arm (SN, EW, Z) by cascading the ABCD matrices of every RF component (balun → matching network → LNA → cable → VGA → ADC balun).
3. **Applies the full instrument response** (effective length × transfer function) to the E-field FFT to obtain voltage traces.
4. **Adds realistic galactic noise** drawn from LFmap sky temperature maps convolved with the antenna effective area, as a function of Local Sidereal Time (LST).

## Repository structure

```
.
├── apply_rfchain.py          # Core RF-chain math (S→ABCD, TF, Leff, E→V conversion)
├── noise.py                  # Galactic-noise computation (sky temperature → PSD → time-domain samples)
├── make_input.py             # Reads the JSON config, loads S-params & Leff maps, returns all needed objects
├── make_DB.py                # Batch script: loops over ROOT dirs, converts E→V, saves HDF5
├── run_all.py                # Single-run example with comparison plots
├── antenna_configs/          # JSON configuration files for the RF chain
│   └── RF_params_new_leffs.json
├── files/
│   ├── electronics2/         # S-parameter files (.s2p / .s1p / .csv)
│   ├── l_eff_maps_2/         # Effective-length maps (.npz, GP300 format)
│   └── LFmap/                # LFmap sky-temperature maps (LFmapshort{freq_MHz}.npy)
├── demo_efield_to_voltage.ipynb  # Step-by-step demo notebook (see below)
└── README.md
```

## Key modules

### `apply_rfchain.py`

| Function | Description |
|---|---|
| `open_event_root(dir, ...)` | Opens shower / E-field ROOT files and returns antenna positions, shower metadata, and E-field traces. |
| `percieved_theta_phi(ant_pos, xmax_pos)` | Computes the perceived (θ, φ) direction from each antenna to Xmax. |
| `get_leff(DataTable, θ, φ, ...)` | Interpolates the antenna effective length and returns the 3-component complex vector in Cartesian coordinates. |
| `smap_2_tf(list_s_maps, zload, zant, freqs, ...)` | Cascades S-parameter files through ABCD matrices to produce the scalar transfer function. |
| `make_full_response_matrix(t_SN, t_EW, t_Z, θ, φ, tf, ...)` | Combines effective lengths and TF into a single `(n_ant, 3_Efield, 3_channels, n_freq)` response tensor. |
| `efield_2_voltage(E_fft, response, ...)` | Multiplies E-field FFT by the response matrix (Einstein summation) and returns time-domain voltage. |
| `voltage_to_adc(voltage)` | Quantises voltages into 14-bit ADC counts. |

### `noise.py`

| Class / Function | Description |
|---|---|
| `compute_noise(lst_res, lat, temp_files, LF_freqs, target_freqs, tf, leff_*, duration)` | Initialises the noise engine: loads sky maps, interpolates Leff and TF onto the LFmap frequency grid. |
| `.noise_power()` | Integrates $P_\nu = \frac{1}{2} A_\text{eff} \cdot B_\nu \, \sin\theta \, d\theta \, d\phi$ over the visible sky for every LST and frequency. |
| `.Voc_psd()` / `.Vout_psd()` | Converts power to open-circuit and post-RF-chain voltage PSD. |
| `.noise_samples(lst_hour, n, seed)` | Draws Gaussian realisations in the frequency domain and returns time-domain noise traces `(n, 3, N_samples)`. |

### `make_input.py`

| Function | Description |
|---|---|
| `load_input_params_from_dict(params_RF)` | One-stop loader: reads the JSON dict, computes TF and loads Leff maps. Returns `(duration, lat, alt, freqs, tf, t_SN, t_EW, t_Z, ...)`. |
| `open_gp300(path)` / `open_horizon(path)` | Load effective-length data from `.npz` (GP300) or `.npy` (Horizon) format into a `DataTable`. |

## Configuration file format

The JSON file (e.g. `antenna_configs/RF_params_new_leffs.json`) contains:

| Key | Example | Description |
|---|---|---|
| `latitude` | `40.96` | Detector latitude in degrees (geographic). |
| `altitude` | `1264` | Detector altitude in metres. |
| `duration` | `2.048e-6` | Trace duration in seconds. |
| `input_sampling_freq` | `2e9` | Sampling frequency of the input E-field traces (Hz). |
| `out_sampling_freq` | `5e8` | Target output sampling frequency (Hz). |
| `s_parameters_path` | `"./files/electronics2"` | Directory with all `.s2p` / `.s1p` / `.csv` files. |
| `balun1_filename`, `matchnet_*`, `LNA_*`, `cable_filename`, `vga_filename`, `balun2_filename` | filenames | S-parameter files for each RF component. |
| `zload_map_filename` | `"S_balun_AD.s1p"` | Load impedance S-parameter file. |
| `zant_map_filename` | `"Z_ant_3.2m.csv"` | Antenna impedance file. |
| `path_to_GP300_SN`, `path_to_GP300_EW`, `path_to_GP300_Z` | paths | Effective-length maps for each arm. |

## Quick start

```python
import json, numpy as np, scipy as sp
from make_input import load_input_params_from_dict
from apply_rfchain import open_event_root, percieved_theta_phi, make_full_response_matrix, efield_2_voltage
from noise import compute_noise

# 1. Load configuration
with open('antenna_configs/RF_params_new_leffs.json') as f:
    params_RF = json.load(f)

duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
    N_samples, sampling_period, freqs, \
    out_N_samples, out_sampling_period, out_freqs, \
    LST_radians, tf, t_SN, t_EW, t_Z = load_input_params_from_dict(params_RF)

# 2. Open an E-field ROOT file
root_dir = "/path/to/ROOT/directory"
antenna_pos, meta_data, efield_data = open_event_root(root_dir)

# 3. Convert E-field → voltage for one event
ev = 0
traces_fft = sp.fft.rfft(efield_data['traces'][ev].astype(np.float64))
theta, phi  = percieved_theta_phi(
    antenna_pos[efield_data['du_id'][ev]], meta_data['xmax_pos'][ev])
response = make_full_response_matrix(t_SN, t_EW, t_Z, theta, phi, tf,
                                     duration=duration, input_sampling_freq=input_sampling_freq)
vout, vout_f = efield_2_voltage(traces_fft, response, current_rate=input_sampling_freq,
                                target_rate=input_sampling_freq)

# 4. Add galactic noise
noise_computer = compute_noise(
    10., latitude,
    [f"files/LFmap/LFmapshort{i}.npy" for i in range(20, 251)],
    np.arange(20, 251) * 1e6, out_freqs, tf,
    leff_x=t_SN, leff_y=t_EW, leff_z=t_Z, duration=duration)
noise_traces, _ = noise_computer.noise_samples(lst_hour=6., n_samples=vout.shape[0])
vout_noisy = vout * 1e6 + noise_traces  # vout in µV after scaling
```

See `demo_efield_to_voltage.ipynb` for a fully worked example with plots.

## Dependencies

- Python ≥ 3.9
- numpy, scipy, matplotlib
- uproot (ROOT file I/O)
- h5py (HDF5 output, used in `make_DB.py`)
- pandas (metadata bookkeeping in `make_DB.py`)

## License

Internal — GRAND Collaboration.
