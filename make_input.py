import numpy as np
import scipy as sp
from apply_rfchain import smap_2_tf, interp
import os

from typing import Union
from dataclasses import dataclass
from numbers import Number

module_set=set(dir())

@dataclass
class DataTable:
    """
    DataTable is a class that represents a data structure for storing and managing
    various parameters related to frequency, angles, and effective lengths.

    Attributes:
        frequency (Union[Number, np.ndarray]): The frequency values, which can be a single number or an array.
        theta (Union[Number, np.ndarray]): The theta angle values, which can be a single number or an array.
        phi (Union[Number, np.ndarray]): The phi angle values, which can be a single number or an array.
        leff_theta (Union[Number, np.ndarray], optional): The effective length for the theta component. Defaults to None.
        phase_theta (Union[Number, np.ndarray], optional): The phase for the theta component. Defaults to None.
        leff_phi (Union[Number, np.ndarray], optional): The effective length for the phi component. Defaults to None.
        phase_phi (Union[Number, np.ndarray], optional): The phase for the phi component. Defaults to None.
        leff_phi_reim (Union[Number, np.ndarray], optional): The real and imaginary parts of the effective length for the phi component. Defaults to None.
        leff_theta_reim (Union[Number, np.ndarray], optional): The real and imaginary parts of the effective length for the theta component. Defaults to None.
    """
    frequency: Union[Number, np.ndarray]
    theta: Union[Number, np.ndarray]
    phi: Union[Number, np.ndarray]
    leff_theta: Union[Number, np.ndarray] = None
    phase_theta: Union[Number, np.ndarray] = None
    leff_phi: Union[Number, np.ndarray] = None
    phase_phi: Union[Number, np.ndarray] = None
    leff_phi_reim: Union[Number, np.ndarray] = None
    leff_theta_reim: Union[Number, np.ndarray] = None

def open_horizon(path_to_horizon):
    """
    Load and process horizon data from a given file.

    This function reads horizon data from a file, processes it, and returns
    a DataTable object containing the processed data. The data includes
    frequency, angular information (theta and phi), effective lengths, and
    phases for both theta and phi polarizations.

    Args:
        path_to_horizon (str): Path to the file containing the horizon data.
            The file is expected to be in a format compatible with `numpy.load`.

    Returns:
        DataTable: A table containing the processed horizon data.

    Notes:
        - The function assumes the input file contains specific arrays in a
          predefined order: f, R, X, theta, phi, lefft, leffp, phaset, phasep.
        - Frequency values are converted from MHz to Hz.
        - Phases are converted from degrees to radians for complex calculations.
        - The shape of the data is inferred based on the unique theta and phi
          values and the dimensions of the input arrays.
        - Ensure that the conversion from radians to degrees does not affect
          calculations elsewhere in the code.
    """
    f, R, X, theta, phi, lefft, leffp, phaset, phasep = np.load(
        path_to_horizon, mmap_mode="r")

    n_f = f.shape[0]
    n_theta = len(np.unique(theta[0, :]))
    n_phi = int(R.shape[1] / n_theta)
    shape = (n_f, n_phi, n_theta)

    dtype = "f4"
    f = f[:, 0].astype(dtype) * 1.0e6  # MHz --> Hz
    theta = theta[0, :n_theta].astype(dtype)  # deg
    phi = phi[0, ::n_theta].astype(dtype)  # deg
    lefft = lefft.reshape(shape).astype(dtype)  # m
    leffp = leffp.reshape(shape).astype(dtype)  # m

    phaset = phaset.reshape(shape).astype(dtype)  # deg
    phasep = phasep.reshape(shape).astype(dtype)  # deg
    leffp_reim = leffp*np.exp(1j*phasep/180*np.pi)
    lefft_reim = lefft*np.exp(1j*phaset/180*np.pi)
    t = DataTable(
        frequency=f,
        theta=theta,
        phi=phi,
        leff_theta_reim=lefft_reim,
        leff_phi_reim=leffp_reim,
        leff_theta=lefft,
        phase_theta=phaset,
        leff_phi=leffp,
        phase_phi=phasep,
    )
    return t


def open_gp300(path_to_gp300):
    """
    Load and process GP300 data from a specified file.

    This function reads a `.npz` file containing GP300 data, processes the data to extract
    frequency, theta, phi, effective lengths (leff) in both theta and phi polarizations,
    and their respective phases. The processed data is returned as a `DataTable` object.

    Args:
        path_to_gp300 (str): Path to the `.npz` file containing GP300 data.

    Returns:
        DataTable

    Notes:
        - The input `.npz` file is expected to contain the following keys:
          `freq_mhz`, `leff_theta`, and `leff_phi`.
        - The frequency values in the file are converted from MHz to Hz.
        - The `leff_theta` and `leff_phi` arrays are reshaped and processed to compute
          their magnitudes and phases.
    """
    f_leff = np.load(path_to_gp300)
    f = f_leff["freq_mhz"] * 1e6   # MHz --> Hz
    theta = np.arange(91).astype(float)
    phi = np.arange(361).astype(float)
    # Real + j Imag. shape (phi, theta, freq) (361, 91, 221)
    lefft_reim = f_leff["leff_theta"]
    # Real + j Imag. shape (phi, theta, freq)
    leffp_reim = f_leff["leff_phi"]
    # shape (phi, theta, freq) --> (freq, phi, theta)
    lefft_reim = np.moveaxis(lefft_reim, -1, 0)
    # shape (phi, theta, freq) --> (freq, phi, theta)
    leffp_reim = np.moveaxis(leffp_reim, -1, 0)
    leffp = np.abs(leffp_reim)
    lefft = np.abs(lefft_reim)

    phaset = np.angle(lefft_reim, deg=True)
    phasep = np.angle(leffp_reim, deg=True)
    t = DataTable(
        frequency=f,
        theta=theta,
        phi=phi,
        leff_theta_reim=lefft_reim,
        leff_phi_reim=leffp_reim,
        leff_theta=lefft,
        leff_phi=leffp,
        phase_theta=phaset,
        phase_phi=phasep,
    )
    return t

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

    s_parameters_path = params_RF['s_parameters_path']
    if "transfer_function_filename" in params_RF and params_RF["transfer_function_filename"] is not None:
        tf_dict = np.load(os.path.join(s_parameters_path, params_RF["transfer_function_filename"]))
        tf = tf_dict['tf'].astype(np.complex128)
        base_freqs = tf_dict['freqs'].astype(np.float64)
        tf = interp.interp1d(base_freqs, tf, kind='linear', axis=1, bounds_error=False, fill_value=0.0)(in_freqs)
        print("Loaded pre-existing transfer function from file.")
    else: 
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

    t_SN = None
    t_EW = None
    t_Z = None
    if 'path_to_GP300_SN' in params_RF and params_RF["path_to_GP300_SN"] is not None:
        t_SN = open_gp300(params_RF["path_to_GP300_SN"])
    elif 'path_to_horizon_SN' in params_RF and params_RF["path_to_horizon_SN"] is not None:
        t_SN = open_horizon(params_RF["path_to_horizon_SN"])

    if 'path_to_GP300_EW' in params_RF and params_RF["path_to_GP300_EW"] is not None:
        t_EW = open_gp300(params_RF["path_to_GP300_EW"])
    elif 'path_to_horizon_EW' in params_RF and params_RF["path_to_horizon_EW"] is not None:
        t_EW = open_horizon(params_RF["path_to_horizon_EW"])

    if 'path_to_GP300_Z' in params_RF and params_RF["path_to_GP300_Z"] is not None:
        t_Z = open_gp300(params_RF["path_to_GP300_Z"])
    elif 'path_to_horizon_Z' in params_RF and params_RF["path_to_horizon_Z"] is not None:
        t_Z = open_horizon(params_RF["path_to_horizon_Z"])

    if t_SN is None or t_EW is None or t_Z is None:
        raise ValueError("Effective lengths for SN, EW, and Z must be provided in the parameters.")
    
    return duration, latitude, altitude, input_sampling_freq, out_sampling_freq, \
           N_samples, sampling_period, in_freqs, \
           out_N_samples, out_sampling_period, out_freqs, \
           LST_radians, tf, t_SN, t_EW, t_Z

all_objects=set(dir())
__all__ = list(all_objects-module_set-{'module_set'})