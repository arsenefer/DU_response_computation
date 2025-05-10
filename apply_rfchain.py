import numpy as np
import uproot
import matplotlib.pyplot as plt
from glob import glob
import scipy as sp
import uproot
from functools import reduce

from scipy.signal import butter, lfilter
import scipy.interpolate as interp

from typing import Union
from dataclasses import dataclass
from numbers import Number
altitude = 1264
kb = 1.38064852e-23
c = 299792458
Z0 = 4 * np.pi * c * 1e-7

def cart2sph(k):
    """
    Convert cartesian coordinate to spherical coordinate
    """
    r = np.linalg.norm(k, axis=1)
    tp = np.linalg.norm(k[:, :2], axis=1)
    theta = np.arctan2(tp, k[:, 2])
    phi = np.arctan2(k[:, 1], k[:, 0])
    return r, theta, phi

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


def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """
    Apply a Butterworth bandpass filter to the input data.

    This function filters the input signal `data` using a Butterworth bandpass filter
    with the specified low and high cutoff frequencies. The filter can be configured
    as either causal or non-causal.

    Args:
        data (array-like): The input signal to be filtered.
        lowcut (float): The low cutoff frequency of the bandpass filter in Hz.
        highcut (float): The high cutoff frequency of the bandpass filter in Hz.
        fs (float): The sampling frequency of the input signal in Hz.

    Returns:
        array-like: The filtered signal.

    Notes:
        - The filter order is set to 6.
        - The function uses `lfilter` for causal filtering. Uncomment the `filtfilt`
          line to use non-causal filtering instead.
    """
    b, a = butter(6, [lowcut / (0.5 * fs), highcut / (0.5 * fs)],
                  btype="band")  # (order, [low, high], btype)

    # return filtfilt(b, a, data)  # non-causal
    return lfilter(b, a, data)  # causal


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


def open_event_root(directory_to_roots, start=0, stop=None, L1_or_L0='0'):
    """
    Open the ROOT file containing the event data.

    Parameters
    ----------
    directory_to_roots : str
        The path to the directory containing the ROOT files.
    start : int, optional
        The starting index for reading entries. Default is 0.
    stop : int, optional
        The stopping index for reading entries. Default is None.
    L1_or_L0 : str, optional
        Specify whether to use L1 or L0 data. Default is '0'.

    Returns
    -------
    tuple
        - antenna_pos : ndarray
            The positions of the antennas.
        - meta_data : dict
            Metadata about the shower, including core position, zenith, azimuth, etc.
        - efield_data : dict
            The electric field time traces and associated data.
    """
    antenna_pos_file = glob(f'{directory_to_roots}/run_*_L0_*.root')[0]
    shower_meta_data_file = glob(f'{directory_to_roots}/shower_*_L0_*.root')[0]
    efield_file = glob(f'{directory_to_roots}/efield_*_L{L1_or_L0}_*.root')[0]

    with uproot.open(antenna_pos_file) as f:
        antenna_pos = f['trun']['du_xyz'].array().to_numpy()[0]
    with uproot.open(shower_meta_data_file) as f:
        shower_meta_data = f['tshower']
        shower_core_pos = shower_meta_data['shower_core_pos'].array(
            entry_start=start, entry_stop=stop).to_numpy()
        zenith = shower_meta_data['zenith'].array(
            entry_start=start, entry_stop=stop).to_numpy() * np.pi / 180
        azimuth = shower_meta_data['azimuth'].array(
            entry_start=start, entry_stop=stop).to_numpy() * np.pi / 180
        energy_primary = shower_meta_data['energy_primary'].array(
            entry_start=start, entry_stop=stop).to_numpy()
        xmax_grams = shower_meta_data['xmax_grams'].array(
            entry_start=start, entry_stop=stop).to_numpy()
        xmax_pos = shower_meta_data['xmax_pos_shc'].array(
            entry_start=start, entry_stop=stop).to_numpy()
        ptypes = shower_meta_data['primary_type'].array(
            entry_start=start, entry_stop=stop).to_numpy()

    print("Loading ROOT efield files...")
    with uproot.open(efield_file) as f:
        efield_trace = f['tefield']['trace'].array(
            entry_start=start, entry_stop=stop)
        efield_du_ns = f['tefield']['du_nanoseconds'].array(
            entry_start=start, entry_stop=stop)
        efield_du_s = f['tefield']['du_seconds'].array(
            entry_start=start, entry_stop=stop)
        efield_du_id = f['tefield']['du_id'].array(
            entry_start=start, entry_stop=stop)
        efield_event_number = f['tefield']['event_number'].array(
            entry_start=start, entry_stop=stop)
    print("Loaded ROOT efield files...")


    xmax_pos = xmax_pos + shower_core_pos - np.array([[0, 0, altitude]])
    # xmax_pos = xmax_pos + shower_core_pos
    meta_data = {
        'core_pos': shower_core_pos,
        'zenith': zenith,
        'azimuth': azimuth,
        'energy_primary': energy_primary,
        'xmax_grams': xmax_grams,
        'xmax_pos': xmax_pos,
        'p_types': ptypes
    }
    efield_data = {
        'traces': efield_trace,
        'du_s': efield_du_s,
        'du_ns': efield_du_ns,
        'du_id': efield_du_id,
        'event_number': efield_event_number
    }
    return antenna_pos, meta_data, efield_data


def percieved_theta_phi(antenna_pos, xmax_pos):
    """
    Calculate the perceived theta and phi angles of an antenna relative to a source.

    This function computes the spherical coordinates (theta and phi) of the direction
    from the source position (`xmax_pos`) to the antenna position (`antenna_pos`).

    Args:
        antenna_pos (numpy.ndarray): A 3D vector representing the position of the antenna.
        xmax_pos (numpy.ndarray): A 3D vector representing the position of the source.

    Returns:
        tuple: A tuple containing:
            - antenna_percieved_theta (float): The perceived theta angle in radians.
            - antenna_percieved_phi (float): The perceived phi angle in radians, normalized to [0, 2π).
    """
    direction_to_source = antenna_pos - xmax_pos
    _, antenna_percieved_theta, antenna_percieved_phi = cart2sph(
        -direction_to_source)
    return antenna_percieved_theta, antenna_percieved_phi % (2 * np.pi)


def get_leff(t, antenna_percieved_theta, antenna_percieved_phi, duration=4.096e-6, input_sampling_freq=2e9):
    """
    Compute the effective length vector (l_eff) in Cartesian coordinates for an antenna.

    Parameters:
        t (object): An object containing the following attributes:
            - frequency (array-like): Array of frequencies.
            - theta (array-like): Array of theta angles (in degrees).
            - phi (array-like): Array of phi angles (in degrees).
            - leff_theta_reim (ndarray): Real and imaginary parts of the effective length
              in the theta direction, with shape (frequency, theta, phi).
            - leff_phi_reim (ndarray): Real and imaginary parts of the effective length
              in the phi direction, with shape (frequency, theta, phi).
        antenna_percieved_theta (array-like): Array of perceived theta angles (in radians)
            for the antenna.
        antenna_percieved_phi (array-like): Array of perceived phi angles (in radians)
            for the antenna.
        N_sample (int, optional): Number of samples for the FFT. Default is 8192.
        duration (float, optional): Duration of the signal in seconds. Default is 4.096e-6.

    Returns:
        ndarray: Effective length vector in Cartesian coordinates with shape
        (len(antenna_percieved_theta), 3, len(input_freqs_in_band)).
    """
    sampling_period = 1 / input_sampling_freq
    N_sample = int(np.round(duration * input_sampling_freq))
    input_freqs = sp.fft.rfftfreq(N_sample, d=sampling_period)

    #interpolating theta and phi
    leff_theta = interp.interpn(
        (t.theta, t.phi),
        t.leff_theta_reim.swapaxes(0, 2),
        (antenna_percieved_theta[:] * 180 / np.pi,
         antenna_percieved_phi[:] * 180 / np.pi),
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    leff_phi = interp.interpn(
        (t.theta, t.phi),
        t.leff_phi_reim.swapaxes(0, 2),
        (antenna_percieved_theta[:] * 180 / np.pi,
         antenna_percieved_phi[:] * 180 / np.pi),
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    #interpolating frequency
    leff_theta = interp.interp1d(
        t.frequency, leff_theta, axis=1, kind='linear', bounds_error=False, fill_value=0
    )(input_freqs)
    leff_phi = interp.interp1d(
        t.frequency, leff_phi, axis=1, kind='linear', bounds_error=False, fill_value=0
    )(input_freqs)

    c_p, s_p = np.cos(antenna_percieved_phi), np.sin(antenna_percieved_phi)
    c_t, s_t = np.cos(antenna_percieved_theta), np.sin(antenna_percieved_theta)
    e_theta_i = np.vstack((c_t * c_p, c_t * s_p, -s_t)).T
    e_phi_i = np.vstack((-s_p, c_p, np.zeros_like(s_p))).T

    leff_cartesian = (
        e_theta_i[:, :, None] * leff_theta[:, None, :] +
        e_phi_i[:, :, None] * leff_phi[:, None, :]
    )
    return leff_cartesian


def apply_leff(event_E_trace, t, event_pos, event_xmax, duration=4.096e-6, input_sampling_freq=2e9):
    """
    Apply the effective length (L_eff) transformation to an event's electric field time traces.

    This function:
    1. Decomposes the input E-field into θ and φ components based on the perceived direction
       from the antenna to the event.
    2. Applies a bandpass filter to remove frequencies outside the 50–250 MHz range.
    3. Restricts the signal further to the 20–300 MHz range when applying the antenna response.
    4. Retrieves the antenna response data (L_eff) for the θ and φ components, interpolates
       it at the frequency points of interest, and applies it in the frequency domain.
    5. Returns the inverse Fourier transform of the combined θ and φ voltage signals.

    Parameters
    ----------
    event_E_trace : ndarray
        The electric field time traces of shape (N_antennas, 3, N_samples), where each trace
        is a 3D vector over time.
    t : ndarray
        Dataclass containing the effective lengths of the antenna.
    event_pos : ndarray
        The position array of the antennas in meters with shape (n, 3).
    event_xmax : ndarray
        The 3D position array (x, y, z) of the maximum emission in meters.
    N_sample : int, optional
        Number of samples for the FFT. Default is 8192.
    duration : float, optional
        Duration of the signal in seconds. Default is 4.096e-6.

    Returns
    -------
    ndarray
        The time-domain voltage signals (N_antennas, N_samples), representing the sum of θ and
        φ components after applying the bandpass filter and the antenna response.
    """
    sampling_period = 1 / input_sampling_freq
    N_sample = int(np.round(duration * input_sampling_freq))
    input_freqs = sp.fft.rfftfreq(N_sample, d=sampling_period)  # Frequency axis

    antenna_perceived_direction = event_pos - event_xmax
    dist, antenna_perceived_theta, antenna_perceived_phi = cart2sph(
        -antenna_perceived_direction)


    print(
        f'Antenna response between {t.frequency.min() / 1e6:.0f} MHz and {t.frequency.max() / 1e6:.0f} MHz')

    antenna_response_theta = interp.interpn(
        (t.theta, t.phi),
        t.leff_theta_reim.swapaxes(0, 2),
        (antenna_perceived_theta[:] * 180 / np.pi,
         antenna_perceived_phi[:] * 180 / np.pi),
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    antenna_response_phi = interp.interpn(
        (t.theta, t.phi),
        t.leff_phi_reim.swapaxes(0, 2),
        (antenna_perceived_theta[:] * 180 / np.pi,
         antenna_perceived_phi[:] * 180 / np.pi),
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    antenna_response_theta_stretch = interp.interp1d(
        t.frequency, antenna_response_theta, axis=1, kind='linear', bounds_error=False, fill_value=0
    )(input_freqs)

    antenna_response_phi_stretch = interp.interp1d(
        t.frequency, antenna_response_phi, axis=1, kind='linear', bounds_error=False, fill_value=0
    )(input_freqs)

    e_theta_i = np.vstack((
        np.cos(antenna_perceived_theta) * np.cos(antenna_perceived_phi),
        np.cos(antenna_perceived_theta) * np.sin(antenna_perceived_phi),
        -np.sin(antenna_perceived_theta)
    )).T
    e_phi_i = np.vstack((
        -np.sin(antenna_perceived_phi),
        np.cos(antenna_perceived_phi),
        np.zeros(len(dist))
    )).T

    event_E_theta = (e_theta_i[:, :, None] * event_E_trace).sum(axis=1)
    event_E_phi = (e_phi_i[:, :, None] * event_E_trace).sum(axis=1)

    event_E_theta_fft = sp.fft.rfft(event_E_theta, axis=1)
    event_E_phi_fft = sp.fft.rfft(event_E_phi, axis=1)


    event_VOC_theta_fft = event_E_theta_fft.copy()
    event_VOC_theta_fft[:, :] *= antenna_response_theta_stretch

    event_VOC_phi_fft = event_E_phi_fft.copy()
    event_VOC_phi_fft[:, :] *= antenna_response_phi_stretch

    tot_fft = event_VOC_theta_fft + event_VOC_phi_fft
    return sp.fft.irfft(tot_fft, axis=1), tot_fft


def make_voc(event_E_trace, t, event_pos, event_xmax, duration=4.096e-6, input_sampling_freq=2e9, bp_filter=False):
    """
    Apply the effective length (L_eff) transformation to an event's electric field time traces. and apply filtering
    This function:
    1. Decomposes the input E-field into θ and φ components based on the perceived direction 
         from the antenna to the event.
    2. Applies a bandpass filter to remove frequencies outside the 50–250 MHz range.
    3. Restricts the signal further to the 20–300 MHz range when applying the antenna response.
    4. Retrieves the antenna response data (L_eff) for the θ and φ components, interpolates 
         it at the frequency points of interest, and applies it in the frequency domain.
    5. Returns the inverse Fourier transform of the combined θ and φ voltage signals.
    Parameters
    ----------
    event_E_trace : ndarray
         The electric field time traces of shape (N_antennas, 3, N_samples), where each trace 
         is a 3D vector over time.
    t : ndarray
         Dataclass containing the effictive lenghts of the antenna.
    zenith : float
         The zenith angle in radians of the incoming signal.
    azimuth : float
         The azimuth angle in radians of the incoming signal.
    event_pos : ndarray
         The position array of the antennas in meter with shape n*3.
    event_xmax : ndarray
         The 3D position array (x, y, z) of the maximum emission in meters.
    Returns
    -------
    ndarray
         The time-domain voltage signals (N_antennas, N_samples), representing the sum of θ and 
         φ components after applying the bandpass filter and the antenna response.
    """
    sampling_period = 1 / input_sampling_freq
    N_sample = int(np.round(duration * input_sampling_freq))
    input_freqs = sp.fft.rfftfreq(N_sample, d=sampling_period)

    if (input_sampling_freq/2 > 250*1e6) & bp_filter:
        event_E_trace_filtered = _butter_bandpass_filter(
            event_E_trace, 50*1e6, 250*1e6, input_sampling_freq)
    else:
        event_E_trace_filtered = event_E_trace

    voc, voc_FFT = apply_leff(event_E_trace_filtered, t, event_pos,
                              event_xmax, N_sample=N_sample, duration=duration)

    return voc, voc_FFT


def interpol_at_new_x(a_x, a_y, new_x):
    """
    Interpolation of discreet function F defined by set of point F(a_x)=a_y for new_x value
    and set to zero outside interval definition a_x

    :param a_x (float, (N)): F(a_x) = a_y, N size of a_x
    :param a_y (float, (N)): F(a_x) = a_y
    :param new_x (float, (M)): new value of x

    :return: F(new_x) (float, (M)): interpolation of F at new_x
    # RK: scipy interpolate gave 0 values for S21 due to fill_values=(0,0)
    #.    which resulted in 'nan' values in A-parameters. Also final transfer
    #     function (TF) outside of the range of 10-300 MHz was weird. TF for Z-port produces a sharp peak around 10 MHz.
    #     So np.interp is used instead.
    """
    assert a_x.shape[0] > 0
    return np.interp(new_x, a_x, a_y, )


def open_Zload(s_map, target_freqs):
    dbs = s_map[:, 1]
    mag = 10 ** (dbs / 20)
    angs = np.deg2rad(s_map[:, 2])
    s = interpol_at_new_x(
        s_map[:, 0], mag*np.cos(angs) + 1j*mag*np.sin(angs), target_freqs)
    Z_load = 50 * (1 + s) / (1 - s)
    Z_load = Z_load
    return Z_load


def open_Zant(zant_map, target_freqs, axis=0):
    freqs_in = zant_map[:, 0]

    Z_ant = interpol_at_new_x(
        freqs_in*1e6, zant_map[:, 1+2*axis] + 1j*zant_map[:, 2+2*axis], target_freqs)
    return Z_ant


def s2abcd(s11, s21, s12, s22):
    """this is a normalized A-matrix represented by [a] in the document."""
    return np.moveaxis(
        np.asarray([
            [((1 + s11) * (1-s22) + s12 * s21) / (2 * s21),
             ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)],
            [((1 - s11) * (1-s22) - s12 * s21) / (2 * s21),
             ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)]
        ], dtype=np.complex128),
        [0, 1], [-2, -1]
    )


def s_file_2_abcd(s_map, target_freqs, db=False):
    """
    Converts S-parameters from a given mapping to an ABCD matrix and interpolates 
    the S-parameters to the specified target frequencies.
    Parameters:
    -----------
    s_map : numpy.ndarray
        A 2D array where each row corresponds to a frequency point
    target_freqs : numpy.ndarray
        A 1D array of target frequency points (Hz) to which the S-parameters 
        will be interpolated.
    db : bool, optional
        If True, the magnitude values in `s_map` are assumed to be in dB and 
        will be converted to linear scale. If False, the magnitude values are 
        assumed to be in linear scale. Default is False.
    Returns:
    --------
    ABCD_matrix : numpy.ndarray
        A 2D array representing the ABCD matrix after converting the S-parameters 
        and applying normalization.
    s_parameters : tuple
        A tuple containing the interpolated S-parameters (s11, s21, s12, s22), 
        where each element is a 1D numpy array corresponding to the target 
        frequency points.
    """
    nb_freqs = len(target_freqs)
    freqs_in = s_map[:, 0]

    def s_from_mag_angle(mag, angle, freqs_in, target_freqs):
        res = mag * np.cos(angle)
        ims = mag * np.sin(angle)
        s = res + 1j * ims
        return interpol_at_new_x(freqs_in, s, target_freqs)

    def s_from_DB_angle(DB, angle, freqs_in, target_freqs):
        mag = 10**(DB/20)
        res = mag * np.cos(angle)
        ims = mag * np.sin(angle)
        s = res + 1j * ims
        return interpol_at_new_x(freqs_in, s, target_freqs)

    to_s = s_from_DB_angle if db else s_from_mag_angle

    amplitudes11 = s_map[:, 1].astype(np.complex128)  # is it in dB or not?
    # angle in deg, converted to rad
    angs11 = np.pi / 180 * (s_map[:, 2].astype(np.complex128))
    s11 = to_s(amplitudes11, angs11, freqs_in, target_freqs)

    amplitudes21 = s_map[:, 3].astype(np.complex128)  # is it in dB or not?
    # angle in deg, converted to rad
    angs21 = np.pi / 180 * (s_map[:, 4].astype(np.complex128))
    s21 = to_s(amplitudes21, angs21, freqs_in, target_freqs)

    amplitudes12 = s_map[:, 5].astype(np.complex128)  # is it in dB or not?
    # angle in deg, converted to rad
    angs12 = np.pi / 180 * (s_map[:, 6].astype(np.complex128))
    s12 = to_s(amplitudes12, angs12, freqs_in, target_freqs)

    amplitudes22 = s_map[:, 7].astype(np.complex128)  # is it in dB or not?
    # angle in deg, converted to rad
    angs22 = np.pi / 180 * (s_map[:, 8].astype(np.complex128))
    s22 = to_s(amplitudes22, angs22, freqs_in, target_freqs)

    ABCD_matrix = s2abcd(s11, s21, s12, s22)

    xy_denorm_factor = np.array([[1, 50], [1/50., 1]])
    xy_denorm_factor = xy_denorm_factor
    ABCD_matrix *= xy_denorm_factor

    return ABCD_matrix, (s11, s21, s12, s22)


def total_abcd_matrix(list_abcd, Z_load, balun2_abcd=None):
    total_abcd = reduce(np.matmul, list_abcd)
    Z_in = (total_abcd[:, 0, 0] * Z_load + total_abcd[:, 0, 1]) / \
        (total_abcd[:, 1, 0] * Z_load + total_abcd[:, 1, 1])
    if type(balun2_abcd) is not type(None):
        total_abcd = total_abcd @ balun2_abcd
    return total_abcd, Z_in


def abcd_2_tf(abcd_matrix, Z_ant, Z_in):
    ABCD_inv = np.linalg.inv(abcd_matrix)
    V_I = np.stack([Z_in / (Z_ant + Z_in),  1 / (Z_ant + Z_in)], axis=-1)
    V_I_out_RFchain = (ABCD_inv @ V_I[:, :, None])[:, :, 0]
    return V_I_out_RFchain[:, 0]


def smap_2_tf(list_s_maps, zload_map, zant_map, target_freqs, is_db=None, balun_2_map=None, axis=0):
    """
    Computes the transfer function (TF) from a series of S-parameter maps, load impedance, 
    and antenna impedance over a range of target frequencies.

    Args:
        list_s_maps (list): A list of file paths or data structures containing S-parameter maps.
        zload_map (str or object): File path or data structure representing the load impedance map.
        zant_map (str or object): File path or data structure representing the antenna impedance map.
        target_freqs (array-like): A list or array of target frequencies for which the transfer 
                                   function is computed.
        is_db (list, optional): A list of booleans indicating whether each S-parameter map in 
                                `list_s_maps` is in decibels (True) or linear scale (False). 
                                Defaults to None, which assumes all maps are in linear scale.
        balun_2_map (str or object, optional): File path or data structure representing the 
                                               second balun S-parameter map. Defaults to None.
        axis (int, optional): Axis along which the antenna impedance map is evaluated. 
                              Defaults to 0.

    Returns:
        array-like: The computed transfer function (TF) over the specified target frequencies.
    """
    if type(is_db) is type(None):
        is_db = [False]*len(list_s_maps)
    list_abcd = []
    for s_map, db in zip(list_s_maps, is_db):
        ABCD_matrix, (s11, s21, s12, s22) = s_file_2_abcd(
            s_map, target_freqs, db=db)
        list_abcd.append(ABCD_matrix)
    ABCD_matrix_balun2, _ = s_file_2_abcd(balun_2_map, target_freqs, db=False)

    Z_load = open_Zload(zload_map, target_freqs)
    Z_ant = open_Zant(zant_map, target_freqs, axis=axis)
    ABCD_tot, Z_in = total_abcd_matrix(
        list_abcd, Z_load, balun2_abcd=ABCD_matrix_balun2)
    tf = abcd_2_tf(ABCD_tot, Z_ant, Z_in)
    return tf


def efield_2_voltage(
    event_trace_fft, full_response, target_rate=2e9, current_rate=2e9
):
    """
    Converts electric field data in the frequency domain to voltage data in the time domain.

    Parameters:
        event_trace_fft (numpy.ndarray): A 3D array containing the FFT of the electric field 
            traces. The shape is expected to be (n_events, n_channels, n_frequencies).
        full_response (numpy.ndarray): A 4D array representing the full system response. 
            The shape is expected to be (n_events, n_channels, n_frequencies_in_band, n_frequencies).
        target_rate (float, optional): The target sampling rate for the output time-domain 
            voltage signal, in Hz. Default is 2e9 (2 GHz).
        current_rate (float, optional): The current sampling rate of the input frequency-domain 
            data, in Hz. Default is 2e9 (2 GHz).

    Returns:
        tuple:
            - numpy.ndarray: The time-domain voltage signal after inverse FFT, scaled by the 
              ratio of target_rate to current_rate.
            - numpy.ndarray: The modified frequency-domain voltage signal.
    """
    vout_f = np.einsum(
        "ijk,ijlk->ilk", event_trace_fft, full_response
    )
    # vout_f[:, :, in_antenna_band] = vout_fft_inband
    ratio = target_rate / current_rate
    m = int((vout_f.shape[-1] - 1) * 2 * ratio)
    return sp.fft.irfft(vout_f, m) * ratio, vout_f[...,:m//2+1]

def voltage_to_adc(voltage_traces, micro=True):
    """
    Converts voltage traces to ADC values.

    Parameters:
        voltage_traces (numpy.ndarray): A 2D array of voltage traces with shape 
            (n_events, n_samples).

    Returns:
        numpy.ndarray: A 2D array of ADC values with the same shape as the input.
    """
    adc_bins = .9/8192
    if micro:
        voltage_traces *= 1e-6
    adc_values = np.clip(voltage_traces, -0.9, 0.9)
    adc_values = np.round(adc_values / adc_bins).astype(np.int16)
    return adc_values
    
def latlon2zenaz(detector_lat, lst_rad, lat_map, long_map, mod_pi=True, add_pi=False):
    """
    lst_rad:latitude of the detector
    detector_lat:longitude of the detector
    lat_map:latitude of the source
    long_map:longitude of the source
    add_pi: if True, add pi to the azimuth angle
    """
    coszenithp = + np.cos(lst_rad)*np.sin(detector_lat)*np.cos(long_map)*np.sin(lat_map) \
        + np.sin(lst_rad)*np.sin(detector_lat)*np.sin(long_map)*np.sin(lat_map) \
        + np.cos(detector_lat)*np.cos(lat_map)
    coszenithp = np.clip(coszenithp, -1, 1)

    NX = - np.cos(lst_rad)*np.cos(detector_lat)*np.cos(long_map)*np.sin(lat_map)\
         - np.sin(lst_rad)*np.cos(detector_lat)*np.sin(long_map)*np.sin(lat_map)\
        + np.sin(detector_lat)*np.cos(lat_map)
    WX = + np.sin(lst_rad)*np.cos(long_map)*np.sin(lat_map) \
        - np.cos(lst_rad)*np.sin(long_map)*np.sin(lat_map)
    zenithp = np.arccos(coszenithp)
    azimuthp = np.arctan2(WX, NX)

    assert add_pi ^ mod_pi, "Exactly one of add_pi or mod_pi should be True"

    if add_pi:
        azimuthp = azimuthp + np.pi
    elif mod_pi:
        azimuthp = azimuthp % (2*np.pi)
    return zenithp, azimuthp

def plot_quantities(lst_rad, freq_idx, all_zenith, all_azimuth, leff_interpolated_theta, leff_interpolated_phi, A_eff, B_nu, long_map, lat_map, detector_lat):
    """
    Plot various quantities such as zenith, azimuth, effective lengths, effective area, and brightness temperature.

    Parameters:
        lst_rad (float): Local sidereal time in radians. 
        all_zenith (ndarray): Zenith angles in radians.
        all_azimuth (ndarray): Azimuth angles in radians.
        leff_interpolated_theta (ndarray): Interpolated effective length in the theta direction.
        leff_interpolated_phi (ndarray): Interpolated effective length in the phi direction.
        A_eff (ndarray): Effective area.
        B_nu (ndarray): Brightness temperature.
        long_map (ndarray): Longitude map.
        lat_map (ndarray): Latitude map.
        detector_lat (float): Latitude of the detector in radians.
    """
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex=True, sharey=True)

    ims = axs.T[0, 0].imshow((all_zenith * (all_zenith < np.pi / 2) * 180 / np.pi).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[0, 0])
    axs.T[0, 0].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[0, 0].set_ylabel('S <- Co-DEC -> N')
    axs.T[0, 0].set_title('Zenith [deg]')

    ims = axs.T[1, 0].imshow((all_azimuth * (all_zenith < np.pi / 2) * 180 / np.pi).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[1, 0])
    axs.T[1, 0].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[1, 0].set_title('Azimuth [deg]')

    ims = axs.T[0, 1].imshow((np.abs(leff_interpolated_theta[freq_idx]) * (all_zenith < np.pi / 2)).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[0, 1])
    axs.T[0, 1].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[0, 1].set_title('Leff theta [m]')
    axs.T[0, 1].set_ylabel('S <- Co-DEC -> N')

    ims = axs.T[1, 1].imshow((np.abs(leff_interpolated_phi[freq_idx]) * (all_zenith < np.pi / 2)).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[1, 1])
    axs.T[1, 1].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[1, 1].set_title('Leff phi [m]')

    ims = axs.T[0, 2].imshow(((A_eff[freq_idx]) * (all_zenith < np.pi / 2)).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[0, 2])
    axs.T[0, 2].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[0, 2].set_title('Aeff [m²]')
    axs.T[0, 2].set_ylabel('S <- Co-DEC -> N')
    axs.T[0, 2].set_xlabel('west <- RA -> east')

    ims = axs.T[1, 2].imshow(((B_nu) * (all_zenith < np.pi / 2)).T,
                                extent=[180 / np.pi * long_map.min() - 180,
                                        180 / np.pi * long_map.max() - 180,
                                        180 / np.pi * lat_map.max(),
                                        180 / np.pi * lat_map.min()],
                                cmap='jet')
    plt.colorbar(ims, ax=axs.T[1, 2])
    axs.T[1, 2].scatter(lst_rad * 180 / np.pi - 180, detector_lat * 180 / np.pi, color='red', marker='*', s=50)
    axs.T[1, 2].set_xlabel('west <- RA -> east')
    axs.T[1, 2].set_title('$B_\\nu$ [W.m$^{-2}$.Hz$^{-1}$.sr$^{-1}$]')
    plt.tight_layout()

class compute_noise():
    def __init__(self, 
                 lst_time_resolution, 
                 detector_lat, 
                 list_temp_files, 
                 LF_freqs, 
                 target_freqs,
                 tf_rfchain,
                 duration=4.096e-6,
                 leff_x=None, 
                 leff_y=None, 
                 leff_z=None):
        """
        Initialize the compute_noise class.
        Parameters
        ----------
        lst_time_resolution : float
            Time resolution for the local sidereal time (LST) in hours.
        detector_lat : float
            Latitude of the detector in radians.
        list_temp_files : list
            List of temperature map files.
        LF_freqs : array
            Frequencies for the low-frequency band.
        target_freqs : array
            Target frequencies for the RF chain.
        tf_rfchain : array
            Transfer function of the RF chain.
        duration : float, optional
            Duration of the signal in seconds. Default is 4.096e-6.
        leff_x : object, optional
            Effective length data for the x-direction. Default is None.
        leff_y : object, optional
            Effective length data for the y-direction. Default is None.
        leff_z : object, optional
            Effective length data for the z-direction. Default is None.
        """
        self.detector_lat = detector_lat
        self.lst_hours = np.arange(0, 24, lst_time_resolution)

        self.list_temp_files = list_temp_files
        self.LF_freqs = LF_freqs

        self.long_map, self.lat_map, _ = np.load(list_temp_files[0])

        self.delta_lat = np.abs(self.lat_map[0, 0]-self.lat_map[0, 1])
        self.delta_long = np.abs(self.long_map[0, 0]-self.long_map[1, 0])

        assert (type(leff_x) is not type(None)) or (type(leff_y) is not type(None)) or (type(
            leff_z) is not type(None)), "At least one of leff_x, leff_y, leff_z should be provided"

        self.l_eff_theta = leff_x.theta*np.pi/180
        self.l_eff_phi = leff_x.phi*np.pi/180

        self.leff_x_theta_reim = interp.interp1d(leff_x.frequency, leff_x.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_x_phi_reim = interp.interp1d(leff_x.frequency, leff_x.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        self.leff_y_theta_reim = interp.interp1d(leff_y.frequency, leff_y.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_y_phi_reim = interp.interp1d(leff_y.frequency, leff_y.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        self.leff_z_theta_reim = interp.interp1d(leff_z.frequency, leff_z.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_z_phi_reim = interp.interp1d(leff_z.frequency, leff_z.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        n_freqs = tf_rfchain.shape[-1]
        self.tf = interp.interp1d(np.linspace(0, (n_freqs-1)/duration, n_freqs), tf_rfchain, axis=1, kind='quadratic', bounds_error=False, fill_value=0
        )(target_freqs)
        self.target_freqs = target_freqs
        
    @property
    def lst_rads(self):
        return self.lst_hours / 24 * 2 * np.pi
    
    def freq_index(self, freq):
        """
        Returns the index of the frequency in the list of frequencies.
        """
        return np.where(self.LF_freqs == freq)[0][0]

    def get_temp_map(self, freq_idx):
        """
        Returns the temperature map for the given frequency index.
        """
        _, _, temp_map = np.load(self.list_temp_files[freq_idx])
        return temp_map

    def latlon2zenaz(self, lst_rad, mod_pi=True, add_pi=False):
        """
        lst_rad:latitude of the detector
        detector_lat:longitude of the detector
        lat_map:latitude of the source
        self.long_map:longitude of the source
        add_pi: if True, add pi to the azimuth angle
        """
        coszenithp = + np.cos(lst_rad)*np.sin(self.detector_lat)*np.cos(self.long_map)*np.sin(self.lat_map) \
            + np.sin(lst_rad)*np.sin(self.detector_lat)*np.sin(self.long_map)*np.sin(self.lat_map) \
            + np.cos(self.detector_lat)*np.cos(self.lat_map)
        coszenithp = np.clip(coszenithp, -1, 1)

        NX = - np.cos(lst_rad)*np.cos(self.detector_lat)*np.cos(self.long_map)*np.sin(self.lat_map)\
            - np.sin(lst_rad)*np.cos(self.detector_lat)*np.sin(self.long_map)*np.sin(self.lat_map)\
            + np.sin(self.detector_lat)*np.cos(self.lat_map)
        WX = + np.sin(lst_rad)*np.cos(self.long_map)*np.sin(self.lat_map) \
            - np.cos(lst_rad)*np.sin(self.long_map)*np.sin(self.lat_map)
        zenithp = np.arccos(coszenithp)
        azimuthp = np.arctan2(WX, NX)

        assert add_pi ^ mod_pi, "Exactly one of add_pi or mod_pi should be True"

        if add_pi:
            azimuthp = azimuthp + np.pi
        elif mod_pi:
            azimuthp = azimuthp % (2*np.pi)
        return zenithp, azimuthp

    def noise_power(self, plot=False):
        """
        Calculate the noise power in frequency domains.

        This method computes the noise power for a given local sidereal time (LST) 
        in radians and frequency. It uses effective lengths, temperature maps, and 
        other parameters to calculate the power.

        Args:
            lst_rad (float): Local sidereal time in radians.
            freq (float): Frequency at which the noise power is to be calculated.

        Returns:
            numpy.ndarray: A 1D array containing the noise power for the x, y, and z 
            components.

        Notes:
            - The method interpolates effective lengths (`l_eff`) over azimuth and 
              zenith angles.
            - The noise power is calculated using the Planck law for blackbody 
              radiation and the effective area.
            - The integration considers the latitude map and angular resolution 
              (`delta_lat` and `delta_long`).

        Dependencies:
            - `latlon2zenaz`: Converts latitude and longitude to zenith and azimuth angles.
            - `freq_index`: Maps the frequency to its corresponding index.
            - `get_temp_map`: Retrieves the temperature map for a given frequency index.
            - `interp.interpn`: Performs multi-dimensional interpolation.
            - Constants: `kb` (Boltzmann constant), `c` (speed of light).

        """
        P_nuxyz = np.zeros((len(self.lst_rads), 3, len(self.LF_freqs)))
        for lst_idx, lst_rad in enumerate(self.lst_rads[:]):
            print(
                f"Calculating noise power for LST {lst_rad*12/np.pi:.2f} hours")
            all_zenith, all_azimuth = self.latlon2zenaz(lst_rad, mod_pi=True)

            for coord_idx, l_effs in enumerate([(self.leff_x_theta_reim, self.leff_x_phi_reim),
                                               (self.leff_y_theta_reim,
                                                self.leff_y_phi_reim),
                                               (self.leff_z_theta_reim, self.leff_z_phi_reim)]):
                if type(l_effs) is type(None):
                    continue
                lt = np.rollaxis(l_effs[0], 0, l_effs[0].ndim)
                leff_interpolated_theta = interp.interpn((self.l_eff_phi, self.l_eff_theta),
                                                         lt,
                                                         (all_azimuth, all_zenith),
                                                         bounds_error=False, fill_value=0)
                leff_interpolated_theta = np.rollaxis(
                    leff_interpolated_theta, -1, 0)

                lp = np.rollaxis(l_effs[1], 0, l_effs[1].ndim)
                leff_interpolated_phi = interp.interpn((self.l_eff_phi, self.l_eff_theta),
                                                       lp,
                                                       (all_azimuth, all_zenith),
                                                       bounds_error=False, fill_value=0)
                leff_interpolated_phi = np.rollaxis(
                    leff_interpolated_phi, -1, 0)

                A_eff = np.abs(leff_interpolated_theta) ** 2 + \
                    np.abs(leff_interpolated_phi)**2
                for freq_idx, freq in enumerate(self.LF_freqs):
                    temp_map = self.get_temp_map(freq_idx)
                    B_nu = 2 * (freq)**2 * kb * temp_map/(c**2)
                    if (np.abs(freq - 95e6) < 10) and ((lst_rad*12/np.pi) % 12 == 6) and plot:
                        plot_quantities(lst_rad, freq_idx, all_zenith, all_azimuth,
                                        leff_interpolated_theta, leff_interpolated_phi, A_eff, B_nu, self.long_map, self.lat_map, self.detector_lat)
                        plt.show()

                    P_nu = 1/2 * \
                        A_eff[freq_idx] * B_nu * \
                        np.sin(self.lat_map) * self.delta_lat * self.delta_long
                    P_nu = np.sum(P_nu)
                    P_nuxyz[lst_idx, coord_idx, freq_idx] = P_nu

        P_nuxyz = np.array(P_nuxyz)
        self._P_nu = P_nuxyz
        return P_nuxyz
    
    @property
    def P_nu(self):
        """
        Returns the noise power in frequency domains.
        """
        if not hasattr(self, '_P_nu'):
            return self.noise_power()
        return self._P_nu

    def noise_rms_traces(self):
        """
        Calculate the noise RMS traces.
        Args:
            target_freqs (array-like): Target frequencies for the noise calculation.
            tf_rfchain (array-like): Transfer function of the RF chain.
        Returns:
            numpy.ndarray: The noise RMS traces.
        """
        N = 2 * (len(self.target_freqs)-1) 
        fs = 2 * self.target_freqs[-1]
        P_nu = self.P_nu
        V_rms_voc_2 = 2 * Z0 * P_nu
        V_rms_voc_2_target = interp.interp1d(
            self.LF_freqs, V_rms_voc_2, bounds_error=False, fill_value=0, axis=-1)(self.target_freqs)
        V_rms_voc_target = V_rms_voc_2_target * N * fs / 2
        V_rms_voc_target = np.sqrt(V_rms_voc_target)

        self._noise_spectrum = np.abs(V_rms_voc_target * self.tf)
        self._noise_spectrum = self._noise_spectrum / 2
        return self._noise_spectrum

    @property
    def noise_spectrum(self):
        if hasattr(self, '_noise_spectrum'):
            return self._noise_spectrum
        elif hasattr(self, 'path_to_noise_spectrum'):
            self._noise_spectrum = np.load(self.path_to_noise_spectrum)
            self.LF_freqs = np.load(self.path_to_noise_spectrum.replace(
                '.npy', '_frequencies.npy'))
            self.lst_hours = np.load(self.path_to_noise_spectrum.replace(
                '.npy', '_lsthours.npy'))
            return 
        return self.noise_rms_traces()

    def save_spectrum(self, directory='.', name='noise_spectrum'):
        """
        Save the noise spectrum to a file.

        Args:
            directory (str): Directory where the file will be saved.
            name (str): Name of the file (without extension).
        """
        if not (hasattr(self, 'noise_spectrum')):
            raise ValueError(
                "Noise spectrum not computed. Run noise_rms_traces(target_frqs, tf_rfchain) first.")
        np.save(f'{directory}/{name}.npy', self.noise_spectrum)
        np.save(f'{directory}/{name}_frequencies.npy', self.LF_freqs)
        np.save(f'{directory}/{name}_lsthours.npy', self.lst_hours)


    def noise_samples(self, lst_hour, n_samples=1, seed=None, micro=True):
        """
        Generate noise samples based on the noise spectrum.
        Args:
            lst_hour (float): Local sidereal time in hours.
            n_samples (int): Number of samples to generate.
            seed (int, optional): Random seed for reproducibility.
            micro (bool): If True, convert voltage traces to microvolts.
        Returns:    
            tuple: A tuple containing the complex FFT of the noise samples and the
                     frequency-domain noise samples.
                     v_noise: Time-domain noise samples. (shape: (n_samples, 3, N_samples))
                     v_complex_fft: Complex FFT of the noise samples. (shape: (n_samples, 3, n_freqs))
        """

        lst_idx = np.abs(self.lst_hours - lst_hour).argmin()
        n_freqs = len(self.target_freqs)

        rng = np.random.default_rng(seed)
        
        amp = rng.normal(loc=0, scale=self.noise_spectrum[lst_idx], size=(
            n_samples, 3, len(self.target_freqs)))
        phase = 2 * np.pi * rng.random(size=(n_samples, 3, n_freqs))
        v_complex_fft = amp * np.exp(1j*phase)
        v_noise = np.fft.irfft(v_complex_fft, axis=-1)
        if micro:
            v_noise *= 1e6
            v_complex_fft *= 1e6
        return v_noise, v_complex_fft


