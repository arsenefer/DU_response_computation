import numpy as np 
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from glob import glob
import scipy as sp
import uproot
from functools import reduce

import sys
import os
import scipy.signal as ssi
import random
from scipy.signal import butter, filtfilt, lfilter, hilbert
import scipy.interpolate as interp

from PWF_reconstruction.utils import sph2cart, cart2sph

from typing import Union, Any
from dataclasses import dataclass
from numbers import Number
altitude = 1264

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
    b, a = butter(6, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype="band")  # (order, [low, high], btype)

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
    f, R, X, theta, phi, lefft, leffp, phaset, phasep = np.load(path_to_horizon, mmap_mode="r")

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
    lefft_reim = f_leff["leff_theta"]   # Real + j Imag. shape (phi, theta, freq) (361, 91, 221)
    leffp_reim = f_leff["leff_phi"]     # Real + j Imag. shape (phi, theta, freq)
    lefft_reim = np.moveaxis(lefft_reim, -1, 0) # shape (phi, theta, freq) --> (freq, phi, theta)
    leffp_reim = np.moveaxis(leffp_reim, -1, 0) # shape (phi, theta, freq) --> (freq, phi, theta)
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

    antenna_pos = uproot.open(antenna_pos_file)['trun']['du_xyz'].array().to_numpy()[0]
    shower_meta_data = uproot.open(shower_meta_data_file)['tshower']

    with uproot.open(efield_file) as f:
        efield_trace = f['tefield']['trace'].array(entry_start=start, entry_stop=stop)
        efield_du_ns = f['tefield']['du_nanoseconds'].array(entry_start=start, entry_stop=stop)
        efield_du_s = f['tefield']['du_seconds'].array(entry_start=start, entry_stop=stop)
        efield_du_id = f['tefield']['du_id'].array(entry_start=start, entry_stop=stop)
        efield_event_number = f['tefield']['event_number'].array(entry_start=start, entry_stop=stop)

    shower_core_pos = shower_meta_data['shower_core_pos'].array(entry_start=start, entry_stop=stop).to_numpy()
    zenith = shower_meta_data['zenith'].array(entry_start=start, entry_stop=stop).to_numpy() * np.pi / 180
    azimuth = shower_meta_data['azimuth'].array(entry_start=start, entry_stop=stop).to_numpy() * np.pi / 180
    energy_primary = shower_meta_data['energy_primary'].array(entry_start=start, entry_stop=stop).to_numpy()
    xmax_grams = shower_meta_data['xmax_grams'].array(entry_start=start, entry_stop=stop).to_numpy()
    xmax_pos = shower_meta_data['xmax_pos_shc'].array(entry_start=start, entry_stop=stop).to_numpy()

    xmax_pos = xmax_pos + shower_core_pos - np.array([[0, 0, altitude]])
    meta_data = {
        'core_pos': shower_core_pos,
        'zenith': zenith,
        'azimuth': azimuth,
        'energy_primary': energy_primary,
        'xmax_grams': xmax_grams,
        'xmax_pos': xmax_pos,
        'p_types': shower_meta_data['primary_type'].array(entry_start=start, entry_stop=stop).to_numpy()
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
    _, antenna_percieved_theta, antenna_percieved_phi = cart2sph(-direction_to_source)
    return antenna_percieved_theta, antenna_percieved_phi % (2 * np.pi)

def get_leff(t, antenna_percieved_theta, antenna_percieved_phi, N_sample=8192, duration=4.096e-6):
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
        (len(antenna_percieved_theta), 3, len(target_freqs_in_band)).
    """
    sampling_period = duration / N_sample
    target_freqs = sp.fft.rfftfreq(N_sample, d=sampling_period)
    in_antenna_band = (target_freqs >= t.frequency.min()) & (target_freqs <= t.frequency.max())

    l_eff_theta = interp.interpn(
        (t.theta, t.phi),
        t.leff_theta_reim.swapaxes(0, 2),
        (antenna_percieved_theta[:] * 180 / np.pi, antenna_percieved_phi[:] * 180 / np.pi),
        method='linear'
    )
    l_eff_phi = interp.interpn(
        (t.theta, t.phi),
        t.leff_phi_reim.swapaxes(0, 2),
        (antenna_percieved_theta[:] * 180 / np.pi, antenna_percieved_phi[:] * 180 / np.pi),
        method='linear'
    )
    l_eff_theta = interp.interp1d(
        t.frequency, l_eff_theta, axis=1, kind='cubic'
    )(target_freqs[in_antenna_band])
    l_eff_phi = interp.interp1d(
        t.frequency, l_eff_phi, axis=1, kind='cubic'
    )(target_freqs[in_antenna_band])

    c_p, s_p = np.cos(antenna_percieved_phi), np.sin(antenna_percieved_phi)
    c_t, s_t = np.cos(antenna_percieved_theta), np.sin(antenna_percieved_theta)
    e_theta_i = np.vstack((c_t * c_p, c_t * s_p, -s_t)).T
    e_phi_i = np.vstack((-s_p, c_p, np.zeros_like(s_p))).T

    l_eff_cartesian = (
        e_theta_i[:, :, None] * l_eff_theta[:, None, :] +
        e_phi_i[:, :, None] * l_eff_phi[:, None, :]
    )
    return l_eff_cartesian

def apply_leff(event_E_trace, t, event_pos, event_xmax, N_sample=8192, duration=4.096 * 1e-6):
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
    sampling_period = duration / N_sample

    antenna_perceived_direction = event_pos - event_xmax
    dist, antenna_perceived_theta, antenna_perceived_phi = cart2sph(-antenna_perceived_direction)

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
    freqs = sp.fft.rfftfreq(N_sample, d=sampling_period)  # Frequency axis

    print(f'Antenna response between {t.frequency.min() / 1e6:.0f} MHz and {t.frequency.max() / 1e6:.0f} MHz')

    in_antenna_band = (freqs >= t.frequency.min()) & (freqs <= t.frequency.max())

    event_E_theta_fft_filtered = event_E_theta_fft.copy()
    event_E_phi_fft_filtered = event_E_phi_fft.copy()

    event_E_theta_fft_filtered[:, ~in_antenna_band] = 0
    event_E_phi_fft_filtered[:, ~in_antenna_band] = 0

    antenna_response_theta = interp.interpn(
        (t.theta, t.phi),
        t.leff_theta_reim.swapaxes(0, 2),
        (antenna_perceived_theta[:] * 180 / np.pi, antenna_perceived_phi[:] * 180 / np.pi),
        method='linear'
    )
    antenna_response_phi = interp.interpn(
        (t.theta, t.phi),
        t.leff_phi_reim.swapaxes(0, 2),
        (antenna_perceived_theta[:] * 180 / np.pi, antenna_perceived_phi[:] * 180 / np.pi),
        method='linear'
    )

    antenna_response_theta_interpolator = interp.interp1d(
        t.frequency, antenna_response_theta, axis=1, kind='cubic'
    )
    antenna_response_theta_stretch = antenna_response_theta_interpolator(freqs[in_antenna_band])

    antenna_response_phi_interpolator = interp.interp1d(
        t.frequency, antenna_response_phi, axis=1, kind='cubic'
    )
    antenna_response_phi_stretch = antenna_response_phi_interpolator(freqs[in_antenna_band])

    event_VOC_theta_fft = event_E_theta_fft_filtered.copy()
    event_VOC_theta_fft[:, in_antenna_band] *= antenna_response_theta_stretch

    event_VOC_phi_fft = event_E_phi_fft_filtered.copy()
    event_VOC_phi_fft[:, in_antenna_band] *= antenna_response_phi_stretch

    tot_fft = event_VOC_theta_fft + event_VOC_phi_fft
    return sp.fft.irfft(tot_fft, axis=1), tot_fft


def make_voc(event_E_trace, t, event_pos, event_xmax, N_sample=8192, duration=4.096*1e-6, bp_filter=False):
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
     Sampling_frequency = N_sample/duration
     if (Sampling_frequency/2 > 250*1e6) & bp_filter:
          event_E_trace_filtered = _butter_bandpass_filter(event_E_trace, 50*1e6, 250*1e6, Sampling_frequency)
     else:
         event_E_trace_filtered = event_E_trace
     
     voc, voc_FFT = apply_leff(event_E_trace_filtered, t, event_pos, event_xmax,N_sample=N_sample, duration=duration)
     
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
    s = interpol_at_new_x(s_map[:,0], mag*np.cos(angs) + 1j*mag*np.sin(angs) , target_freqs)
    Z_load = 50 * (1 + s) / (1 - s)
    Z_load = Z_load
    return Z_load

def open_Zant(zant_map, target_freqs, axis=0):
    freqs_in = zant_map[:,0]

    Z_ant = interpol_at_new_x(freqs_in*1e6, zant_map[:,1+2*axis] + 1j*zant_map[:,2+2*axis], target_freqs)
    return Z_ant


def s2abcd(s11, s21, s12, s22):
    """this is a normalized A-matrix represented by [a] in the document."""
    return np.moveaxis(
                np.asarray([
                [((1 + s11) * (1-s22) + s12 * s21) / (2 * s21), ((1 + s11) * (1 + s22) - s12 * s21) / (2 * s21)],
                [((1 - s11) * (1-s22) - s12 * s21) / (2 * s21), ((1 - s11) * (1 + s22) + s12 * s21) / (2 * s21)]
                ], dtype=np.complex128), 
                [0,1],[-2,-1]
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
    freqs_in = s_map[:,0]

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

    amplitudes11 = s_map[:, 1].astype(np.complex128) # is it in dB or not?
    angs11 = np.pi / 180 * (s_map[:, 2].astype(np.complex128)) # angle in deg, converted to rad
    s11 = to_s(amplitudes11, angs11, freqs_in, target_freqs)

    amplitudes21 = s_map[:, 3].astype(np.complex128) # is it in dB or not?
    angs21 = np.pi / 180 * (s_map[:, 4].astype(np.complex128)) # angle in deg, converted to rad
    s21 = to_s(amplitudes21, angs21, freqs_in, target_freqs)

    amplitudes12 = s_map[:, 5].astype(np.complex128) # is it in dB or not?
    angs12 = np.pi / 180 * (s_map[:, 6].astype(np.complex128)) # angle in deg, converted to rad
    s12 = to_s(amplitudes12, angs12, freqs_in, target_freqs)

    amplitudes22 = s_map[:, 7].astype(np.complex128) # is it in dB or not?
    angs22 = np.pi / 180 * (s_map[:, 8].astype(np.complex128)) # angle in deg, converted to rad
    s22 = to_s(amplitudes22, angs22, freqs_in, target_freqs)
    
    ABCD_matrix = s2abcd(s11, s21, s12, s22)

    xy_denorm_factor = np.array([[1, 50], [1/50., 1]])    
    xy_denorm_factor = xy_denorm_factor
    ABCD_matrix *= xy_denorm_factor  

    return ABCD_matrix, (s11, s21, s12, s22)

def total_abcd_matrix(list_abcd, Z_load, balun2_abcd=None):
    total_abcd = reduce(np.matmul, list_abcd)
    Z_in = (total_abcd[:,0,0] * Z_load + total_abcd[:,0,1])/(total_abcd[:,1,0] * Z_load + total_abcd[:,1,1])
    if type(balun2_abcd) is not type(None):
        total_abcd = total_abcd @ balun2_abcd
    return total_abcd, Z_in


def abcd_2_tf(abcd_matrix, Z_ant, Z_in):
    ABCD_inv = np.linalg.inv(abcd_matrix)
    V_I = np.stack([Z_in/ (Z_ant + Z_in),  1/ (Z_ant + Z_in)], axis=-1)
    V_I_out_RFchain = (ABCD_inv @ V_I[:,:,None])[:,:,0]
    return V_I_out_RFchain[:,0]

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
        ABCD_matrix, (s11, s21, s12, s22) = s_file_2_abcd(s_map, target_freqs, db=db)
        list_abcd.append(ABCD_matrix)
    ABCD_matrix_balun2, _ = s_file_2_abcd(balun_2_map, target_freqs, db=False) 

    Z_load = open_Zload(zload_map, target_freqs)
    Z_ant = open_Zant(zant_map, target_freqs, axis=axis)
    ABCD_tot, Z_in = total_abcd_matrix(list_abcd, Z_load, balun2_abcd=ABCD_matrix_balun2)
    tf = abcd_2_tf(ABCD_tot, Z_ant, Z_in)
    return tf

def efield_2_voltage(
    event_trace_fft, in_antenna_band, full_response, target_rate=2e9, current_rate=2e9
):
    """
    Converts electric field data in the frequency domain to voltage data in the time domain.

    Parameters:
        event_trace_fft (numpy.ndarray): A 3D array containing the FFT of the electric field 
            traces. The shape is expected to be (n_events, n_channels, n_frequencies).
        in_antenna_band (slice or array-like): A slice or array specifying the indices of 
            frequencies within the antenna's operational band.
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
    vout_f = np.zeros_like(event_trace_fft)
    vout_fft_inband = np.einsum(
        "ijk,ijlk->ilk", event_trace_fft[:, :, in_antenna_band], full_response
    )
    vout_f[:, :, in_antenna_band] = vout_fft_inband
    vout_f = vout_f.astype(np.complex128)
    ratio = target_rate / current_rate
    m = int((vout_f.shape[-1] - 1) * 2 * ratio)
    return sp.fft.irfft(vout_f, m) * ratio, vout_f
