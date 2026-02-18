import numpy as np
import scipy.interpolate as interp
from apply_rfchain import kb, c, Z0
import matplotlib.pyplot as plt
from apply_rfchain import plot_quantities


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

class compute_noise():
    def __init__(self, 
                 lst_time_resolution, 
                 detector_lat, 
                 list_temp_files, 
                 LF_freqs, 
                 target_freqs,
                 tf_rfchain,
                 duration,
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
        self.lst_hours = (np.arange(0, 24, lst_time_resolution) + 12) % 24  # Shift LST by 12 hours to correct for the previous error

        self.list_temp_files = list_temp_files
        self.LF_freqs = LF_freqs

        self.long_map, self.lat_map, _ = np.load(list_temp_files[0])


        delta_lat_array = np.diff(self.lat_map, axis=1)
        delta_lat_array = np.concatenate((delta_lat_array, delta_lat_array[:, -1:]), axis=1)

        delta_long_array = np.diff(self.long_map, axis=0)
        delta_long_array = np.concatenate((delta_long_array, delta_long_array[-1:, :]), axis=0)

        self.delta_lat2, self.delta_long2 = delta_lat_array, delta_long_array

        self.delta_lat = np.abs(self.lat_map[0, 0]-self.lat_map[0, 1])
        self.delta_long = np.abs(self.long_map[0, 0]-self.long_map[1, 0])

        assert (type(leff_x) is not type(None)) or (type(leff_y) is not type(None)) or (type(
            leff_z) is not type(None)), "At least one of leff_x, leff_y, leff_z should be provided"

        self.l_eff_theta = leff_x.theta*np.pi/180
        self.l_eff_phi = leff_x.phi*np.pi/180

        self.leff_x_theta_reim_LF = interp.interp1d(leff_x.frequency, leff_x.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_x_phi_reim_LF = interp.interp1d(leff_x.frequency, leff_x.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        self.leff_y_theta_reim_LF = interp.interp1d(leff_y.frequency, leff_y.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_y_phi_reim_LF = interp.interp1d(leff_y.frequency, leff_y.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        self.leff_z_theta_reim_LF = interp.interp1d(leff_z.frequency, leff_z.leff_theta_reim,
                                                 axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)
        self.leff_z_phi_reim_LF = interp.interp1d(leff_z.frequency, leff_z.leff_phi_reim,
                                               axis=0, kind='linear', bounds_error=False, fill_value=0)(self.LF_freqs)

        n_freqs = tf_rfchain.shape[-1]
        self.tf_LF = interp.interp1d(np.linspace(0, (n_freqs-1)/duration, n_freqs), tf_rfchain, axis=1, kind='quadratic', bounds_error=False, fill_value=0
        )(LF_freqs)
        self.tf_target = interp.interp1d(np.linspace(0, (n_freqs-1)/duration, n_freqs), tf_rfchain, axis=1, kind='quadratic', bounds_error=False, fill_value=0
        )(target_freqs)
        self.target_freqs = target_freqs
        
    @property
    def lst_rads(self):
        return self.lst_hours / 24 * 2 * np.pi

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
        return latlon2zenaz(self.detector_lat, lst_rad, self.lat_map, self.long_map, mod_pi=mod_pi, add_pi=add_pi)

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
$            - `get_temp_map`: Retrieves the temperature map for a given frequency index.
            - `interp.interpn`: Performs multi-dimensional interpolation.
            - Constants: `kb` (Boltzmann constant), `c` (speed of light).

        """
        P_nuxyz = np.zeros((len(self.lst_rads), 3, len(self.LF_freqs)))
        for lst_idx, lst_rad in enumerate(self.lst_rads[:]):
            print(
                f"Calculating noise power for LST {lst_rad*12/np.pi:.2f} hours")
            # all_zenith, all_azimuth = self.latlon2zenaz((lst_rad + np.pi)%(2*np.pi), mod_pi=True)
            all_zenith, all_azimuth = self.latlon2zenaz(lst_rad, mod_pi=True)

            for coord_idx, l_effs in enumerate([(self.leff_x_theta_reim_LF, self.leff_x_phi_reim_LF),
                                                (self.leff_y_theta_reim_LF, self.leff_y_phi_reim_LF),
                                                (self.leff_z_theta_reim_LF, self.leff_z_phi_reim_LF)]):
                if type(l_effs) is type(None):
                    continue
                complete_l_eff_phi = np.linspace(0, 2 * np.pi, 361, endpoint=True)
                leff_theta, leff_phi = l_effs
                leff_theta = np.rollaxis(leff_theta, 0, leff_theta.ndim)
                leff_phi = np.rollaxis(leff_phi, 0, leff_phi.ndim)
                leff_theta = interp.interp1d(self.l_eff_phi, leff_theta,
                                             axis=0, kind='cubic', fill_value='extrapolate')(complete_l_eff_phi)  #Interpolating leff at new azimuths
                leff_phi = interp.interp1d(self.l_eff_phi, leff_phi,
                                           axis=0, kind='cubic', fill_value='extrapolate')(complete_l_eff_phi)  #Interpolating leff at new azimuths
                leff_theta_interpolated_dir = interp.interpn((complete_l_eff_phi, self.l_eff_theta),
                                                         leff_theta,
                                                         (all_azimuth, all_zenith),
                                                         bounds_error=False, fill_value=0)  #Interpolating leff at new directions


                leff_phi_interpolated_dir = interp.interpn((complete_l_eff_phi, self.l_eff_theta),
                                                       leff_phi,
                                                       (all_azimuth, all_zenith),
                                                       bounds_error=False, fill_value=0)  #Interpolating leff at new directions
                
                leff_theta_interpolated_dir = np.rollaxis(
                    leff_theta_interpolated_dir, -1, 0)
                leff_phi_interpolated_dir = np.rollaxis(
                    leff_phi_interpolated_dir, -1, 0)

                A_eff = np.abs(leff_theta_interpolated_dir) ** 2 + \
                    np.abs(leff_phi_interpolated_dir)**2
                
                for freq_idx, freq in enumerate(self.LF_freqs):
                    temp_map = self.get_temp_map(freq_idx)
                    B_nu = 2 * (freq)**2 * kb * temp_map/(c**2)
                    if (np.abs(freq - 95e6) < 10) and ((lst_rad*12/np.pi) % 12 == 6) and plot:
                        plot_quantities(lst_rad, freq_idx, all_zenith, all_azimuth,
                                        leff_theta_interpolated_dir, leff_phi_interpolated_dir, A_eff, B_nu, self.long_map, self.lat_map, self.detector_lat)
                        plt.show()

                    P_nu = 1/2 * \
                        A_eff[freq_idx] * B_nu * \
                        np.sin(self.lat_map) * self.delta_lat2 * self.delta_long2
                    P_nu = np.sum(P_nu)
                    P_nuxyz[lst_idx, coord_idx, freq_idx] = P_nu

        P_nuxyz = np.array(P_nuxyz)
        self._P_nu = P_nuxyz   # W/Hz
        return P_nuxyz
    
    @property
    def P_nu(self):
        """
        Returns the noise power in frequency domains.
        """
        if not hasattr(self, '_P_nu'):
            return self.noise_power()  #*2 # Artificial factor. Should need to be removed
        return self._P_nu

    def Voc_psd(self):
        return  self.P_nu * Z0   ## V^2/Hz poutr les 221 frequqnce de LFmap
    def Vout_psd(self):
        return self.Voc_psd() * (np.abs(self.tf_LF) * np.abs(self.tf_LF))
    
    def noise_fourrier_traces(self):
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
        self.noise_variance = self.Vout_psd()
        
        self.noise_variance = interp.interp1d(self.LF_freqs, self.Vout_psd(), 
                                              bounds_error=False, fill_value=0, axis=-1)(self.target_freqs)
        self._noise_fourrier_spectrum = np.sqrt(self.noise_variance * N * fs / 2)  # V
        self._noise_fourrier_spectrum_sqrtfreq = np.sqrt(self.noise_variance)  # V
        return self._noise_fourrier_spectrum

    @property
    def noise_fourrier_spectrum(self):
        if hasattr(self, '_noise_fourrier_spectrum'):
            return self._noise_fourrier_spectrum
        elif hasattr(self, 'path_to_noise_spectrum'):
            self._noise_fourrier_spectrum = np.load(self.path_to_noise_spectrum)
            self.LF_freqs = np.load(self.path_to_noise_spectrum.replace(
                '.npy', '_frequencies.npy'))
            self.lst_hours = np.load(self.path_to_noise_spectrum.replace(
                '.npy', '_lsthours.npy'))
            return self._noise_fourrier_spectrum
        return self.noise_fourrier_traces()

    def save_spectrum(self, directory='.', name='noise_fourrier_spectrum'):
        """
        Save the noise spectrum to a file.

        Args:
            directory (str): Directory where the file will be saved.
            name (str): Name of the file (without extension).
        """
        if not (hasattr(self, 'noise_fourrier_spectrum')):
            raise ValueError(
                "Noise spectrum not computed. Run noise_rms_traces(target_frqs, tf_rfchain) first.")
        np.save(f'{directory}/{name}.npy', self.noise_fourrier_spectrum)
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
        
        amp = rng.normal(loc=0, scale=self.noise_fourrier_spectrum[lst_idx], size=(
            n_samples, 3, len(self.target_freqs)))
        phase = 2 * np.pi * rng.random(size=(n_samples, 3, n_freqs))
        v_complex_fft = amp * np.exp(1j*phase)
        v_noise = np.fft.irfft(v_complex_fft, axis=-1)
        if micro:
            v_noise *= 1e6
            v_complex_fft *= 1e6
        return v_noise, v_complex_fft


def add_jitter(du_ns, sigma=5, seed=None):
    """
    Add Gaussian jitter to the given time series.

    Parameters:
        du_ns (numpy.ndarray): The time series data to which jitter will be added in ns.
        sigma (float): The standard deviation of the Gaussian noise to be added in ns.
        sample_rate (float): The sampling rate of the time series data in Hz.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        numpy.ndarray: The time series with added Gaussian jitter.
    """
    rng = np.random.default_rng(seed)
    jitter_time = np.round(rng.normal(0, sigma, size=du_ns.shape)).astype(du_ns.dtype) #jittered time in whole nanoseconds
    return du_ns + jitter_time

def add_smearing(amplitudes, sigma=7):
    """
    Add Gaussian smearing to the given amplitude data.

    Parameters:
        amplitudes (numpy.ndarray): The amplitude data to which smearing will be added.
        sigma (float): The standard deviation of the Gaussian noise to be added.
    
    Returns:
        numpy.ndarray: The amplitude data with added Gaussian smearing.
    """
    rng = np.random.default_rng()
    noise = rng.normal(1, sigma/100, size=amplitudes.shape[0])
    return amplitudes * noise