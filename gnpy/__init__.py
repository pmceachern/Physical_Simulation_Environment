import numpy as np

"""
GNPy: a Python 3 implementation of the Gaussian Noise (GN) Model of nonlinear
propagation, developed by the OptCom group, Department of Electronics and
Telecommunications, Politecnico di Torino, Italy
"""

__credits__ = ["Mattia Cantono", "Vittorio Curri", "Alessio Ferrari"]


def raised_cosine_comb(f, rs, roll_off, center_freq, power):
    """ Returns an array storing the PSD of a WDM comb of raised cosine shaped
    channels at the input frequencies defined in array f

    :param f: Array of frequencies in THz
    :param rs: Array of Symbol Rates in TBaud. One Symbol rate for each channel
    :param roll_off: Array of roll-off factors [0,1). One per channel
    :param center_freq: Array of channels central frequencies in THz. One per channel
    :param power: Array of channel powers in W. One per channel
    :return: PSD of the WDM comb evaluated over f
    """
    ts_arr = 1 / rs
    passband_arr = (1 - roll_off) / (2 * ts_arr)
    stopband_arr = (1 + roll_off) / (2 * ts_arr)
    g = power / rs
    psd = np.zeros(np.shape(f))
    for ind in range(np.size(center_freq)):
        f_nch = center_freq[ind]
        g_ch = g[ind]
        ts = ts_arr[ind]
        passband = passband_arr[ind]
        stopband = stopband_arr[ind]
        ff = np.abs(f - f_nch)
        tf = ff - passband
        if roll_off[ind] == 0:
            psd = np.where(tf <= 0, g_ch, 0.) + psd
        else:
            psd = g_ch * (np.where(tf <= 0, 1., 0.) + 1 / 2 * (1 + np.cos(np.pi * ts / roll_off[ind] *
                                                                          tf)) * np.where(tf > 0, 1., 0.) *
                          np.where(np.abs(ff) <= stopband, 1., 0.)) + psd

    return psd


def fwm_eff(a, Lspan, b2, ff):
    """ Computes the four-wave mixing efficiency given the fiber characteristics
    over a given frequency set ff
    :param a: Fiber loss coefficient in 1/km
    :param Lspan: Fiber length in km
    :param b2: Fiber Dispersion coefficient in ps/THz/km
    :param ff: Array of Frequency points in THz
    :return: FWM efficiency rho
    """
    rho = np.power(np.abs((1 - np.exp(-2 * a * Lspan + 1j * 4 * np.pi * np.pi * b2 * Lspan * ff)) / (
        2 * a - 1j * 4 * np.pi * np.pi * b2 * ff)), 2)
    return rho


def get_freqarray(f, Bopt, fmax, max_step, f_dense_low, f_dense_up, df_dense):
    """ Returns a non-uniformly spaced frequency array useful for fast GN-model.
    integration. The frequency array is made of a denser area, sided by two
    log-spaced arrays
    :param f: Central frequency at which NLI is evaluated in THz
    :param Bopt: Total optical bandwidth of the system in THz
    :param fmax: Upper limit of the integration domain in THz
    :param max_step: Maximum step size for frequency array definition in THz
    :param f_dense_low: Lower limit of denser frequency region in THz
    :param f_dense_up: Upper limit of denser frequency region in THz
    :param df_dense: Step size to be used in the denser frequency region in THz
    :return: Non uniformly defined frequency array
    """
    f_dense = np.arange(f_dense_low, f_dense_up, df_dense)
    k = Bopt / 2 / (Bopt / 2 - max_step)  # Compute Step ratio for log-spaced array definition
    if f < 0:
        Nlog_short = np.ceil(np.log(fmax / np.abs(f_dense_low)) / np.log(k) + 1)
        f1_short = -(np.abs(f_dense_low) * np.power(k, np.arange(Nlog_short, 0.0, -1.0) - 1.0))
        k = (Bopt / 2 + (np.abs(f_dense_up) - f_dense_low)) / (Bopt / 2 - max_step + (np.abs(f_dense_up) - f_dense_up))
        Nlog_long = np.ceil(np.log((fmax + (np.abs(f_dense_up) - f_dense_up)) / abs(f_dense_up)) * 1 / np.log(k) + 1)
        f1_long = np.abs(f_dense_up) * np.power(k, (np.arange(1, Nlog_long + 1) - 1)) - (
            np.abs(f_dense_up) - f_dense_up)
        f1_array = np.concatenate([f1_short, f_dense[1:], f1_long])
    else:
        Nlog_short = np.ceil(np.log(fmax / np.abs(f_dense_up)) / np.log(k) + 1)
        f1_short = f_dense_up * np.power(k, np.arange(1, Nlog_short + 1) - 1)
        k = (Bopt / 2 + (abs(f_dense_low) + f_dense_low)) / (Bopt / 2 - max_step + (abs(f_dense_low) + f_dense_low))
        Nlog_long = np.ceil(np.log((fmax + (np.abs(f_dense_low) + f_dense_low)) / np.abs(f_dense_low)) / np.log(k) + 1)
        f1_long = -(np.abs(f_dense_low) * np.power(k, np.arange(Nlog_long, 0, -1) - 1)) + (
            abs(f_dense_low) + f_dense_low)
        f1_array = np.concatenate([f1_long, f_dense[1:], f1_short])
    return f1_array


def GN_integral(b2, Lspan, a_dB, gam, f_ch, rs, roll_off, power, Nch, model_param):
    """ GN_integral computes the GN reference formula via smart brute force integration. The Gaussian Noise model is
    applied in its incoherent form (phased-array factor =1). The function computes the integral by columns: for each f1,
    a non-uniformly spaced f2 array is generated, and the integrand function is computed there. At the end of the loop
    on f1, the overall GNLI is computed. Accuracy can be tuned by operating on model_param argument.

    :param b2: Fiber dispersion coefficient in ps/THz/km. Scalar
    :param Lspan: Fiber Span length in km. Scalar
    :param a_dB: Fiber loss coeffiecient in dB/km. Scalar
    :param gam: Fiber nonlinear coefficient in 1/W/km. Scalar
    :param f_ch: Baseband channels center frequencies in THz. Array of size 1xNch
    :param rs: Channels' Symbol Rates in TBaud. Array of size 1xNch
    :param roll_off: Channels' Roll-off factors [0,1). Array of size 1xNch
    :param power: Channels' power values in W. Array of size 1xNch
    :param Nch: Number of channels. Scalar
    :param model_param: Dictionary with model parameters for accuracy tuning
                        model_param['min_FWM_inv']: Minimum FWM efficiency value to be considered for high density
                        integration in dB
                        model_param['n_grid']: Maximum Number of integration points to be used in each frequency slot of
                        the spectrum
                        model_param['n_grid_min']: Minimum Number of integration points to be used in each frequency
                        slot of the spectrum
                        model_param['f_array']: Frequencies at which evaluate GNLI, expressed in THz
    :return: GNLI: power spectral density in W/THz of the nonlinear interference at frequencies model_param['f_array']
    """
    alpha_lin = a_dB / 20.0 / np.log10(np.e)  # Conversion in linear units 1/km
    min_FWM_inv = np.power(10, model_param['min_FWM_inv'] / 10)  # Conversion in linear units
    n_grid = model_param['n_grid']
    n_grid_min = model_param['n_grid_min']
    f_array = model_param['f_array']
    fmax = (f_ch[-1] - (rs[-1] / 2)) - (f_ch[0] - (rs[0] / 2))  # Get frequency limit
    f2eval = np.max(np.diff(f_ch))
    Bopt = f2eval * Nch  # Overall optical bandwidth [THz]
    min_step = f2eval / n_grid  # Minimum integration step
    max_step = f2eval / n_grid_min  # Maximum integration step
    f_dense_start = np.abs(
        np.sqrt(np.power(alpha_lin, 2) / (4 * np.power(np.pi, 4) * b2 * b2) * (min_FWM_inv - 1)) / f2eval)
    f_ind_eval = 0
    GNLI = np.full(f_array.size, np.nan)  # Pre-allocate results
    for f in f_array:  # Loop over f
        f_dense_low = f - f_dense_start
        f_dense_up = f + f_dense_start
        if f_dense_low < -fmax:
            f_dense_low = -fmax
        if f_dense_low == 0.0:
            f_dense_low = -min_step
        if f_dense_up == 0.0:
            f_dense_up = min_step
        if f_dense_up > fmax:
            f_dense_up = fmax
        f_dense_width = np.abs(f_dense_up - f_dense_low)
        n_grid_dense = np.ceil(f_dense_width / min_step)
        df = f_dense_width / n_grid_dense
        # Get non-uniformly spaced f1 array
        f1_array = get_freqarray(f, Bopt, fmax, max_step, f_dense_low, f_dense_up, df)
        G1 = raised_cosine_comb(f1_array, rs, roll_off, f_ch, power)  # Get corresponding spectrum
        Gpart = np.zeros(f1_array.size)  # Pre-allocate partial result for inner integral
        f_ind = 0
        for f1 in f1_array:  # Loop over f1
            if f1 != f:
                f_lim = np.sqrt(np.power(alpha_lin, 2) / (4 * np.power(np.pi, 4) * b2 * b2) * (min_FWM_inv - 1)) / (
                    f1 - f) + f
                f2_dense_up = np.maximum(f_lim, -f_lim)
                f2_dense_low = np.minimum(f_lim, -f_lim)
                if f2_dense_low == 0:
                    f2_dense_low = -min_step
                if f2_dense_up == 0:
                    f2_dense_up = min_step
                if f2_dense_low < -fmax:
                    f2_dense_low = -fmax
                if f2_dense_up > fmax:
                    f2_dense_up = fmax
            else:
                f2_dense_up = fmax
                f2_dense_low = -fmax
            f2_dense_width = np.abs(f2_dense_up - f2_dense_low)
            n2_grid_dense = np.ceil(f2_dense_width / min_step)
            df2 = f2_dense_width / n2_grid_dense
            # Get non-uniformly spaced f2 array
            f2_array = get_freqarray(f, Bopt, fmax, max_step, f2_dense_low, f2_dense_up, df2)
            f2_array = f2_array[f2_array >= f1]  # Do not consider points below the bisector of quadrants I and III
            if f2_array.size > 0:
                G2 = raised_cosine_comb(f2_array, rs, roll_off, f_ch, power)  # Get spectrum there
                f3_array = f1 + f2_array - f  # Compute f3
                G3 = raised_cosine_comb(f3_array, rs, roll_off, f_ch, power)  # Get spectrum over f3
                G = G2 * G3 * G1[f_ind]
                if np.count_nonzero(G):
                    FWM_eff = fwm_eff(alpha_lin, Lspan, b2, (f1 - f) * (f2_array - f))  # Compute FWM efficiency
                    Gpart[f_ind] = 2 * np.trapz(FWM_eff * G, f2_array)  # Compute inner integral
            f_ind += 1
            # Compute outer integral. Nominal span loss already compensated
        GNLI[f_ind_eval] = 16 / 27 * gam * gam * np.trapz(Gpart, f1_array)
        f_ind_eval += 1  # Next frequency
    return GNLI  # Return GNLI array in W/THz
