import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy import interpolate
def spike_removal(y, width_threshold, prominence_threshold, moving_average_window=10,
                  width_param_rel=0.8, interp_kind='linear'):
    """
    Detects and replaces spikes in the input spectrum signal with interpolated values.
    
    Based on the publication by N. Coca-Lopez "An intuitive approach for spike removal in Raman spectra 
    based on peaks’ prominence and width" https://doi.org/10.1016/j.aca.2024.342312

    Parameters:
    y (numpy.ndarray): Input signal array.
    width_threshold (float): Threshold for peak width.
    prominence_threshold (float): Threshold for peak prominence.
    moving_average_window (int): Number of points in moving average window.
    width_param_rel (float): Relative height parameter for peak width.
    interp_kind (str): Type of interpolation ('linear', 'quadratic', 'cubic').

    Returns:
    numpy.ndarray: Signal with spikes replaced by interpolated values.
    """
    peaks, _ = find_peaks(y, prominence=prominence_threshold)
    spikes = np.zeros(len(y))
    widths = peak_widths(y, peaks)[0]
    widths_left_end, widths_right_end = peak_widths(y, peaks, rel_height=width_param_rel)[2:4]
    for a, width, ext_a, ext_b in zip(range(len(widths)), widths, widths_left_end, widths_right_end):
        if width < width_threshold:
            spikes[int(ext_a) - 1:int(ext_b) + 2] = 1
    y_out = y.copy()
    for i, spike in enumerate(spikes):
        if spike:
            window = np.arange(max(i - moving_average_window, 0), min(i + moving_average_window + 1, len(y)))
            window_exclude_spikes = window[spikes[window] == 0]
            interpolator = interpolate.interp1d(window_exclude_spikes, y[window_exclude_spikes], kind=interp_kind)
            y_out[i] = interpolator(i)
    return y_out
