"""
This module provides functions for analyzing waveform data that has made it through the first pass filter.
The waveform data is expected to be a list of floating point [pressure, time] pairs.
Returns the unweighted mean of the most stable section of the waveform.
"""

import numpy as np

def analyze_waveform(waveform, blanking_time=0.1, quantize_mode="round"):
    """
    Analyzes a waveform to count pressure occurrences with a blanking time.

    Args:
        waveform (list): A list of [pressure, time] pairs.
        blanking_time (float): The time in seconds to ignore subsequent pressures.
        quantize_mode (str): The quantization mode, either "round" or "floor".

    Returns:
        np.ndarray: An array of counts for each integer pressure value.
    """
    if not waveform:
        return np.array([])

    # Find the maximum pressure to determine the size of the bins array.
    # We assume pressures are non-negative.
    max_pressure = 0
    if waveform:
        if quantize_mode == "round":
            max_pressure = int(np.round(max(p for p, t in waveform))) + 1
        else:
            max_pressure = int(max(p for p, t in waveform)) + 1


    bins = np.zeros(max_pressure, dtype=int)
    last_time = -blanking_time  # Initialize to allow the first event to be recorded.

    for pressure, time in waveform:
        if time - last_time >= blanking_time:
            if quantize_mode == "round":
                pressure_int = int(np.round(pressure))
            elif quantize_mode == "floor":
                pressure_int = int(pressure)
            else:
                raise ValueError("quantize_mode must be 'round' or 'floor'")

            if 0 <= pressure_int < max_pressure:
                bins[pressure_int] += 1
            last_time = time

    return bins


from collections import Counter

def find_pa_mean(bins):
    """
    Finds a representative pressure from a bin array by finding the mode of the counts.

    Args:
        bins (np.ndarray): An array of counts for each integer pressure value.

    Returns:
        float: The median of the bin indices that match the modal count.
               Returns None if there are no non-zero counts.
    """
    non_zero_counts = bins[bins > 0]
    if non_zero_counts.size == 0:
        return None

    # Find the mode of all non-zero counts
    count_frequencies = Counter(non_zero_counts)
    max_freq = max(count_frequencies.values())
    modes = [count for count, freq in count_frequencies.items() if freq == max_freq]
    modal_count = max(modes)

    # Select all bins from the original array that have the modal count
    final_bin_indices = np.where(bins == modal_count)[0]

    if final_bin_indices.size == 0:
        # This case should ideally not be reached if non_zero_counts is not empty
        return None

    # Return the median of the resulting bin indices
    return np.median(final_bin_indices)
