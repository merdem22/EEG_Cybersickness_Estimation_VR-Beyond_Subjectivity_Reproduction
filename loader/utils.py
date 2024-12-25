import numpy as np
from scipy.signal import spectrogram


def compute_spectrogram(segment, fs):
    """
    Compute the spectrogram of a single EEG segment for all channels.

    Parameters:
    - segment: 2D NumPy array of EEG data (channels x time) for one event.
    - fs: Sampling frequency (in Hz).

    Returns:
    - spec_data: 3D NumPy array of spectrogram data (channels x frequency bins x time frames).
    """
    n_channels, n_samples = segment.shape
    spec_data = []

    # Loop over each channel to compute the spectrogram
    for ch in range(n_channels):
        import pdb

        pdb.set_trace()
        f, t, Sxx = spectrogram(segment[ch], fs=fs, nperseg=127, noverlap=64)

        # Take the magnitude of the spectrogram and log-transform it
        Sxx = np.log10(Sxx + 1e-10)  # Log transformation for stability

        # Ensure that the frequency bins and time frames match the required shape
        Sxx_resized = Sxx[:64, :53]  # Keep 64 frequency bins and 53 time frames
        print(segment[ch].shape, Sxx.shape)

        spec_data.append(Sxx_resized)

    # Stack all channels together into a 3D array (channels x frequency bins x time frames)
    return np.stack(spec_data, axis=0)


def process_eeg_to_spectrogram(eeg_segments, fs):
    """
    Process EEG segments and convert them to spectrograms of shape (8, 53, 64).

    Parameters:
    - eeg_segments: 3D NumPy array of segmented EEG data (n_events x n_channels x segment_length).
    - fs: Sampling frequency (in Hz).

    Returns:
    - spectrogram_data: 4D NumPy array (n_events x 8 x 53 x 64).
    """
    all_spectrograms = []

    # Loop over each EEG segment (one per event)
    for segment in eeg_segments:
        spec = compute_spectrogram(segment, fs=fs)
        all_spectrograms.append(spec)

    # Return the spectrograms as a 4D array (n_events x channels x 53 x 64)
    return np.array(all_spectrograms)


# Example usage:
# Assume eeg_segments is a 3D NumPy array of segmented EEG data (n_events x 8 x segment_length)
fs = 300  # Sampling frequency in Hz

# Example segmented EEG data (replace this with actual segmented data)
n_events = 133  # Number of events (for example)
segment_length = 1 * fs  # Length of each EEG segment in samples
eeg_segments = np.random.randn(n_events, 8, segment_length)  # Simulated data

# Process the EEG data to obtain the spectrograms of shape (n_events, 8, 53, 64)
spectrogram_data = process_eeg_to_spectrogram(eeg_segments, fs)
print(f"Spectrogram data shape: {spectrogram_data.shape}")
