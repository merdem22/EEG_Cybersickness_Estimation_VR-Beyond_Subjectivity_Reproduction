import pathlib
import torchdatasets as td
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft
import numpy as np
import os


# Desired columns for the EEG files
desired_columns = [
    # "Millis",
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "T3",
    "T4",
    "P3",
    "P4",
]


class JoystickDataset(td.Dataset):
    def __init__(
        self,
        root: str,
        patient: int,
        position: str,
        window_size: int = 1,
        fs: int = 300,
    ):
        super().__init__()

        filename = f"{patient:04d}_{position}_EEG.csv"
        fname = pathlib.Path(root) / f"{patient:04d}" / filename
        assert os.path.isfile(fname), f"{fname} is not a file"
        assert filename.endswith("_EEG.csv"), f"{fname} is not a EEG csv file"

        eeg_data = preprocess_eeg(str(fname))
        eeg_segments = segment_eeg(eeg_data, window_size, fs)
        f, t, Sxx = eeg_spectrogram = spectrogram(
            eeg_segments, fs=fs, nperseg=127, noverlap=124
        )
        self.eeg_spectogram = Sxx
        self.num_channels = 8
        self.num_freq_bins = 64
        self.num_time_frames = 53
        assert self.eeg_spectogram.shape[1] == self.num_channels
        assert self.eeg_spectogram.shape[2] == self.num_freq_bins
        cache_filename = f"{patient:04d}_{position}_EEG"
        cache = "/tmp/cache/juliette-eeg-dataset/" + cache_filename
        # import pdb; pdb.set_trace()
        self.cache(td.cachers.Tensor(pathlib.Path(cache)))

    def __getitem__(self, index):
        eeg = torch.from_numpy(self.eeg_spectogram[index])
        return eeg[:, :, : self.num_time_frames].float()

    def __len__(self):
        return len(self.eeg_spectogram)


# Directory to save the .npz cache files
cache_dir = "/tmp/data/torch_runner/data-cache"
os.makedirs(cache_dir, exist_ok=True)


# Function to preprocess EEG data
def preprocess_eeg(fname: str):
    # Rename columns
    desired_df = pd.read_csv(fname)[desired_columns]

    # Normalize each channel (excluding 'Millis' and 'Hardware')
    desired_df = (desired_df - desired_df.mean()) / (desired_df.std() + 1e-5)

    return desired_df.to_numpy().transpose()


def segment_eeg(data, window_size, fs):
    """
    Splits EEG data into fixed-size windows.

    Parameters:
    - data: A 2D numpy array (channels x time).
    - window_size: The size of each segment in seconds.
    - fs: The sampling frequency (in Hz).

    Returns:
    - segments: A 3D numpy array (n_segments x channels x window_length).
    """
    window_length = window_size * fs  # Convert window size to samples
    n_samples = data.shape[1]
    n_channels = data.shape[0]

    # Calculate number of windows
    n_windows = n_samples // window_length

    # Reshape into (n_windows x channels x window_length)
    segments = np.split(data[:, : n_windows * window_length], n_windows, axis=1)

    return np.array(segments)


if __name__ == "__main__":
    # Example usage:
    # Assuming `eeg_data` is a 2D numpy array of shape (channels, time) and fs is 250Hz
    # eeg_data = np.random.randn(8, 3450)  # Example EEG data (replace with actual data)
    window_size = 1  # Window size in seconds
    fs = 300  # Sampling frequency in Hz
    fname = "/home/adhd/src/research/human-computer-interaction/eeg/StudyExport/0003/0003_FR_EEG.csv"
    ds = EEGDataset(
        "/home/adhd/src/research/human-computer-interaction/eeg/StudyExport/", 3, "FR"
    )

    eeg_data = preprocess_eeg(fname)
    eeg_segments = segment_eeg(eeg_data, window_size, fs)
    f, t, Sxx = eeg_spectrogram = spectrogram(
        eeg_segments, fs=fs, nperseg=127, noverlap=124
    )

    fix, axes = plt.subplots(2, 4, figsize=(12, 48))
    for i in range(8):
        idx = i // 4, i % 4
        pcm = axes[idx].pcolormesh(t, f, Sxx[0, i], shading="gouraud")

        axes[idx].set_title(desired_columns[i])
        axes[idx].set_ylabel("Frequency [Hz]")
        axes[idx].set_xlabel("Time [sec")

        # Add the colorbar to the current axis
        plt.colorbar(pcm, ax=axes[idx])
    # plt.tight_layout()
    plt.show()

    """
    reshaped_eeg_data = eeg_segments.transpose(1,2,0)

    for segment_idx in range(reshaped_eeg_data.shape[1]):
        segment_data = reshaped_eeg_data[:, segment_idx, :]

        fft_result = np.abs(fft(segment_data, n=53, axis=-1))
    """
