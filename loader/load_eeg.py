import pathlib
import torchdatasets as td
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.fft import fft
from scipy.signal import resample
import numpy as np
import os

# Desired columns for the EEG files
desired_columns = [
    # "Millis",  # Uncomment if needed
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "T3",
    "T4",
    "P3",
    "P4",
]


class EEGDataset(td.Dataset):
    def __init__(
        self,
        root: str,
        patient: int,
        position: str,
        window_size: int = 1,
        fs: int = 300,
    ):
        super().__init__()

        # Define file paths
        eeg_path = (
            pathlib.Path(root) / f"{patient:04d}" / f"{patient:04d}_{position}_EEG.csv"
        )
        joystick_path = (
            pathlib.Path(root)
            / f"{patient:04d}"
            / f"{patient:04d}_{position}_SubjectiveCs.csv"
        )

        # Ensure EEG file exists
        assert eeg_path.is_file(), f"{eeg_path} does not exist."

        # Read timestamps
        eeg_timestamp = pd.read_csv(eeg_path)["Millis"]
        joy_timestamp = pd.read_csv(joystick_path)["Millis"]

        # Determine overlapping time range
        min_time = max(eeg_timestamp.min(), joy_timestamp.min())
        max_time = min(eeg_timestamp.max(), joy_timestamp.max())

        # Preprocess EEG and joystick data
        eeg_segments = preprocess_eeg(
            eeg_path,
            min_timestamp=min_time,
            max_timestamp=max_time,
            window_size=window_size,
            fs=fs,
        )
        joystick_segments = preprocess_joystick(
            joystick_path,
            fs=fs,
            window_size=window_size,
            min_timestamp=min_time,
            max_timestamp=max_time,
        )

        # Compute spectrogram for each segment and channel
        # Assuming eeg_segments shape: (n_segments, channels, samples)
        n_segments, channels, samples = eeg_segments.shape
        spectrograms = []
        for segment in eeg_segments:
            segment_spectrogram = []
            for channel_data in segment:
                f, t, Sxx = spectrogram(channel_data, fs=fs, nperseg=127, noverlap=124)
                segment_spectrogram.append(Sxx)
            spectrograms.append(
                np.stack(segment_spectrogram)
            )  # (channels, freq_bins, time_frames)
        self.eeg_spectrogram = np.array(
            spectrograms
        )  # (n_segments, channels, freq_bins, time_frames)

        # Define expected dimensions
        self.num_channels = 8
        self.num_freq_bins = self.eeg_spectrogram.shape[2]
        self.num_time_frames = self.eeg_spectrogram.shape[3]

        # Assertions to ensure data integrity
        assert (
            self.eeg_spectrogram.shape[1] == self.num_channels
        ), "Number of channels mismatch."
        # Additional shape assertions can be added if needed

        # Setup caching (ensure cache directory exists)
        cache_filename = f"{patient:04d}_{position}_EEG"
        cache_dir = pathlib.Path("/tmp/torch-runner/data-cache/juliette-eeg-dataset/")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / cache_filename
        self.cache(td.cachers.Tensor(cache_path))

    def __getitem__(self, index):
        eeg = torch.from_numpy(self.eeg_spectrogram[index])
        return eeg[:, :, : self.num_time_frames].float()

    def __len__(self):
        return len(self.eeg_spectrogram)


# Directory to save the .npz cache files
cache_dir = "/tmp/torch-runner/data-cache"
os.makedirs(cache_dir, exist_ok=True)


# Function to preprocess EEG data
def preprocess_eeg(
    fname: pathlib.Path,
    min_timestamp: int,
    max_timestamp: int,
    window_size: int = 1,
    fs: int = 300,
):
    """
    Splits EEG data into fixed-size windows.

    Parameters:
    - fname: Path to the EEG CSV file.
    - min_timestamp: Minimum timestamp to include.
    - max_timestamp: Maximum timestamp to include.
    - window_size: The size of each segment in seconds.
    - fs: The sampling frequency (in Hz).

    Returns:
    - segments: A 3D numpy array (n_segments x channels x window_length).
    """
    # Read and filter the data
    df = pd.read_csv(fname)
    index = (df["Millis"] >= min_timestamp) & (df["Millis"] <= max_timestamp)
    desired_df = df[index][desired_columns].copy()

    # Normalize each channel
    desired_df = (desired_df - desired_df.mean()) / (desired_df.std() + 1e-5)

    # Assign window indices
    desired_df["Window"] = (df[index]["Millis"] - min_timestamp) // (1000 * window_size)

    # Resample each window
    desired_arrays = [
        resample(DF.drop("Window", axis=1).to_numpy(), num=fs).transpose()
        for _, DF in desired_df.groupby("Window")
    ]

    return np.stack(desired_arrays)


def preprocess_joystick(
    fname: pathlib.Path,
    window_size: int,
    fs: int,
    min_timestamp: int,
    max_timestamp: int,
):
    """
    Preprocess joystick data by resampling and handling missing windows.

    Parameters:
    - fname: Path to the joystick CSV file.
    - window_size: The size of each segment in seconds.
    - fs: The sampling frequency (in Hz).
    - min_timestamp: Minimum timestamp to include.
    - max_timestamp: Maximum timestamp to include.

    Returns:
    - segments: A 3D numpy array (n_segments x 1 x window_length).
    """
    df = pd.read_csv(fname)
    index = (df["Millis"] >= min_timestamp) & (df["Millis"] <= max_timestamp)
    desired_df = df[index][["Rating"]].copy()
    desired_df["Window"] = (df[index]["Millis"] - min_timestamp) // (1000 * window_size)

    desired_arrays = []

    # Identify missing windows
    total_windows = (max_timestamp - min_timestamp) // (1000 * window_size)
    existing_windows = set(desired_df["Window"])
    missing_windows = set(range(total_windows)) - existing_windows

    # Fill missing windows with the last available rating
    for idx in sorted(missing_windows):
        if idx - 1 in existing_windows:
            last_rating = desired_df[desired_df["Window"] == (idx - 1)].iloc[-1]
            filled_row = last_rating.copy()
            filled_row["Window"] = idx
            desired_df = pd.concat(
                [desired_df, filled_row.to_frame().T], ignore_index=True
            )

    # Sort and resample
    desired_df = desired_df.sort_values(by="Window", kind="stable")
    for _, DF in desired_df.groupby("Window"):
        resampled = resample(DF.drop("Window", axis=1).to_numpy(), num=fs).transpose()
        desired_arrays.append(resampled)

    return np.stack(desired_arrays)


if __name__ == "__main__":
    # Configuration
    window_size = 1  # Window size in seconds
    fs = 300  # Sampling frequency in Hz
    root_dir = "/home/adhd/src/research/human-computer-interaction/eeg/StudyExport/"
    patient = 3
    position = "FR"
    eeg_fname = (
        pathlib.Path(root_dir) / f"{patient:04d}" / f"{patient:04d}_{position}_EEG.csv"
    )

    # Initialize the dataset
    ds = EEGDataset(
        root=root_dir,
        patient=patient,
        position=position,
        window_size=window_size,
        fs=fs,
    )

    # Verify dataset is not empty
    if len(ds) == 0:
        print("Dataset is empty. Please check the data files and timestamps.")
    else:
        # Access the first EEG spectrogram
        eeg_tensor = ds[0]  # Shape: (channels, freq_bins, time_frames)
        eeg_numpy = eeg_tensor.numpy()

        # Plot spectrograms for each channel in the first segment
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for i in range(ds.num_channels):
            row, col = divmod(i, 4)
            ax = axes[row, col]
            pcm = ax.pcolormesh(
                np.arange(ds.num_time_frames),  # Time frames
                np.linspace(0, fs / 2, ds.num_freq_bins),  # Frequency axis
                eeg_numpy[i],
                shading="gouraud",
            )
            ax.set_title(desired_columns[i])
            ax.set_ylabel("Frequency [Hz]")
            ax.set_xlabel("Time [sec]")
            fig.colorbar(pcm, ax=ax)
        plt.tight_layout()
        plt.show()

    # Example: Iterate through the dataset and perform additional processing
    # Uncomment and modify as needed
    """
    for segment_idx in range(len(ds)):
        eeg_tensor = ds[segment_idx]  # Shape: (channels, freq_bins, time_frames)
        eeg_numpy = eeg_tensor.numpy()
        # Example FFT processing
        fft_results = np.abs(fft(eeg_numpy, n=53, axis=-1))
        # Further processing...
    """
