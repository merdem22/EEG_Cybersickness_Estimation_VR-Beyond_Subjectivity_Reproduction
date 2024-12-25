import os
import pathlib
import torchdatasets as td
from torch.utils.data import ConcatDataset
from .load_video import VideoDataset
from .load_video import video_to_numpy_moviepy
from .load_eeg import EEGDataset


class EEGVideoDataset(td.Dataset):
    def __init__(self, root: str, patient: int, position: str):
        super().__init__()

        root_dir = pathlib.Path(root)
        video_dir = str(root_dir / "footage")
        eeg_dir = str(root_dir / "StudyExport")

        def create_fname(position: str) -> str:
            return f"{eeg_dir}/{patient:04d}/{patient:04d}_{position}_EEG.csv"

        if position[0] == "F":
            fname = "forward.mp4"
        elif position[0] == "S":
            fname = "side.mp4"
        else:
            raise ValueError()

        self.eeg = EEGDataset(eeg_dir, patient=patient, position=position)
        self.vid = VideoDataset(video_dir, patient=patient, filename=fname)

    def __getitem__(self, index):
        return self.eeg[index], self.vid[index]

    def __len__(self):
        return min(len(self.eeg), len(self.vid))


fname = "/home/adhd/src/research/human-computer-interaction/eeg/StudyExport/0003/0003_FR_EEG.csv"


if __name__ == '__main__':
    import numpy as np
    from scipy import signal
    from scipy.fft import fftshift
    rng = np.random.default_rng()


    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    noise_power = 0.01 * fs / 2
    time = np.arange(N) / float(fs)
    mod = 500 * np.cos(2 * np.pi * 0.25 * time)
    carrier = amp * np.sin(2 * np.pi * 3e3 * time + mod)
    noise = rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    noise *= np.exp(-time / 5)
    x = carrier + noise

    x = x[:99900].reshape(333, 300)

    # f, t = 64, 53
    f, t, Sxx = signal.spectrogram(x, fs=fs, nperseg=127, noverlap=50)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec')
    # plt.show()
