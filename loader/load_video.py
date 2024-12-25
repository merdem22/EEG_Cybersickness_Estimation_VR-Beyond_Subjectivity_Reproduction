import pdb
from moviepy.editor import VideoFileClip
from PIL import Image
import numpy as np
import pathlib
import os
from torch.utils.data import Dataset
import torch
import torchdatasets as td


class VideoDataset(td.Dataset):
    def __init__(self, root: str, patient: int, filename: str):
        fname = pathlib.Path(root) / f"Take{patient:02d}" / filename
        assert os.path.isfile(fname), f"{fname} is not directory."
        assert filename.endswith(".mp4"), f"{fname} is not mp4 file."

        super().__init__()

        self.root = root
        self.patient = patient
        self.filename = filename
        self.frames = video_to_numpy_moviepy(str(fname))
        cache_filename = f"Take{patient:02d}-{filename[:-4]}"
        cache = "/tmp/cache/juliette-eeg-dataset/" + cache_filename
        self.cache(td.cachers.Tensor(pathlib.Path(cache)))

    def __getitem__(self, index):
        return torch.from_numpy(self.frames[index]) / 255

    def __len__(self):
        return len(self.frames)


def video_to_numpy_moviepy(video_path):
    """
    Convert an MP4 video to a NumPy array using MoviePy.

    Parameters:
    - video_path: Path to the video file (MP4).

    Returns:
    - frames_array: A NumPy array containing the video frames (frames x height x width x channels).
    """
    # Load video using MoviePy
    clip = VideoFileClip(video_path)
    frame_size = int(clip.fps)
    num_frames = int(clip.duration)

    # Extract frames from the video
    frames = []
    for i, frame in enumerate(clip.iter_frames()):
        frame = Image.fromarray(frame)
        resized_frame = frame.resize((224, 224), Image.BILINEAR)
        resized_frame = np.array(resized_frame)
        resized_frame = resized_frame.transpose(2, 0, 1)
        frames.append(resized_frame)

    # Convert list of frames to a NumPy array
    frames_array = np.array(frames)

    clipped_array = frames_array[: frame_size * num_frames]
    return clipped_array.reshape((num_frames, frame_size, 3, 224, 224))


if __name__ == "__main__":
    # Example usage
    video_path = "/home/adhd/data/juliette-eeg/footage/Take03/side.mp4"  # Replace with your actual video path
    video_array = video_to_numpy_moviepy(video_path)
    print(f"Video shape: {video_array.shape}")
