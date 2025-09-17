# preprocessing.py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
import os
from mne.time_frequency import psd_array_multitaper


# ---------------------------
# EEG preprocessing (aligns with repo helpers)
# ---------------------------
def preprocess_eeg(eeg_data,
                   orig_fs: int = 300,
                   target_fs: int = 100,
                   bp_lower: float = 1.0,
                   bp_upper: float = 40.0,
                   notch_freq: float = 50.0,
                   notch_q: float = 5.0):
    """
    Resample + band-pass + notch. (No extra centering/z-scoring; matches repo style.)

    Args
    ----
    eeg_data : (n_samples, n_channels) array at orig_fs
    orig_fs  : original sampling rate (Hz)
    target_fs: target sampling rate (Hz)
    bp_lower, bp_upper: band-pass cutoffs (Hz)
    notch_freq: notch center freq (Hz)
    notch_q   : notch quality factor (repo uses small Q ~5)

    Returns
    -------
    eeg_out : (n_samples_new, n_channels) at target_fs
    """
    # 1) resample (polyphase; stable vs FFT)
    seg = resample_poly(eeg_data, up=target_fs, down=orig_fs, axis=0)

    # 2) band-pass @ target_fs
    nyq = target_fs / 2.0
    b_bp, a_bp = butter(4, [bp_lower / nyq, bp_upper / nyq], btype='band')
    seg = filtfilt(b_bp, a_bp, seg, axis=0)

    # 3) 50 Hz notch (Q ~ 5 like repo)
    w0 = notch_freq / nyq
    b_notch, a_notch = iirnotch(w0, notch_q)
    seg = filtfilt(b_notch, a_notch, seg, axis=0)

    return seg


# ---------------------------
# Time alignment helpers
# ---------------------------
def find_common_span(eeg_millis, cs_millis):
    """
    Largest overlapping time span (inclusive) shared by EEG & CS timestamps.
    Returns (start_millis, end_millis) or (None, None) if no overlap.
    """
    if len(eeg_millis) == 0 or len(cs_millis) == 0:
        return None, None
    start = max(np.min(eeg_millis), np.min(cs_millis))
    end   = min(np.max(eeg_millis), np.max(cs_millis))
    if start > end:
        return None, None
    return int(start), int(end)


def match_cs_to_eeg(eeg_millis, cs_millis, cs_ratings):
    """
    Last-observation-carried-forward mapping from CS → EEG timestamps.

    Returns
    -------
    matched_cs : array(len(eeg_millis),)
    """
    matched_cs = np.zeros(len(eeg_millis), dtype=float)
    for i, t in enumerate(eeg_millis):
        idx = np.searchsorted(cs_millis, t, side='right') - 1
        if idx < 0:
            idx = 0  # clamp to first rating if EEG starts before first CS
        matched_cs[i] = cs_ratings[idx]
    return matched_cs


# ---------------------------
# Repo-style CS target construction
# ---------------------------
def compute_cs_targets(cs_values, threshold: float = 0.1,
                       clip_min: float = 0.0, clip_max: float = 0.9):
    """
    Implements the repo's joystick_erp + running max logic.

    Steps:
      cs  := clip(cs_values, [clip_min, clip_max])
      d   := cs[t] - cs[t-1], with d[0] = 0
      erp[t] = d[t] - max(d[:t])   # record-breaking upward surplus
      erp[erp < threshold] = 0     # ignore small/negative changes
      progress := running max of erp (monotone)
      event    := (erp > 0).astype(float)

    Returns
    -------
    progress : 1D float array (monotone)
    event    : 1D float array (0/1 flags)
    erp      : 1D float array (post-threshold)
    """
    cs = np.asarray(cs_values, dtype=np.float32)
    cs = np.clip(cs, clip_min, clip_max)

    d = np.empty_like(cs)
    d[0] = 0.0
    d[1:] = cs[1:] - cs[:-1]

    # previous running max of d (exclude current t)
    runmax = np.maximum.accumulate(d)
    prev_max = np.concatenate([[0.0], runmax[:-1]])

    erp = d - prev_max
    erp[erp < threshold] = 0.0

    progress = np.maximum.accumulate(erp)
    event = (erp > 0.0).astype(np.float32)
    return progress, event, erp


# ---------------------------
# Window extraction (3 s windows by default)
# ---------------------------
def extract_windows_with_labels(participant_path,
                                task: str = "regression",
                                target_len: int = 300,     # 3 s at 100 Hz
                                window_samples: int = 900, # 3 s at 300 Hz
                                step_samples: int = 900,
                                event_threshold: float = 0.1,
                                clip_min: float = 0.0,
                                clip_max: float = 0.9):
    """
    Build per-window EEG and labels aligned with the repo.

    - EEG: resample 300→100 Hz, band-pass 1–40, notch 50 Hz (Q≈5).
    - Labels:
        task='regression'     → progress[end_of_window]
        task='classification' → event[end_of_window]  (0/1)
    """
    assert task in {"regression", "classification"}

    # discover files
    eeg_files = [f for f in os.listdir(participant_path) if f.endswith("_EEG.xlsx")]
    cs_files  = [f for f in os.listdir(participant_path) if f.endswith("SubjectiveCs.xlsx")]

    participant_id = os.path.basename(participant_path)
    all_eeg_windows, all_cs_labels = [], []

    for eeg_file in eeg_files:
        # pair EEG/CS by prefix
        prefix = eeg_file.replace("_EEG.xlsx", "")
        matches = [f for f in cs_files if f.startswith(prefix)]
        if not matches:
            continue
        cs_file = matches[0]

        # load
        eeg_df = pd.read_excel(os.path.join(participant_path, eeg_file))
        cs_df  = pd.read_excel(os.path.join(participant_path, cs_file))
        if len(eeg_df) < window_samples:
            continue

        # parse
        eeg_millis = eeg_df["Millis"].values
        cs_millis  = cs_df["Millis"].values
        cs_values  = cs_df["Rating"].values

        cols = list(eeg_df.columns)
        eeg_ch = cols[2:]  # drop 'Millis','Hardware'
        eeg_raw = eeg_df[eeg_ch].values  # (n_samples, n_channels)

        # overlap span
        start_ms, end_ms = find_common_span(eeg_millis, cs_millis)
        if start_ms is None:
            continue

        eeg_mask = (eeg_millis >= start_ms) & (eeg_millis <= end_ms)
        cs_mask  = (cs_millis  >= start_ms) & (cs_millis  <= end_ms)

        eeg_m = eeg_millis[eeg_mask]
        cs_m  = cs_millis[cs_mask]
        eeg_d = eeg_raw[eeg_mask]
        cs_v  = cs_values[cs_mask]

        # map CS to EEG timeline
        matched_cs = match_cs_to_eeg(eeg_m, cs_m, cs_v)

        # build per-sample targets on raw 300 Hz timeline
        progress, event, _ = compute_cs_targets(
            matched_cs, threshold=event_threshold, clip_min=clip_min, clip_max=clip_max
        )

        # windowing (non-overlapping if step==window)
        n = len(eeg_d)
        if n < window_samples:
            continue
        n_windows = 1 + (n - window_samples) // step_samples

        for w in range(n_windows):
            s = w * step_samples
            e = s + window_samples
            if e > n:
                break

            # label at END of window (segment-wise, like repo)
            raw_label = float(matched_cs[e - 1])
            if task == "regression":
                cs_label = raw_label
            else:  # classification
                cs_label = float(event[e - 1])

            # EEG → preprocess → ensure target_len at 100 Hz (3 s → 300)
            eeg_win = eeg_d[s:e]                     # (900, n_ch) at 300 Hz
            eeg_proc = preprocess_eeg(eeg_win)       # (~300, n_ch) at 100 Hz
            if eeg_proc.shape[0] != target_len:
                eeg_proc = signal.resample(eeg_proc, target_len, axis=0)

            # store as (C, T)
            all_eeg_windows.append(eeg_proc.T)
            all_cs_labels.append(cs_label)

    if not all_eeg_windows:
        return None, None, None

    eeg_windows = np.stack(all_eeg_windows, axis=0)      # (N, C, T) with T=target_len
    cs_labels   = np.asarray(all_cs_labels, dtype=float) # (N,)
    participant_ids = np.full(len(all_eeg_windows), participant_id)



    return eeg_windows, cs_labels, participant_ids



def compute_psd_batch(
    eeg_windows,          # shape: (N_win, N_chan, N_samp)
    sfreq=100,                # Hz
    fmin=0.5,             # Hz
    fmax=40.0,            # Hz
    bandwidth=1.0,        # Hz (multitaper smoothing)
    match_eeg_shape=True, # if True → return (N_win, N_chan, N_samp)
    return_raw=False,     # if True → also return the raw (N_win, N_chan, N_freqs)
    log_power=False,      # set True if you want log10 power
    eps=1e-12             # floor to avoid log(0)
    ):
    """
    Compute multitaper PSD per window (no resampling/filtering), then optionally
    interpolate along the frequency axis so PSD has the same trailing length as EEG.
    """
    eeg_windows = np.asarray(eeg_windows, dtype=np.float64)  # mne likes float64
    n_win, n_chan, n_samp = eeg_windows.shape

    psd_list = []
    freqs = None
    for w in range(n_win):
        psd, f = psd_array_multitaper(
            eeg_windows[w], sfreq=sfreq,
            fmin=fmin, fmax=fmax, bandwidth=bandwidth,
            adaptive=True, low_bias=True, normalization='full', verbose=False
        )  # psd: (n_chan, n_freqs)
        if freqs is None:
            freqs = f
        psd_list.append(psd)

    psd_raw = np.stack(psd_list, axis=0)  # (N_win, N_chan, N_freqs)
    if log_power:
        psd_raw = np.log10(np.maximum(psd_raw, eps))

    if not match_eeg_shape or psd_raw.shape[-1] == n_samp:
        out = psd_raw
    else:
        # interpolate PSD along freq axis to length = n_samp
        target_freqs = np.linspace(freqs[0], freqs[-1], num=n_samp)
        out = np.empty((n_win, n_chan, n_samp), dtype=psd_raw.dtype)
        # simple, robust interpolation
        for i in range(n_win):
            for c in range(n_chan):
                out[i, c, :] = np.interp(target_freqs, freqs, psd_raw[i, c, :])

    if return_raw:
        return out.astype(np.float32), freqs.astype(np.float32), psd_raw.astype(np.float32)
    return out.astype(np.float32), freqs.astype(np.float32)