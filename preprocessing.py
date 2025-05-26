import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from mne.time_frequency import psd_array_multitaper

def preprocess_eeg_batch(eeg_batch,
                         orig_sfreq=300,
                         target_sfreq=100,
                         bp_lower=1.0,
                         bp_upper=40.0,
                         notch_freq=50.0,
                         notch_q=5.0):
    """
    Resample + bandpass + notch each segment in eeg_batch.

    Parameters
    ----------
    eeg_batch : np.ndarray
        Shape (n_segments, n_channels, n_samples_orig) at orig_sfreq.
    orig_sfreq : float
        Original sampling rate (Hz).
    target_sfreq : float
        Desired sampling rate (Hz).
    bp_lower, bp_upper : float
        Bandpass cut-offs (Hz).
    notch_freq : float
        Center freq of notch (Hz).
    notch_q : float
        Quality factor for notch filter.

    Returns
    -------
    processed : np.ndarray
        Shape (n_segments, n_channels, n_samples_new) at target_sfreq.
    """
    processed = []
    down = int(orig_sfreq // target_sfreq)
    # design bandpass
    nyq = target_sfreq / 2.
    b_bp, a_bp = butter(4,
                        [bp_lower/nyq, bp_upper/nyq],
                        btype='band')
    # design notch
    w0 = notch_freq / nyq
    b_notch, a_notch = iirnotch(w0, notch_q)
    for seg in eeg_batch:
        # 1) resample
        seg_rs = resample_poly(seg,
                               up=target_sfreq,
                               down=orig_sfreq,
                               axis=-1)
        # 2) band-pass
        seg_bp = filtfilt(b_bp, a_bp, seg_rs, axis=-1)
        # 3) notch
        seg_out = filtfilt(b_notch, a_notch, seg_bp, axis=-1)
        processed.append(seg_out)
    return np.stack(processed, axis=0)


def compute_temporal_relative_psd(eeg_batch,
                                  sfreq=100.,
                                  window_sec=3.,
                                  fmin=0.5,
                                  fmax=40.,
                                  bandwidth=1.0):
    """
    Compute temporal relative PSD per segment.

    Parameters
    ----------
    eeg_batch : np.ndarray
        Shape (n_segments, n_channels, n_samples)
        Each segment must be window_sec long at sfreq.
    sfreq : float
        Sampling rate after preprocessing.
    window_sec : float
        Duration of each segment (s).
    fmin, fmax : float
        Frequency range for PSD.
    bandwidth : float
        Smoothing bandwidth (Hz) for multitaper.

    Returns
    -------
    tr_psd : np.ndarray
        Shape (n_segments, n_channels, n_freqs),
        PSD minus mean of first three PSDs.
    freqs : np.ndarray
        Frequency bins corresponding to PSD.
    """
    n_segs, n_ch, n_samps = eeg_batch.shape
    expected = int(window_sec * sfreq)
    if n_samps != expected:
        raise ValueError(f"Each segment must have {expected} samples, got {n_samps}")

    psd_list = []
    freqs = None
    for seg in eeg_batch:
        # seg: (n_channels, n_samples)
        psd, freq = psd_array_multitaper(seg,
                                         sfreq=sfreq,
                                         fmin=fmin,
                                         fmax=fmax,
                                         bandwidth=bandwidth,
                                         adaptive=True,
                                         low_bias=True,
                                         normalization='full',
                                         verbose=False)
        psd_list.append(psd)
        if freqs is None:
            freqs = freq

    psds = np.stack(psd_list, axis=0)  # (n_segments, n_channels, n_freqs)
    init_mean = psds[:3].mean(axis=0)   # (n_channels, n_freqs)
    tr_psd = psds - init_mean[None, ...]
    return tr_psd, freqs


def process_and_extract_tr_psd(eeg_batch,
                               orig_sfreq=300,
                               target_sfreq=100,
                               **kwargs):
    """
    Full pipeline: preprocess + temporal relative PSD.

    Returns
    -------
    tr_psd : np.ndarray
    freqs : np.ndarray
    """
    proc = preprocess_eeg_batch(eeg_batch,
                                orig_sfreq=orig_sfreq,
                                target_sfreq=target_sfreq,
                                **{k:kwargs[k] for k in ['bp_lower','bp_upper',
                                                         'notch_freq','notch_q']
                                   if k in kwargs})
    tr_psd, freqs = compute_temporal_relative_psd(proc,
                                                  sfreq=target_sfreq,
                                                  **{k:kwargs[k] for k in ['window_sec',
                                                                           'fmin','fmax',
                                                                           'bandwidth']
                                                     if k in kwargs})
    return tr_psd, freqs

# Example use:
# eeg_batch = np.random.randn(10, 32, 300*3)  # 10 segments, 32 channels, 3 s at 300 Hz
# tr_psd, freqs = process_and_extract_tr_psd(
#     eeg_batch,
#     bp_lower=1,
#     bp_upper=40,
#     notch_freq=50,
#     notch_q=5,
#     window_sec=3,
#     fmin=0.5,
#     fmax=40,
#     bandwidth=1.0
# )
