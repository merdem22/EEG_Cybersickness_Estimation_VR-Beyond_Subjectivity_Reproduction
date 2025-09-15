
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# --- Optional MNE (multitaper); fall back to Welch if not available ---
try:
    from mne.time_frequency import psd_array_multitaper
    _HAS_MNE = True
except Exception:
    from scipy.signal import welch
    _HAS_MNE = False

# --- Import your preprocessing pipeline ---
# Expects a local preprocessing.py with preprocess_eeg_batch(...) and compute_temporal_relative_psd(...)
try:
    from preprocessing import preprocess_eeg_batch, compute_temporal_relative_psd
except Exception as e:
    raise RuntimeError("Could not import preprocessing.py. Make sure it's on PYTHONPATH") from e


def _collect_sessions(base: str) -> Dict[str, List[str]]:
    """Group files as <patient>_<SESS>_* (e.g., 0001_FN_*)."""
    parts: Dict[str, List[str]] = {}
    for f in os.listdir(base):
        if f.endswith(".xlsx"):
            try:
                pid, sess, _ = f.split("_", 2)
                key = f"{pid}_{sess}"
                parts.setdefault(key, []).append(f)
            except ValueError:
                continue
    return {k: sorted(v) for k, v in parts.items()}


def _read_if_exists(path: str, min_rows: int = 30) -> Optional[pd.DataFrame]: #this is done to avoid reading empty files.
    """Read Excel if it exists and has enough rows; else return None."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_excel(path)
        if len(df) < min_rows:
            return None
        return df
    except Exception:
        return None


def _estimate_fs_from_millis(millis: np.ndarray, n_rows: int) -> float:
    """Roughly estimate sampling rate from total span and row count; snap to common EEG rates."""
    millis = np.asarray(millis, dtype=float)
    dt = (np.nanmax(millis) - np.nanmin(millis)) / 1000.0  # seconds
    if dt <= 0:
        return 300.0
    fs = n_rows / max(dt, 1e-9)
    for cand in [128, 200, 256, 300, 500, 512, 1000]:
        if abs(fs - cand) / max(cand, 1e-9) < 0.2:
            return float(cand)
    return float(fs)


def build_cache_for_session(key: str,
                            base: str,
                            outdir: str,
                            window_sec: float = 3.0,
                            step_sec: float = 3.0,
                            target_sfreq: float = 100.0,
                            min_rows: int = 30) -> Tuple[str, dict]:
    """
    Convert raw Excel for a session (e.g., 0001_FN) to a single .npz cache expected by loader.py.

    Produces <outdir>/<patient>_<SESS>.npz with key 'dataset' holding:
      - eeg:     (N, C, S) segments (C=channels, S=samples per 3 s window at original fs)
      - psd:     (N, C, F) log10 power (0.5–40 Hz)
      - psd_raw: (N, C, F) absolute power
      - tr_psd:  (N, C, F) temporal-relative PSD (optional convenience)
      - tf:      list[np.ndarray], each (Lt, 6) from Transforms within the window
      - pth:     list[np.ndarray], each (Lp, 7) from Path within the window
      - joy:     list[[float]] from SubjectiveCs (mean rating per window)
      - meta:    dict with freqs, fs_est, etc.
    """
    pid, sess = key.split("_")
    eeg_path = os.path.join(base, f"{key}_EEG.xlsx")
    path_path = os.path.join(base, f"{key}_Path.xlsx")
    tf_path   = os.path.join(base, f"{key}_Transforms.xlsx")
    subc_path = os.path.join(base, f"{key}_SubjectiveCs.xlsx")

    EEG = _read_if_exists(eeg_path, min_rows)
    SUB = _read_if_exists(subc_path, min_rows)
    PTH = _read_if_exists(path_path, min_rows)
    TF  = _read_if_exists(tf_path,  min_rows)

    if EEG is None or SUB is None:
        raise RuntimeError(f"{key}: missing required EEG or SubjectiveCs (or too few rows)")

    # Ensure time column presence
    for df, name in [(EEG, "EEG"), (SUB, "SubjectiveCs"), (PTH, "Path"), (TF, "Transforms")]:
        if df is None:
            continue
        if "Millis" not in df.columns:
            raise RuntimeError(f"{key}: '{name}' is missing 'Millis' column")

    # Maximal overlapping interval across available modalities
    mins, maxs = [], []
    for df in [EEG, SUB, PTH, TF]:
        if df is not None and len(df) >= min_rows:
            ms = df["Millis"].to_numpy(dtype=float)
            mins.append(np.nanmin(ms))
            maxs.append(np.nanmax(ms))
    start_ms = max(mins)
    end_ms   = min(maxs)
    if not (np.isfinite(start_ms) and np.isfinite(end_ms) and end_ms > start_ms):
        raise RuntimeError(f"{key}: could not determine a valid overlapping interval")

    def trim(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        ms = df["Millis"].to_numpy(dtype=float)
        m = (ms >= start_ms) & (ms <= end_ms)
        return df.loc[m].reset_index(drop=True)

    EEG = trim(EEG); SUB = trim(SUB); PTH = trim(PTH); TF = trim(TF)

    # Columns / arrays
    eeg_ch_cols = [c for c in EEG.columns if c.startswith("ch_")]
    if not eeg_ch_cols:
        raise RuntimeError(f"{key}: no EEG channels found (columns starting with 'ch_')")
    tf_cols  = ["HeadPosition_X","HeadPosition_Y","HeadPosition_Z","HeadRotation_Yaw","HeadRotation_Pitch","HeadRotation_Roll"]
    pth_cols = ["ProgressMeters","Position_X","Position_Y","Position_Z","Tangent_X","Tangent_Y","Tangent_Z"]

    eeg_ms = EEG["Millis"].to_numpy(dtype=float)
    eeg_X  = EEG[eeg_ch_cols].to_numpy(dtype=float)
    sub_ms = SUB["Millis"].to_numpy(dtype=float)
    sub_y  = SUB["Rating"].to_numpy(dtype=float)

    if TF is not None and all(c in TF.columns for c in tf_cols):
        tf_ms = TF["Millis"].to_numpy(dtype=float)
        tf_X  = TF[tf_cols].to_numpy(dtype=float)
    else:
        tf_ms, tf_X = None, None

    if PTH is not None and all(c in PTH.columns for c in pth_cols):
        pth_ms = PTH["Millis"].to_numpy(dtype=float)
        pth_X  = PTH[pth_cols].to_numpy(dtype=float)
    else:
        pth_ms, pth_X = None, None

    # Build 3 s non-overlapping windows on EEG
    fs0 = _estimate_fs_from_millis(eeg_ms, len(eeg_ms))
    win0  = int(round(window_sec * fs0))
    step0 = int(round(step_sec * fs0))
    idx0 = np.arange(0, len(eeg_X) - win0 + 1, step0, dtype=int)
    if len(idx0) == 0:
        raise RuntimeError(f"{key}: not enough EEG samples for a single {window_sec}s window")

    eeg_segments: List[np.ndarray] = []
    tf_segments:  List[np.ndarray] = []
    pth_segments: List[np.ndarray] = []
    joy_segments: List[List[float]] = []

    for start in idx0:
        stop = start + win0
        seg_ms = eeg_ms[start:stop]
        seg_eeg = eeg_X[start:stop, :]                       # (win0, C)
        t0, t1 = float(seg_ms[0]), float(seg_ms[-1])

        # Slice TF/PTH rows within this window
        if tf_ms is not None:
            m = (tf_ms >= t0) & (tf_ms <= t1)
            tf_segments.append(tf_X[m, :])
        if pth_ms is not None:
            m = (pth_ms >= t0) & (pth_ms <= t1)
            pth_segments.append(pth_X[m, :])

        # Label from SubjectiveCs
        m = (sub_ms >= t0) & (sub_ms <= t1)
        if np.any(m):
            joy_segments.append([float(np.nanmean(sub_y[m]))])
        else:
            near = int(np.argmin(np.abs(sub_ms - 0.5*(t0+t1))))
            joy_segments.append([float(sub_y[near])])

        eeg_segments.append(seg_eeg.T.astype(np.float32))     # (C, win0)

    eeg_segments = np.stack(eeg_segments, axis=0)             # (N, C, win0)

    # --- Preprocess EEG to target_sfreq and compute PSDs ---
    proc = preprocess_eeg_batch(eeg_segments,
                                orig_sfreq=fs0,
                                target_sfreq=target_sfreq,
                                bp_lower=1.0,
                                bp_upper=40.0,
                                notch_freq=50.0,
                                notch_q=5.0)                   # (N, C, win_new)

    if _HAS_MNE:
        psd_list = []
        freqs = None
        for seg in proc:
            psd, fr = psd_array_multitaper(seg,
                                           sfreq=target_sfreq,
                                           fmin=0.5, fmax=40.0,
                                           bandwidth=1.0,
                                           adaptive=True,
                                           low_bias=True,
                                           normalization='full',
                                           verbose=False)
            psd_list.append(psd.astype(np.float32))
            if freqs is None:
                freqs = fr
        psd_raw = np.stack(psd_list, axis=0).astype(np.float32)  # (N, C, F)
    else:
        # Welch fallback: (N, C, F) limited to 0.5–40 Hz
        from scipy.signal import welch
        nperseg = int(target_sfreq)  # ~1 s
        freqs = None
        psd_raw_list = []
        for seg in proc:  # (C, samples)
            f, p = welch(seg, fs=target_sfreq, nperseg=min(nperseg, seg.shape[-1]), axis=-1)
            band = (f >= 0.5) & (f <= 40.0)
            if freqs is None:
                freqs = f[band]
            psd_raw_list.append(p[..., band].astype(np.float32))
        psd_raw = np.stack(psd_raw_list, axis=0).astype(np.float32)

    psd_log = np.log10(np.maximum(psd_raw, 1e-12)).astype(np.float32)

    # Temporal-relative PSD (not used by loader but handy)
    try:
        tr_psd, tr_freqs = compute_temporal_relative_psd(proc,
                                                         sfreq=target_sfreq,
                                                         window_sec=window_sec,
                                                         fmin=0.5, fmax=40.0,
                                                         bandwidth=1.0)
        tr_psd = tr_psd.astype(np.float32)
    except Exception:
        tr_psd = np.zeros_like(psd_raw, dtype=np.float32)

    dataset = {
        "eeg": eeg_segments.astype(np.float32),
        "psd": psd_log,
        "psd_raw": psd_raw,
        "tr_psd": tr_psd,
        "tf": tf_segments if tf_segments else [],
        "pth": pth_segments if pth_segments else [],
        "joy": joy_segments,
        "meta": {
            "freqs": np.asarray(freqs, dtype=np.float32),
            "fs_est": float(fs0),
            "target_sfreq": float(target_sfreq),
            "window_sec": float(window_sec),
            "start_ms": float(start_ms),
            "end_ms": float(end_ms),
            "channels": [str(c) for c in eeg_ch_cols]
        }
    }

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{pid}_{sess}.npz")
    np.savez_compressed(outpath, dataset=dataset)  # key 'dataset' for loader.py
    return outpath, {"N": int(eeg_segments.shape[0]), "F": int(psd_raw.shape[-1])}


def main():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=".", help="Folder containing raw Excel files")
    p.add_argument("--outdir", default="./.cache", help="Output folder for .npz cache")
    p.add_argument("--window-sec", type=float, default=3.0)
    p.add_argument("--step-sec", type=float, default=3.0)
    p.add_argument("--target-sfreq", type=float, default=100.0)
    p.add_argument("--min-rows", type=int, default=30, help="Minimum rows to consider a sheet non-empty")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    sessions = _collect_sessions(args.base)
    print(f"Found sessions: {sorted(sessions)}")
    results = {}
    for key in sorted(sessions):
        try:
            out, meta = build_cache_for_session(key,
                                                base=args.base,
                                                outdir=args.outdir,
                                                window_sec=args.window_sec,
                                                step_sec=args.step_sec,
                                                target_sfreq=args.target_sfreq,
                                                min_rows=args.min_rows)
            results[key] = {"out": out, **meta}
            print(f"Wrote {out} -> {meta}")
        except Exception as e:
            results[key] = {"error": str(e)}
            print(f"[SKIP] {key}: {e}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
        
""" #for reference
  python build_cache.py \                                                                                                  
  --base datasets/ \
  --outdir datasets/.cache \
  --window-sec 3 --step-sec 3 --target-sfreq 100
"""

