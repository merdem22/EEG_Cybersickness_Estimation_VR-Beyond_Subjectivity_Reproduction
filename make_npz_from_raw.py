
import os
import argparse
import numpy as np
from pathlib import Path
import pandas as pd

# Import your provided preprocessing helpers
from preprocess_omer import (
    extract_windows_with_labels,
    compute_psd_batch,
    find_common_span,
    match_cs_to_eeg,
)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def compute_window_end_cs(participant_path: Path,
                          window_samples: int = 900,  # 3 s at 300 Hz
                          step_samples: int = 900):
    """Return the raw (clipped) CS value at the END of each 3 s window
    in the same order that extract_windows_with_labels iterates files.

    This reconstructs the per-window 'joy' expected by the loader, i.e. the
    joystick level (after clipping) per time step, not the derived 'progress'.
    """
    eeg_files = [f for f in os.listdir(participant_path) if f.endswith("_EEG.xlsx")]
    cs_files  = [f for f in os.listdir(participant_path) if f.endswith("SubjectiveCs.xlsx")]

    cs_window_values = []

    for eeg_file in eeg_files:
        prefix = eeg_file.replace("_EEG.xlsx", "")
        matches = [f for f in cs_files if f.startswith(prefix)]
        if not matches:
            continue
        cs_file = matches[0]

        eeg_df = pd.read_excel(Path(participant_path) / eeg_file)
        cs_df  = pd.read_excel(Path(participant_path) / cs_file)
        if len(eeg_df) < window_samples:
            continue

        eeg_millis = eeg_df["Millis"].values
        cs_millis  = cs_df["Millis"].values
        cs_values  = cs_df["Rating"].values

        start_ms, end_ms = find_common_span(eeg_millis, cs_millis)
        if start_ms is None:
            continue

        eeg_mask = (eeg_millis >= start_ms) & (eeg_millis <= end_ms)
        cs_mask  = (cs_millis  >= start_ms) & (cs_millis  <= end_ms)

        eeg_m = eeg_millis[eeg_mask]
        cs_m  = cs_millis[cs_mask]
        cs_v  = cs_values[cs_mask]

        matched_cs = match_cs_to_eeg(eeg_m, cs_m, cs_v)  # len == len(eeg_m)

        n = len(eeg_m)
        if n < window_samples:
            continue
        n_windows = 1 + (n - window_samples) // step_samples

        for w in range(n_windows):
            s = w * step_samples
            e = s + window_samples
            if e > n:
                break
            cs_end = float(matched_cs[e - 1])
            # clip like in loader
            cs_end = np.clip(cs_end, 0.0, 0.9)
            cs_window_values.append(np.array([cs_end], dtype=np.float32))

    return cs_window_values

def build_single_npz(participant_path: Path, out_dir: Path, session_idx: int = 1,
                     task: str = "regression",
                     sfreq: int = 100,
                     fmin: float = 0.5,
                     fmax: float = 40.0,
                     bandwidth: float = 1.0):
    # 1) Extract windows and (progress-based) labels for sanity
    windows, cs_labels, participant_ids = extract_windows_with_labels(
        participant_path=str(participant_path),
        task=task,
    )
    if windows is None or len(windows) == 0:
        print(f"[skip] no windows for {participant_path.name}")
        return None

    # 1a) Build 'joy' = raw CS at window end (clipped), so loader's pipeline matches ours
    joy_list = compute_window_end_cs(participant_path)
    if len(joy_list) != len(windows):
        print(f"[warn] joy count ({len(joy_list)}) != windows ({len(windows)}); proceeding with min length")
        L = min(len(joy_list), len(windows))
        windows = windows[:L]
        cs_labels = cs_labels[:L]
        joy_list = joy_list[:L]

    # 2) PSDs (both time-aligned and raw-frequency versions)
    #    psd_time: (N, C, T) -> we will save as (N, T, C)
    #    psd_raw : (N, C, F) -> leave as-is for loader's psd_raw
    psd_time, freqs, psd_raw = compute_psd_batch(
        windows, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth, return_raw=True
    )

    N, C, T = windows.shape
    assert psd_time.shape == (N, C, T), f"psd_time shape mismatch: {psd_time.shape} vs {(N,C,T)}"
    assert psd_raw.shape[0] == N and psd_raw.shape[1] == C, "psd_raw shape mismatch"

    # 3) Conform to loader expectations
    eeg_for_file = np.transpose(windows, (0, 2, 1)).astype(np.float32)    # (N, T, C)
    psd_for_file = np.transpose(psd_time, (0, 2, 1)).astype(np.float32)   # (N, T, C)
    psd_raw_for_file = psd_raw.astype(np.float32)                          # (N, C, F)

    # 5) Optional/unused keys kept empty so the kinematic loop does nothing if left in code.
    tf_list, pth_list = [], []

    dataset = dict(
        eeg=eeg_for_file,
        psd=psd_for_file,
        psd_raw=psd_raw_for_file,
        joy=joy_list,
        tf=tf_list,
        pth=pth_list,
    )

    pid = str(participant_ids[0]) if participant_ids is not None else participant_path.name
    out_name = f"{pid}_{session_idx:02d}.npz"
    out_path = out_dir / out_name
    np.savez_compressed(out_path, dataset=dataset)
    print(f"[ok] wrote {out_path}  → dataset keys: {list(dataset.keys())}  shapes: "
          f"eeg{eeg_for_file.shape}, psd{psd_for_file.shape}, psd_raw{psd_raw_for_file.shape}, joy[{len(joy_list)}]")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, required=True, help="Folder with per-participant subfolders (e.g., datasets/raw)")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder for NPZ cache (e.g., datasets/.cache)")
    ap.add_argument("--task", type=str, default="regression", choices=["regression","classification"])
    ap.add_argument("--sfreq", type=int, default=100)
    ap.add_argument("--fmin", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    participants = [p for p in raw_root.iterdir() if p.is_dir()]
    if not participants:
        print(f"[warn] No participant folders found in {raw_root}")
        return

    for p in sorted(participants):
        build_single_npz(
            participant_path=p,
            out_dir=out_dir,
            session_idx=1,
            task=args.task,
            sfreq=args.sfreq,
            fmin=args.fmin,
            fmax=args.fmax,
            bandwidth=args.bandwidth,
        )

if __name__ == "__main__":
    main()
