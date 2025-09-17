
import os
import argparse
import numpy as np
from pathlib import Path
from preprocess_omer import extract_windows_with_labels, compute_psd_batch

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def build_single_npz(participant_path: Path, out_dir: Path, session_idx: int = 1,
                     task: str = "regression",
                     sfreq: int = 100,
                     fmin: float = 0.5,
                     fmax: float = 40.0,
                     bandwidth: float = 1.0,
                     psd_raw_source: str = "time"):
    windows, cs_labels, participant_ids = extract_windows_with_labels(
        participant_path=str(participant_path),
        task=task,
    )
    if windows is None or len(windows) == 0:
        print(f"[skip] no windows for {participant_path.name}")
        return None

    psd_time, freqs, psd_raw = compute_psd_batch(
        windows, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth, return_raw=True
    )

    N, C, T = windows.shape

    # Store eeg/psd for compatibility (N, T, C)
    eeg_for_file = np.transpose(windows, (0, 2, 1)).astype(np.float32)
    psd_for_file = np.transpose(psd_time, (0, 2, 1)).astype(np.float32)

    # Choose psd_raw source:
    # - 'time' → (N, C, T=300): robust for CNN with multiple pools
    # - 'raw'  → (N, C, F): true frequency bins (may be too short for your CNN)
    if psd_raw_source == "time":
        psd_raw_for_file = psd_time.astype(np.float32)           # (N, C, 300)
    else:
        psd_raw_for_file = psd_raw.astype(np.float32)            # (N, C, F)

    joy_list = [np.array([float(x)], dtype=np.float32) for x in cs_labels]
    dataset = dict(
        eeg=eeg_for_file,
        psd=psd_for_file,
        psd_raw=psd_raw_for_file,
        joy=joy_list,
        tf=[],
        pth=[],
    )
    pid = str(participant_ids[0]) if participant_ids is not None else participant_path.name
    out_name = f"{pid}_{session_idx:02d}.npz"
    out_path = out_dir / out_name
    np.savez_compressed(out_path, dataset=dataset)
    print(f"[ok] wrote {out_path}  → psd_raw.shape={psd_raw_for_file.shape}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--task", type=str, default="regression", choices=["regression","classification"])
    ap.add_argument("--sfreq", type=int, default=100)
    ap.add_argument("--fmin", type=float, default=0.5)
    ap.add_argument("--fmax", type=float, default=40.0)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--psd-raw", type=str, default="time", choices=["time","raw"],
                    help="Store psd_raw as 'time' (N,C,300) or 'raw' (N,C,F). Use 'time' for your CNN.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

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
            psd_raw_source=args.psd_raw,
        )

if __name__ == "__main__":
    main()
