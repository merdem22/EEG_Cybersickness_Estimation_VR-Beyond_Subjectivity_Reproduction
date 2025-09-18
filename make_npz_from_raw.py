# make_npz_from_raw.py (speed-optimized)
import os, argparse, shutil
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from preprocess_omer import extract_windows_with_labels, compute_psd_batch

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _build_one(args_tuple):
    (participant_path, out_dir, session_idx, task, sfreq, fmin, fmax,
     bandwidth, psd_raw_source, compress, joy_format, skip_existing, stage_dir) = args_tuple

    try:
        pid_hint = participant_path.name
        # precompute output path
        # we’ll try to get pid from extractor later; if not, use folder name
        out_name = f"{pid_hint}_{session_idx:02d}.npz"
        final_out = out_dir / out_name
        if skip_existing and final_out.exists():
            return f"[skip] exists: {final_out}"

        # load windows & labels
        windows, cs_labels, participant_ids = extract_windows_with_labels(
            participant_path=str(participant_path), task=task,
        )
        if windows is None or len(windows) == 0:
            return f"[skip] no windows for {participant_path.name}"

        # determine pid (stable naming)
        pid = str(participant_ids[0]) if participant_ids is not None else participant_path.name
        out_name = f"{pid}_{session_idx:02d}.npz"
        final_out = out_dir / out_name
        if skip_existing and final_out.exists():
            return f"[skip] exists: {final_out}"

        # compute PSDs; avoid extra work if psd_raw not needed
        need_raw = (psd_raw_source == "raw")
        psd_time, freqs, psd_raw = compute_psd_batch(
            windows, sfreq=sfreq, fmin=fmin, fmax=fmax, bandwidth=bandwidth, return_raw=need_raw
        )

        # shapes & dtype once
        windows = np.asarray(windows, dtype=np.float32, order="C")
        psd_time = np.asarray(psd_time, dtype=np.float32, order="C")
        if need_raw:
            psd_raw = np.asarray(psd_raw, dtype=np.float32, order="C")

        # Store eeg/psd for compatibility (N, T, C) / (N, F, C) via transpose
        eeg_for_file = np.transpose(windows, (0, 2, 1))   # (N, T, C)
        psd_for_file = np.transpose(psd_time, (0, 2, 1))  # (N, F, C) or (N, 300, C)

        if psd_raw_source == "time":
            psd_raw_for_file = psd_time                    # (N, C, 300) before transpose; we keep original as (N, C, 300)
        else:
            psd_raw_for_file = psd_raw                     # (N, C, F)

        # joy formatting
        N = eeg_for_file.shape[0]
        if joy_format == "legacy":
            joy = [np.array([float(x)], dtype=np.float32) for x in cs_labels]  # list-of-1 arrays (original)
        else:
            joy = np.asarray(cs_labels, dtype=np.float32).reshape(N, 1)        # faster & pickle-free for loaders that accept ndarrays

        dataset = dict(
            eeg=eeg_for_file,             # (N, T, C)
            psd=psd_for_file,             # (N, F, C)
            psd_raw=np.asarray(psd_raw_for_file, dtype=np.float32),  # (N, C, 300) or (N, C, F)
            joy=joy,
            tf=np.empty((0,), dtype=np.float32),   # keep empty arrays (avoid object pickle)
            pth=np.empty((0,), dtype="U1"),
        )

        # stage locally (fast) then move to Drive (slow, but once)
        tmp_dir = Path(stage_dir) if stage_dir else final_out.parent
        ensure_dir(tmp_dir)
        tmp_path = tmp_dir / (out_name + ".tmp")

        if compress == "zip":
            np.savez_compressed(tmp_path, dataset=dataset)
        else:
            np.savez(tmp_path, dataset=dataset)

        # atomic-ish move to final destination
        ensure_dir(final_out.parent)
        shutil.move(str(tmp_path), str(final_out))
        return f"[ok] {final_out} → psd_raw.shape={np.asarray(psd_raw_for_file).shape}"
    except Exception as e:
        return f"[err] {participant_path.name}: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",  type=str, required=True)
    ap.add_argument("--out-dir",   type=str, required=True)
    ap.add_argument("--task",      type=str, default="regression", choices=["regression","classification"])
    ap.add_argument("--sfreq",     type=int, default=100)
    ap.add_argument("--fmin",      type=float, default=0.5)
    ap.add_argument("--fmax",      type=float, default=40.0)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--psd-raw",   type=str, default="time", choices=["time","raw"],
                    help="Store psd_raw as 'time' (N,C,300) or 'raw' (N,C,F). If 'time', we skip computing raw freqs for speed.")
    ap.add_argument("--n-workers", type=int, default=os.cpu_count(), help="Parallel processes across participants.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip participants that already have an .npz.")
    ap.add_argument("--compress", choices=["zip","none"], default="zip", help="zip is smaller but slower; none is faster but larger.")
    ap.add_argument("--joy-format", choices=["legacy","array"], default="legacy",
                    help="Use 'legacy' to keep list-of-1 arrays; 'array' to store an (N,1) float32 array (faster).")
    ap.add_argument("--stage-dir", type=str, default=None,
                    help="Optional local staging dir (e.g., /content/cache_stage) for fast writes before moving to Drive.")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_dir  = Path(args.out_dir)
    ensure_dir(out_dir)

    participants = sorted([p for p in raw_root.iterdir() if p.is_dir()])
    if not participants:
        print(f"[warn] No participant folders found in {raw_root}")
        return

    jobs = []
    for p in participants:
        jobs.append((p, out_dir, 1, args.task, args.sfreq, args.fmin, args.fmax,
                     args.bandwidth, args.psd_raw, args.compress, args.joy_format,
                     args.skip_existing, args.stage_dir))

    # Parallel fan-out per participant
    n_workers = max(1, int(args.n_workers or 1))
    print(f"[info] Participants: {len(jobs)} | n_workers={n_workers}")
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_build_one, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            print(f"[{i:02d}/{len(jobs)}] {fut.result()}")

if __name__ == "__main__":
    main()
