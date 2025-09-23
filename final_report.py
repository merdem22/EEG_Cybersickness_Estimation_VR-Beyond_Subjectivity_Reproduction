# final_report.py
import re
import os
import sys
import json
import subprocess
from pathlib import Path
from collections import defaultdict
import csv

CACHE_DIR = Path("datasets/.cache")  # change if needed
LOGPREFIX = "runs/exp_final"         # just for your run isolation

# Command template: tweak defaults to match your setup
BASE_ARGS = [
    sys.executable, "main.py",
    "--seed", "10",
    "--task", "regression",
    "--input-type", "power-spectral-no-kinematic",
    "--num-epochs", "20",
    "--batch-size", "8",
    "--logprefix", LOGPREFIX,
    "--output",
    "--no-cuda",
    "--acc-threshold", "0.10",
    "--acc-neighborhood", "1",
    "--no-load-model"               # remove if you actually want to load weights
]
# If you need to load a checkpoint:
# BASE_ARGS += ["--load-model", "runs/exp1/best.ckpt"]

PREDICT_LINE = re.compile(
    r"Predict:\s+MAE=(?P<mae>\d+\.\d+)\s+\|\s+MSE=(?P<mse>\d+\.\d+)\s+\|\s+Baseline\(MAE=(?P<mae_b>\d+\.\d+),\s+MSE=(?P<mse_b>\d+\.\d+)\)\s+\|\s+Acc@0\.10\[r0,r1,r2,r5\]=\[(?P<a0>\d+\.\d+),(?P<a1>\d+\.\d+),(?P<a2>\d+\.\d+),(?P<a5>\d+\.\d+)\]\s+\|\s+N=(?P<N>\d+)"
)

def discover_patients(cache_dir: Path):
    ids = set()
    if not cache_dir.exists():
        print(f"[ERR] cache dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)
    # Common patterns: directories named 0001, 0002, ..., or files containing these
    for p in cache_dir.rglob("*"):
        name = p.name
        m = re.match(r"^(\d{4})$", name)
        if m:
            ids.add(m.group(1))
        else:
            m = re.search(r"(\d{4})", name)
            if m:
                ids.add(m.group(1))
    return sorted(ids)

def run_one_patient(pid: str):
    args = BASE_ARGS + ["--patient", pid]
    proc = subprocess.run(args, capture_output=True, text=True)
    out = proc.stdout + "\n" + proc.stderr
    m = PREDICT_LINE.search(out)
    if not m:
        print(f"[WARN] Could not parse Predict line for patient {pid}.\n--- STDOUT/ERR ---\n{out}\n------------------")
        return None, out
    d = {k: float(v) for k, v in m.groupdict().items()}
    return d, out

def main():
    patients = discover_patients(CACHE_DIR)
    if not patients:
        print("[ERR] No patients discovered under datasets/.cache", file=sys.stderr)
        sys.exit(2)
    print(f"[INFO] Found {len(patients)} patients: {patients}")

    rows = []
    pooled = dict(SUM_ABS=0.0, SUM_SQ=0.0, SUM_ABS_BASE=0.0, SUM_SQ_BASE=0.0, N=0)
    macro = defaultdict(float)
    count_macro = 0

    for pid in patients:
        metrics, raw = run_one_patient(pid)
        if metrics is None:
            continue
        rows.append(dict(patient=pid, **metrics))
        # pooled MAE/MSE using sums
        pooled["SUM_ABS"]      += metrics["mae"] * metrics["N"]
        pooled["SUM_SQ"]       += metrics["mse"] * metrics["N"]
        pooled["SUM_ABS_BASE"] += metrics["mae_b"] * metrics["N"]
        pooled["SUM_SQ_BASE"]  += metrics["mse_b"] * metrics["N"]
        pooled["N"]            += metrics["N"]
        # macro (simple mean of per-patient)
        macro["MAE"]     += metrics["mae"]
        macro["MSE"]     += metrics["mse"]
        macro["MAE_BASE"]+= metrics["mae_b"]
        macro["MSE_BASE"]+= metrics["mse_b"]
        macro["Acc_r0"]  += metrics["a0"]
        macro["Acc_r1"]  += metrics["a1"]
        macro["Acc_r2"]  += metrics["a2"]
        macro["Acc_r5"]  += metrics["a5"]
        count_macro += 1

    if pooled["N"] == 0 or count_macro == 0:
        print("[ERR] No metrics aggregated.", file=sys.stderr)
        sys.exit(3)

    # pooled metrics
    pooled_MAE      = pooled["SUM_ABS"]      / pooled["N"]
    pooled_MSE      = pooled["SUM_SQ"]       / pooled["N"]
    pooled_MAE_BASE = pooled["SUM_ABS_BASE"] / pooled["N"]
    pooled_MSE_BASE = pooled["SUM_SQ_BASE"]  / pooled["N"]

    # macro means
    macro_MAE      = macro["MAE"]      / count_macro
    macro_MSE      = macro["MSE"]      / count_macro
    macro_MAE_BASE = macro["MAE_BASE"] / count_macro
    macro_MSE_BASE = macro["MSE_BASE"] / count_macro
    macro_Acc_r0   = macro["Acc_r0"]   / count_macro
    macro_Acc_r1   = macro["Acc_r1"]   / count_macro
    macro_Acc_r2   = macro["Acc_r2"]   / count_macro
    macro_Acc_r5   = macro["Acc_r5"]   / count_macro

    print("\n=== FINAL REPORT (All cached patients) ===")
    print(f"Pooled:  MAE={pooled_MAE:.6f}  MSE={pooled_MSE:.6f}  |  Baseline: MAE={pooled_MAE_BASE:.6f}  MSE={pooled_MSE_BASE:.6f}")
    print(f"Macro :  MAE={macro_MAE:.6f}  MSE={macro_MSE:.6f}    |  Baseline: MAE={macro_MAE_BASE:.6f}  MSE={macro_MSE_BASE:.6f}")
    print(f"Macro Acc@0.10 : r0={macro_Acc_r0:.2f}%  r1={macro_Acc_r1:.2f}%  r2={macro_Acc_r2:.2f}%  r5={macro_Acc_r5:.2f}%")
    print(f"Total windows (pooled N): {int(pooled['N'])}")

    # write CSV
    out_csv = Path(LOGPREFIX).with_name("final_report.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["patient", "mae", "mse", "mae_b", "mse_b", "a0", "a1", "a2", "a5", "N"]
    with out_csv.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow({
                "patient": r["patient"],
                "mae": r["mae"],
                "mse": r["mse"],
                "mae_b": r["mae_b"],
                "mse_b": r["mse_b"],
                "a0": r["a0"],
                "a1": r["a1"],
                "a2": r["a2"],
                "a5": r["a5"],
                "N": int(r["N"]),
            })
    print(f"[OK] Wrote per-patient CSV to: {out_csv}")

if __name__ == "__main__":
    main()
