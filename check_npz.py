# inspect_loader_safe.py
import numpy as np
from loader import load_train_test_datasets
from torch.utils.data import DataLoader

CACHE_DIR = "datasets/.cache"
INPUT_TYPE = "power-spectral-difference"   # or "power-spectral-no-kinematic" / "kinematic"
HELD_OUT   = "0002"                         # held-out patient for TEST

train_ds, test_dss = load_train_test_datasets(
    prefix=CACHE_DIR,
    patient=HELD_OUT,
    task="regression",
    input_type=INPUT_TYPE,
    validation=False
)

def describe_field(name, val, sample_n=5):
    if isinstance(val, np.ndarray):
        print(f"  {name}: array{val.shape} {val.dtype}")
        if val.size == 0:
            print("    [WARN] empty array")
        if not np.isfinite(val).all():
            print("    [WARN] NaN/Inf detected")
    elif isinstance(val, (list, tuple)):
        n = len(val)
        print(f"  {name}: list n={n}")
        if n == 0:
            print("    [WARN] empty list")
            return
        # shape homogeneity check
        shapes = [getattr(x, "shape", None) for x in val]
        first_shape = shapes[0]
        homog = all(s == first_shape for s in shapes)
        print(f"    homogeneous_shapes={homog}, first_shape={first_shape}")
        # spot-check finiteness on a few items
        for i, x in enumerate(val[:min(sample_n, n)]):
            if isinstance(x, np.ndarray):
                fin = np.isfinite(x).all()
                print(f"    sample[{i}]: shape={x.shape}, finite={fin}")
            else:
                print(f"    sample[{i}]: type={type(x).__name__}")
    else:
        print(f"  {name}: {type(val).__name__}")

print("=== TRAIN ===")
print("len(train_ds) =", len(train_ds))
for k, v in train_ds.args.items():
    describe_field(k, v)

print("\n=== TEST SESSIONS ===")
for name, ds in test_dss.items():
    print(f"[{name}] len={len(ds)}")
    for k, v in ds.args.items():
        describe_field(k, v)

# Also: grab a real batch exactly like training sees it
print("\n=== One real batch (collated by DataLoader) ===")
dl = DataLoader(train_ds, batch_size=4, shuffle=True)
batch = next(iter(dl))
for k, v in batch.items():
    try:
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")
    except Exception:
        print(f"  {k}: {type(v).__name__} (len={len(v)})")
