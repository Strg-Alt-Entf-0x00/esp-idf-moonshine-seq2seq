import argparse
import math
import numpy as np
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Compare Golden Data vs Target Data")
    parser.add_argument("golden_bin", type=str, help="Path to golden .bin file (Python)")
    parser.add_argument("target_bin", type=str, help="Path to target .bin file (ESP32)")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Max absolute error tolerance")
    args = parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance must be a finite, non-negative number")

    print(f"Comparing:")
    print(f"  Golden: {args.golden_bin}")
    print(f"  Target: {args.target_bin}")

    try:
        golden_bytes = Path(args.golden_bin).read_bytes()
        target_bytes = Path(args.target_bin).read_bytes()
    except OSError as e:
        print(f"Error reading tensor files: {e}")
        sys.exit(1)

    if not golden_bytes or not target_bytes:
        print("[FAIL] Tensor files must not be empty")
        sys.exit(1)
    if len(golden_bytes) != len(target_bytes):
        print(f"[FAIL] Size mismatch! Golden: {len(golden_bytes)} bytes, Target: {len(target_bytes)} bytes")
        sys.exit(1)
    if len(golden_bytes) % np.dtype(np.float32).itemsize:
        print(f"[FAIL] File size is not a multiple of float32: {len(golden_bytes)} bytes")
        sys.exit(1)

    # Convert to float32 arrays
    golden = np.frombuffer(golden_bytes, dtype="<f4")
    target = np.frombuffer(target_bytes, dtype="<f4")

    if not np.isfinite(golden).all() or not np.isfinite(target).all():
        print("[FAIL] Tensor data contains NaN or infinite values")
        sys.exit(1)
    diff = np.abs(golden - target)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"\nStats over {len(golden)} floats:")
    print(f"  Max Absolute Error:  {max_diff:.8f}")
    print(f"  Mean Absolute Error: {mean_diff:.8f}")

    if max_diff > args.tolerance:
        print(f"\n[FAIL] Max difference {max_diff:.8f} exceeds tolerance {args.tolerance}")
        sys.exit(1)
    else:
        print(f"\n[PASS] Output matches within tolerance {args.tolerance}!")
        sys.exit(0)

if __name__ == '__main__':
    main()
