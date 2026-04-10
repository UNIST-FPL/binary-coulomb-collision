import argparse
from pathlib import Path

import numpy as np


def summarize_diff(reference_path: Path, candidate_path: Path):
    reference = np.load(reference_path, allow_pickle=False)
    candidate = np.load(candidate_path, allow_pickle=False)

    reference_keys = set(reference.files)
    candidate_keys = set(candidate.files)
    if reference_keys != candidate_keys:
        missing = sorted(reference_keys - candidate_keys)
        extra = sorted(candidate_keys - reference_keys)
        print(f"{reference_path.name}: key mismatch")
        if missing:
            print(f"  missing in candidate: {missing}")
        if extra:
            print(f"  extra in candidate: {extra}")
        return

    print(reference_path.name)
    for key in sorted(reference.files):
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            print(f"  {key}: shape mismatch {ref.shape} vs {cand.shape}")
            continue

        abs_diff = np.abs(cand - ref)
        max_abs = float(np.max(abs_diff))
        denom = np.maximum(np.abs(ref), 1.0e-30)
        max_rel = float(np.max(abs_diff / denom))
        print(f"  {key}: max_abs={max_abs:.6e}, max_rel={max_rel:.6e}")


def main():
    parser = argparse.ArgumentParser(description="Summarize differences between two baseline directories.")
    parser.add_argument("reference_dir", type=Path, help="Reference baseline directory.")
    parser.add_argument("candidate_dir", type=Path, help="Candidate baseline directory.")
    args = parser.parse_args()

    reference_files = sorted(args.reference_dir.glob("*.npz"))
    candidate_names = {path.name for path in args.candidate_dir.glob("*.npz")}

    for reference_path in reference_files:
        candidate_path = args.candidate_dir / reference_path.name
        if reference_path.name not in candidate_names:
            print(f"{reference_path.name}: missing in candidate directory")
            continue
        summarize_diff(reference_path, candidate_path)

    extra_files = sorted(candidate_names - {path.name for path in reference_files})
    for extra in extra_files:
        print(f"{extra}: extra file in candidate directory")


if __name__ == "__main__":
    main()
