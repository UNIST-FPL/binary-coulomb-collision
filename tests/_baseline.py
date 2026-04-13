from pathlib import Path

import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "data"


def load_baseline(name: str):
    return np.load(DATA_DIR / name, allow_pickle=False)
