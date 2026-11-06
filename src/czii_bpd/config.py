import os
from pathlib import Path


def _data_dir():
    for p in (
        os.environ.get("CZII_DATA"),
        "/kaggle/input/competitions/czii-cryo-et-object-identification",
        "/kaggle/input/czii-cryo-et-object-identification",
    ):
        if p and Path(p).exists():
            return Path(p)
    return Path("data")


DATA_DIR = _data_dir()
RUNS = ["TS_5_4", "TS_69_2", "TS_6_4", "TS_6_6", "TS_73_6", "TS_86_3", "TS_99_9"]

SCALE = 10.012444196428572  # Å per voxel
PATCH = (128, 128, 128)
NUM_CLASSES = 6
P_LO, P_HI = -1.1769942479337e-05, 1.2801160441345688e-05  # mean 5th / 99th percentiles of the training runs
Q_LO, Q_HI = -0.5, 1.5  # uint8 range in normalised units, wider than [0, 1] to keep the tails

# name: (class index, radius in Å, probability threshold)
PARTICLES = {
    "apo-ferritin": (1, 60, 0.0575),
    "beta-galactosidase": (2, 90, 0.0475),
    "ribosome": (3, 150, 0.0575),
    "thyroglobulin": (4, 130, 0.0475),
    "virus-like-particle": (5, 135, 0.125),
}
WEIGHTS = {"apo-ferritin": 1, "beta-galactosidase": 2, "ribosome": 1, "thyroglobulin": 2, "virus-like-particle": 1}
BLOB_FACTOR = 7  # minimum component size = label sphere volume / BLOB_FACTOR
STRIDE = {"full": 1, "half": 2}  # output voxel size of each variant, in input voxels
