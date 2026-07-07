"""
data_loader.py — XJTU-SY Rolling Element Bearing Dataset Loader

Dataset: Xi'an Jiaotong University & Changxing Sumyoung Technology
15 bearings, 3 operating conditions, run-to-failure vibration data.
Each CSV file = one 1.28-second recording (25,600 Hz x 1.28 s = 32,768 samples).
Two channels: horizontal (col 0) and vertical (col 1) vibration.

Usage:
    from src.data_loader import XJTUSYLoader
    loader = XJTUSYLoader(data_root="data/raw/XJTU-SY")
    bearing_data = loader.load_bearing("1-1")
    all_data = loader.load_all()
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ── Dataset metadata ─────────────────────────────────────────────────

OPERATING_CONDITIONS = {
    1: {"speed_hz": 35.0, "speed_rpm": 2100, "load_kn": 12.0,
        "bearings": ["Bearing1_1", "Bearing1_2", "Bearing1_3",
                      "Bearing1_4", "Bearing1_5"]},
    2: {"speed_hz": 37.5, "speed_rpm": 2250, "load_kn": 11.0,
        "bearings": ["Bearing2_1", "Bearing2_2", "Bearing2_3",
                      "Bearing2_4", "Bearing2_5"]},
    3: {"speed_hz": 40.0, "speed_rpm": 2400, "load_kn": 10.0,
        "bearings": ["Bearing3_1", "Bearing3_2", "Bearing3_3",
                      "Bearing3_4", "Bearing3_5"]},
}

FAILURE_MODES = {
    "Bearing1_1": "Outer race",
    "Bearing1_2": "Outer race",
    "Bearing1_3": "Outer race",
    "Bearing1_4": "Cage",
    "Bearing1_5": "Outer race + Ball",
    "Bearing2_1": "Inner race",
    "Bearing2_2": "Outer race",
    "Bearing2_3": "Cage",
    "Bearing2_4": "Outer race",
    "Bearing2_5": "Outer race",
    "Bearing3_1": "Outer race",
    "Bearing3_2": "Inner race + Ball",
    "Bearing3_3": "Inner race",
    "Bearing3_4": "Inner race",
    "Bearing3_5": "Outer race",
}

SAMPLING_RATE = 25600
RECORDING_DURATION = 1.28
SAMPLES_PER_FILE = 32768
RECORDING_INTERVAL = 60

CV_FOLDS = {
    1: ["Bearing1_1", "Bearing2_1", "Bearing3_1"],
    2: ["Bearing1_2", "Bearing2_2", "Bearing3_2"],
    3: ["Bearing1_3", "Bearing2_3", "Bearing3_3"],
    4: ["Bearing1_4", "Bearing2_4", "Bearing3_4"],
    5: ["Bearing1_5", "Bearing2_5", "Bearing3_5"],
}


class XJTUSYLoader:
    """Loader for the XJTU-SY rolling element bearing dataset."""

    CONDITION_FOLDERS = {
        1: "35Hz12kN",
        2: "37.5Hz11kN",
        3: "40Hz10kN",
    }

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.data_root}\n"
                f"Download XJTU-SY from IEEE DataPort and place it there."
            )

    def _get_bearing_path(self, bearing_id: str) -> Path:
        bearing_id = self._normalize_id(bearing_id)
        condition = int(bearing_id.split("_")[0].replace("Bearing", ""))
        folder = self.CONDITION_FOLDERS[condition]
        path = self.data_root / folder / bearing_id
        if not path.exists():
            raise FileNotFoundError(f"Bearing folder not found: {path}")
        return path

    @staticmethod
    def _normalize_id(bearing_id: str) -> str:
        if bearing_id.startswith("Bearing"):
            return bearing_id
        parts = bearing_id.split("-")
        if len(parts) == 2:
            return f"Bearing{parts[0]}_{parts[1]}"
        raise ValueError(f"Invalid bearing ID: {bearing_id}")

    @staticmethod
    def _short_id(bearing_id: str) -> str:
        return bearing_id.replace("Bearing", "").replace("_", "-")

    def get_condition(self, bearing_id: str) -> int:
        bearing_id = self._normalize_id(bearing_id)
        return int(bearing_id.split("_")[0].replace("Bearing", ""))

    def get_failure_mode(self, bearing_id: str) -> str:
        bearing_id = self._normalize_id(bearing_id)
        return FAILURE_MODES.get(bearing_id, "Unknown")

    def get_operating_params(self, bearing_id: str) -> dict:
        cond = self.get_condition(bearing_id)
        return OPERATING_CONDITIONS[cond]

    def load_bearing(self, bearing_id: str, verbose: bool = True) -> Dict:
        """Load all recordings for a single bearing.

        Returns dict with keys:
            bearing_id, short_id, condition, speed_hz, load_kn,
            failure_mode, n_recordings, lifetime_min,
            raw_signals (shape: n_recordings x 32768 x 2)
        """
        bearing_id = self._normalize_id(bearing_id)
        path = self._get_bearing_path(bearing_id)

        csv_files = sorted(path.glob("*.csv"), key=lambda f: int(f.stem))
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {path}")

        n_files = len(csv_files)
        if verbose:
            print(f"Loading {bearing_id} ({self._short_id(bearing_id)}): "
                  f"{n_files} recordings, "
                  f"failure mode: {self.get_failure_mode(bearing_id)}")

        raw_signals = np.zeros((n_files, SAMPLES_PER_FILE, 2), dtype=np.float32)

        for i, csv_file in enumerate(csv_files):
            try:
                df = pd.read_csv(csv_file)
                if df.shape[1] >= 2:
                    n_samples = min(df.shape[0], SAMPLES_PER_FILE)
                    raw_signals[i, :n_samples, 0] = df.iloc[:n_samples, 0].values.astype(np.float32)
                    raw_signals[i, :n_samples, 1] = df.iloc[:n_samples, 1].values.astype(np.float32)
            except Exception as e:
                print(f"  ERROR loading {csv_file.name}: {e}")

        params = self.get_operating_params(bearing_id)

        result = {
            "bearing_id": bearing_id,
            "short_id": self._short_id(bearing_id),
            "condition": self.get_condition(bearing_id),
            "speed_hz": params["speed_hz"],
            "load_kn": params["load_kn"],
            "failure_mode": self.get_failure_mode(bearing_id),
            "n_recordings": n_files,
            "lifetime_min": n_files,
            "raw_signals": raw_signals,
        }

        if verbose:
            mem_mb = raw_signals.nbytes / 1e6
            print(f"  -> Shape: {raw_signals.shape}, Memory: {mem_mb:.1f} MB")

        return result

    def load_all(self, conditions: Optional[List[int]] = None,
                 verbose: bool = True) -> Dict[str, Dict]:
        """Load all bearings (or subset by condition)."""
        all_data = {}
        for cond_num, cond_info in OPERATING_CONDITIONS.items():
            if conditions is not None and cond_num not in conditions:
                continue
            if verbose:
                print(f"\n=== Condition {cond_num}: "
                      f"{cond_info['speed_hz']} Hz, {cond_info['load_kn']} kN ===")
            for bearing_id in cond_info["bearings"]:
                try:
                    data = self.load_bearing(bearing_id, verbose=verbose)
                    all_data[bearing_id] = data
                except FileNotFoundError as e:
                    print(f"  SKIPPING {bearing_id}: {e}")

        if verbose:
            print(f"\n=== Loaded {len(all_data)} bearings ===")
            total_mem = sum(d["raw_signals"].nbytes for d in all_data.values()) / 1e6
            print(f"Total memory: {total_mem:.1f} MB")
        return all_data

    def get_fold_split(self, fold: int) -> Tuple[List[str], List[str]]:
        """Return (train_ids, test_ids) for a given CV fold (1-5)."""
        if fold not in CV_FOLDS:
            raise ValueError(f"Fold must be 1-5, got {fold}")
        test_ids = CV_FOLDS[fold]
        train_ids = [bid for f, bids in CV_FOLDS.items()
                     for bid in bids if f != fold]
        return train_ids, test_ids

    def summary(self) -> pd.DataFrame:
        """Summary DataFrame of all bearings without loading raw data."""
        rows = []
        for cond_num, cond_info in OPERATING_CONDITIONS.items():
            for bearing_id in cond_info["bearings"]:
                try:
                    path = self._get_bearing_path(bearing_id)
                    n_files = len(list(path.glob("*.csv")))
                except FileNotFoundError:
                    n_files = 0
                rows.append({
                    "bearing_id": bearing_id,
                    "short_id": self._short_id(bearing_id),
                    "condition": cond_num,
                    "speed_hz": cond_info["speed_hz"],
                    "load_kn": cond_info["load_kn"],
                    "failure_mode": FAILURE_MODES.get(bearing_id, "Unknown"),
                    "n_recordings": n_files,
                    "lifetime_min": n_files,
                    "cv_fold": next(f for f, bids in CV_FOLDS.items()
                                    if bearing_id in bids),
                })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw/XJTU-SY"
    loader = XJTUSYLoader(root)
    print(loader.summary().to_string(index=False))
