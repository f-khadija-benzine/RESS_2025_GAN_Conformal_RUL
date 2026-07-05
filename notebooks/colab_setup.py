"""
colab_setup.py — Run this FIRST in any Colab notebook
======================================================
Mounts Google Drive, locates XJTU-SY dataset, sets up paths.

FIRST TIME ONLY:
  1. Download XJTU-SY from https://www.kaggle.com/datasets/zwming/xjtu-sy
  2. Upload the zip to your Google Drive
  3. Extract it so you have: MyDrive/datasets/XJTU-SY/35Hz12kN/ etc.

EVERY TIME:
  Paste this as the first cell in Colab:

    !git clone https://github.com/YOUR_USERNAME/RESS_2025_GAN_Conformal_RUL.git
    %cd RESS_2025_GAN_Conformal_RUL
    %run notebooks/colab_setup.py
"""

import os
import sys
import subprocess

# ── 1. Mount Google Drive ────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted.")

# ── 2. Locate XJTU-SY on Drive ──────────────────────────────────────

# UPDATE THIS if you put the dataset in a different Drive folder
DRIVE_DATA_PATH = "/content/drive/MyDrive/datasets/XJTU-SY"

# Common alternative paths to check automatically
SEARCH_PATHS = [
    "/content/drive/MyDrive/datasets/XJTU-SY",
    "/content/drive/MyDrive/XJTU-SY",
    "/content/drive/MyDrive/data/XJTU-SY",
    "/content/drive/MyDrive/XJTU-SY_Bearing_Datasets",
    "/content/drive/MyDrive/datasets/XJTU-SY_Bearing_Datasets",
]

EXPECTED_FOLDERS = {"35Hz12kN", "37.5Hz11kN", "40Hz10kN"}
DATA_ROOT = None

# Search for the dataset
for path in SEARCH_PATHS:
    if os.path.exists(path):
        contents = set(os.listdir(path))
        if EXPECTED_FOLDERS.issubset(contents):
            DATA_ROOT = path
            break
        # Check one level deeper
        for sub in os.listdir(path):
            sub_path = os.path.join(path, sub)
            if os.path.isdir(sub_path):
                sub_contents = set(os.listdir(sub_path))
                if EXPECTED_FOLDERS.issubset(sub_contents):
                    DATA_ROOT = sub_path
                    break

if DATA_ROOT:
    print(f"XJTU-SY found at: {DATA_ROOT}")
else:
    print("="*60)
    print("XJTU-SY NOT FOUND on Google Drive!")
    print("="*60)
    print()
    print("Do this ONCE:")
    print("  1. Download from: https://www.kaggle.com/datasets/zwming/xjtu-sy")
    print("  2. Upload the zip to Google Drive")
    print("  3. Extract it so you have:")
    print("     MyDrive/datasets/XJTU-SY/")
    print("       ├── 35Hz12kN/")
    print("       ├── 37.5Hz11kN/")
    print("       └── 40Hz10kN/")
    print()
    print("  Or extract directly in Colab (run once):")
    print("    !mkdir -p /content/drive/MyDrive/datasets/XJTU-SY")
    print("    !unzip /content/drive/MyDrive/xjtu-sy.zip -d /content/drive/MyDrive/datasets/XJTU-SY/")
    print()
    print("Then re-run this setup script.")
    print("="*60)

# ── 3. Add src to Python path ────────────────────────────────────────

PROJECT_ROOT = os.getcwd()
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

# ── 4. Install dependencies ──────────────────────────────────────────

subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)
print("Dependencies installed.")

# ── 5. Verify dataset ────────────────────────────────────────────────

if DATA_ROOT:
    from src.data_loader import XJTUSYLoader
    try:
        loader = XJTUSYLoader(DATA_ROOT)
        summary = loader.summary()
        total = summary['n_recordings'].sum()
        print(f"\nVerification: {len(summary)} bearings, {total} total recordings")
        print("Setup complete. Ready to go.")
    except Exception as e:
        print(f"\nVerification failed: {e}")

print(f"\nDATA_ROOT = '{DATA_ROOT}'")
