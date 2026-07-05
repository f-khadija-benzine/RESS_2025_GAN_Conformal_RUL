"""
01_explore_data.py — XJTU-SY Dataset Exploration
=================================================
Paste each section (separated by # %%) into a separate Colab cell.

Before running:
1. Upload XJTU-SY dataset to Google Drive
2. Mount Google Drive in Colab
3. Update DATA_ROOT below
"""

# %% ── Cell 0: Colab Setup (run ONCE) ────────────────────────────────
#
# PASTE THIS BLOCK INTO THE FIRST COLAB CELL:
#
#   !git clone https://github.com/YOUR_USERNAME/RESS_2025_GAN_Conformal_RUL.git
#   %cd RESS_2025_GAN_Conformal_RUL
#   %run notebooks/colab_setup.py
#
# After that, DATA_ROOT will be set automatically.
# If you already ran colab_setup.py, skip this cell.

# %% ── Cell 1: Imports ──────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# DATA_ROOT is set by colab_setup.py — override here if running locally
# DATA_ROOT = "/content/XJTU-SY"  # default Colab path

if 'DATA_ROOT' not in dir():
    DATA_ROOT = "/content/XJTU-SY"

print(f"DATA_ROOT = {DATA_ROOT}")
if os.path.exists(DATA_ROOT):
    print("Dataset found. Contents:")
    for item in sorted(os.listdir(DATA_ROOT)):
        print(f"  {item}/")
else:
    print("NOT found — run colab_setup.py first")

# %% ── Cell 2: Summary table ────────────────────────────────────────

from src.data_loader import XJTUSYLoader, OPERATING_CONDITIONS, CV_FOLDS

loader = XJTUSYLoader(DATA_ROOT)
summary_df = loader.summary()

print("=" * 70)
print("XJTU-SY BEARING DATASET SUMMARY")
print("=" * 70)
print(summary_df.to_string(index=False))
print(f"\nTotal bearings: {len(summary_df)}")
print(f"Total recordings: {summary_df['n_recordings'].sum()}")
print(f"Shortest: {summary_df['lifetime_min'].min()} min "
      f"({summary_df.loc[summary_df['lifetime_min'].idxmin(), 'short_id']})")
print(f"Longest:  {summary_df['lifetime_min'].max()} min "
      f"({summary_df.loc[summary_df['lifetime_min'].idxmax(), 'short_id']})")

# %% ── Cell 3: Lifetime bar chart ───────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 5))
colors = {1: '#2196F3', 2: '#FF9800', 3: '#4CAF50'}
ax.bar(summary_df['short_id'], summary_df['lifetime_min'],
       color=[colors[c] for c in summary_df['condition']], edgecolor='white')
ax.set_xlabel('Bearing ID')
ax.set_ylabel('Lifetime (minutes)')
ax.set_title('Bearing Lifetimes — XJTU-SY Dataset')
ax.tick_params(axis='x', rotation=45)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(facecolor=colors[1], label='Cond. 1 (35Hz, 12kN)'),
    Patch(facecolor=colors[2], label='Cond. 2 (37.5Hz, 11kN)'),
    Patch(facecolor=colors[3], label='Cond. 3 (40Hz, 10kN)'),
])
plt.tight_layout()
plt.savefig('results/figures/01_lifetime_distribution.png', dpi=150)
plt.show()

# %% ── Cell 4: Load a sample bearing ─────────────────────────────────

sample = loader.load_bearing("Bearing3_2")
signals = sample['raw_signals']
n_rec = signals.shape[0]
print(f"\nBearing: {sample['short_id']}")
print(f"Lifetime: {sample['lifetime_min']} min")
print(f"Failure mode: {sample['failure_mode']}")
print(f"Raw signals shape: {signals.shape}")

# %% ── Cell 5: Raw vibration at 3 life stages ───────────────────────

from src.data_loader import SAMPLES_PER_FILE, SAMPLING_RATE

early, mid, late = 1, n_rec // 2, n_rec - 2
time_ms = np.arange(SAMPLES_PER_FILE) / SAMPLING_RATE * 1000

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
for row, (idx, label) in enumerate(zip(
    [early, mid, late],
    [f'Early (t={early} min)', f'Mid (t={mid} min)', f'Near failure (t={late} min)']
)):
    axes[row, 0].plot(time_ms, signals[idx, :, 0], lw=0.3, color='#2196F3')
    axes[row, 0].set_ylabel('Amplitude (g)')
    axes[row, 0].set_title(f'{label} — Horizontal')
    axes[row, 1].plot(time_ms, signals[idx, :, 1], lw=0.3, color='#FF5722')
    axes[row, 1].set_title(f'{label} — Vertical')
for ax in axes[-1]:
    ax.set_xlabel('Time (ms)')
fig.suptitle(f'Raw Vibration — {sample["short_id"]} ({sample["failure_mode"]})',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/01_raw_vibration.png', dpi=150)
plt.show()

# %% ── Cell 6: RMS health indicator ─────────────────────────────────

rms_h = np.sqrt(np.mean(signals[:, :, 0]**2, axis=1))
rms_v = np.sqrt(np.mean(signals[:, :, 1]**2, axis=1))

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(np.arange(n_rec), rms_h, color='#2196F3', lw=1)
axes[0].set_title('RMS — Horizontal')
axes[0].set_ylabel('RMS (g)')
axes[1].plot(np.arange(n_rec), rms_v, color='#FF5722', lw=1)
axes[1].set_title('RMS — Vertical')
for ax in axes:
    ax.set_xlabel('Time (min)')
    ax.grid(True, alpha=0.3)
fig.suptitle(f'RMS Health Indicator — {sample["short_id"]}',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/01_rms_health_indicator.png', dpi=150)
plt.show()

# %% ── Cell 7: Frequency spectrum at 3 stages ───────────────────────

from scipy.fft import fft, fftfreq

freq = fftfreq(SAMPLES_PER_FILE, 1.0 / SAMPLING_RATE)
pos_freq = freq[:SAMPLES_PER_FILE // 2]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for col, (idx, label) in enumerate(zip(
    [early, mid, late], ['Early', 'Mid', 'Near failure']
)):
    spectrum = np.abs(fft(signals[idx, :, 0]))[:SAMPLES_PER_FILE // 2]
    axes[col].plot(pos_freq / 1000, spectrum, lw=0.3, color='#2196F3')
    axes[col].set_title(f'{label} (t={idx} min)')
    axes[col].set_xlabel('Frequency (kHz)')
    axes[col].set_xlim(0, 12)
    axes[col].grid(True, alpha=0.3)
axes[0].set_ylabel('Magnitude')
fig.suptitle(f'Frequency Spectrum — {sample["short_id"]} (Horizontal)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/01_frequency_spectrum.png', dpi=150)
plt.show()

# %% ── Cell 8: All 15 bearings RMS overview ──────────────────────────

fig, axes = plt.subplots(3, 5, figsize=(20, 10))
for cond in [1, 2, 3]:
    for j, bid in enumerate(OPERATING_CONDITIONS[cond]['bearings']):
        ax = axes[cond - 1, j]
        try:
            d = loader.load_bearing(bid, verbose=False)
            rms = np.sqrt(np.mean(d['raw_signals'][:, :, 0]**2, axis=1))
            ax.plot(np.arange(len(rms)), rms, lw=0.8, color=colors[cond])
            ax.set_title(f'{d["short_id"]}\n{d["failure_mode"]}\n{d["lifetime_min"]} min',
                         fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f'Error', transform=ax.transAxes, ha='center')
        if j == 0:
            ax.set_ylabel(f'Cond. {cond}\nRMS (g)', fontsize=9)
        if cond == 3:
            ax.set_xlabel('Time (min)', fontsize=8)
        ax.tick_params(labelsize=7)
fig.suptitle('RMS Overview — All 15 Bearings', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('results/figures/01_all_bearings_rms.png', dpi=150)
plt.show()

# %% ── Cell 9: Cross-validation folds ────────────────────────────────

print("\n" + "=" * 50)
print("CROSS-VALIDATION FOLDS")
print("=" * 50)
for fold in range(1, 6):
    train_ids, test_ids = loader.get_fold_split(fold)
    test_str = ", ".join([loader._short_id(b) for b in test_ids])
    print(f"Fold {fold} | Test: {test_str} | Train: {len(train_ids)} bearings")

# %% ── Cell 10: Data quality check ──────────────────────────────────

print("\n" + "=" * 50)
print("DATA QUALITY CHECK")
print("=" * 50)
all_data = loader.load_all(verbose=False)
for bid, data in all_data.items():
    sig = data['raw_signals']
    n_nan = np.isnan(sig).sum()
    n_inf = np.isinf(sig).sum()
    status = "OK" if (n_nan == 0 and n_inf == 0) else "ISSUE"
    print(f"  {data['short_id']:>5s}: {status} "
          f"(range: [{sig.min():.4f}, {sig.max():.4f}])")

print("\nStep 1 complete. Proceed to Step 2.")
