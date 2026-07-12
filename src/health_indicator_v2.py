"""
health_indicator.py — Health Indicator Construction and Stage Labeling

Two functions:
  1. compute_scalar_hi()  → scalar HI per recording (for FPT/EOL/stage detection)
  2. extract_features()   → multivariate features per recording (for model input)

Stage labels are derived automatically from the scalar HI using FPT/EOL thresholds.
The multivariate features inherit these stage labels.

Usage:
    from health_indicator import HealthIndicatorPipeline
    pipeline = HealthIndicatorPipeline()
    result = pipeline.process_bearing(bearing_data)
"""

import numpy as np
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional


# ── Time-domain feature functions ────────────────────────────────────

def rms(x):
    """Root Mean Square."""
    return np.sqrt(np.mean(x**2))

def peak(x):
    """Maximum absolute amplitude."""
    return np.max(np.abs(x))

def peak_to_peak(x):
    """Peak-to-peak amplitude."""
    return np.max(x) - np.min(x)

def crest_factor(x):
    """Peak / RMS — detects impulsive signals."""
    r = rms(x)
    return peak(x) / r if r > 0 else 0.0

def shape_factor(x):
    """RMS / mean absolute value."""
    ma = np.mean(np.abs(x))
    return rms(x) / ma if ma > 0 else 0.0

def impulse_factor(x):
    """Peak / mean absolute value."""
    ma = np.mean(np.abs(x))
    return peak(x) / ma if ma > 0 else 0.0

def clearance_factor(x):
    """Peak / (mean of sqrt(|x|))^2."""
    mean_sqrt = np.mean(np.sqrt(np.abs(x)))
    return peak(x) / (mean_sqrt**2) if mean_sqrt > 0 else 0.0

def kurtosis(x):
    """4th moment / variance^2 — measures tail heaviness."""
    m = np.mean(x)
    var = np.mean((x - m)**2)
    if var == 0:
        return 0.0
    return np.mean((x - m)**4) / (var**2)

def skewness(x):
    """3rd moment — measures asymmetry."""
    m = np.mean(x)
    var = np.mean((x - m)**2)
    if var == 0:
        return 0.0
    return np.mean((x - m)**3) / (var**1.5)


# ── Feature names (for reference) ────────────────────────────────────

FEATURE_NAMES_PER_CHANNEL = [
    'rms', 'peak', 'peak_to_peak', 'std', 'kurtosis',
    'skewness', 'crest_factor', 'shape_factor',
    'impulse_factor', 'clearance_factor'
]

def get_all_feature_names():
    """Return full list of feature names for both channels."""
    names = []
    for ch in ['horiz', 'vert']:
        for feat in FEATURE_NAMES_PER_CHANNEL:
            names.append(f'{ch}_{feat}')
    return names


# ── Core extraction functions ────────────────────────────────────────

def extract_features_single(signal_1d):
    """Extract 10 time-domain features from a 1D signal.

    Args:
        signal_1d: 1D array of shape (n_samples,)

    Returns:
        1D array of shape (10,)
    """
    return np.array([
        rms(signal_1d),
        peak(signal_1d),
        peak_to_peak(signal_1d),
        np.std(signal_1d),
        kurtosis(signal_1d),
        skewness(signal_1d),
        crest_factor(signal_1d),
        shape_factor(signal_1d),
        impulse_factor(signal_1d),
        clearance_factor(signal_1d),
    ], dtype=np.float32)


def extract_features_all(raw_signals):
    """Extract multivariate features from all recordings of a bearing.

    Args:
        raw_signals: array of shape (n_recordings, n_samples, 2)

    Returns:
        features: array of shape (n_recordings, 20)
            10 features per channel × 2 channels
    """
    n_rec = raw_signals.shape[0]
    n_features = len(FEATURE_NAMES_PER_CHANNEL) * 2  # 20
    features = np.zeros((n_rec, n_features), dtype=np.float32)

    for i in range(n_rec):
        # Horizontal channel (column 0)
        feat_h = extract_features_single(raw_signals[i, :, 0])
        # Vertical channel (column 1)
        feat_v = extract_features_single(raw_signals[i, :, 1])
        # Concatenate: [horiz_features | vert_features]
        features[i] = np.concatenate([feat_h, feat_v])

    return features


def compute_velocity_rms(raw_signals, shaft_freq_hz, sampling_rate=25600):
    """Compute velocity-domain RMS health indicator (Lu et al. 2022).

    Converts acceleration to velocity via integration in frequency domain,
    then computes RMS in the band [0.2*shaft_freq, fs/2].

    Args:
        raw_signals: array of shape (n_recordings, n_samples, 2)
        shaft_freq_hz: shaft rotational frequency in Hz
        sampling_rate: sampling rate in Hz

    Returns:
        hi_horiz: 1D array of shape (n_recordings,) — horizontal channel HI
        hi_vert:  1D array of shape (n_recordings,) — vertical channel HI
    """
    n_rec = raw_signals.shape[0]
    n_samples = raw_signals.shape[1]
    hi_horiz = np.zeros(n_rec, dtype=np.float64)
    hi_vert = np.zeros(n_rec, dtype=np.float64)

    freq = fftfreq(n_samples, 1.0 / sampling_rate)
    freq_low = 0.2 * shaft_freq_hz
    freq_high = sampling_rate / 2.0

    # Band mask (positive frequencies only)
    band_mask = (np.abs(freq) >= freq_low) & (np.abs(freq) <= freq_high)

    for i in range(n_rec):
        for ch, hi_arr in enumerate([hi_horiz, hi_vert]):
            accel = raw_signals[i, :, ch]

            # FFT of acceleration
            accel_fft = fft(accel)

            # Convert acceleration to velocity in frequency domain:
            # V(f) = A(f) / (j * 2 * pi * f)
            # Avoid division by zero at f=0
            omega = 2.0 * np.pi * freq
            omega[0] = 1.0  # avoid div by zero, DC component will be zeroed
            vel_fft = accel_fft / (1j * omega)
            vel_fft[0] = 0.0  # remove DC

            # RMS in the specified frequency band (Parseval's theorem)
            vel_spectrum = np.abs(vel_fft)
            band_energy = np.sum(vel_spectrum[band_mask]**2) / 2.0
            hi_arr[i] = np.sqrt(band_energy / n_samples)

    return hi_horiz, hi_vert


def compute_simple_rms(raw_signals):
    """Compute simple time-domain RMS per recording per channel.

    Simpler alternative to velocity-domain RMS. Uses raw acceleration directly.

    Args:
        raw_signals: array of shape (n_recordings, n_samples, 2)

    Returns:
        rms_horiz: 1D array of shape (n_recordings,)
        rms_vert:  1D array of shape (n_recordings,)
    """
    rms_horiz = np.sqrt(np.mean(raw_signals[:, :, 0]**2, axis=1))
    rms_vert = np.sqrt(np.mean(raw_signals[:, :, 1]**2, axis=1))
    return rms_horiz, rms_vert


def smooth(signal, window=3):
    """Moving average smoothing."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.pad(signal, (window // 2, window // 2), mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(signal)]


# ── Stage labeling ───────────────────────────────────────────────────

def detect_fpt(hi, n_consecutive=5, min_relative_rise=0.10):
    """Detect First Prediction Time using a guarded 3-sigma method.

    FPT = first time `n_consecutive` consecutive HI values exceed BOTH:
      (a) the statistical threshold  mu + 3*sigma, and
      (b) a relative floor          mu * (1 + min_relative_rise),
    where mu and sigma are estimated from the first 20% of the trajectory
    (assumed healthy baseline).

    WHY TWO GUARDS
    --------------
    The plain 3-sigma rule fails on long-lived bearings. When a bearing runs
    for thousands of minutes its baseline is extremely stable, so sigma is
    tiny and mu + 3*sigma sits only a hair above the mean. Ordinary
    operational noise (settling transients, lubrication shifts) then crosses
    it within the first few minutes, and the bearing gets labelled
    "degrading" for ~99% of its life. On XJTU-SY the unguarded rule produced
    FPT = 23 min for Bearing 3-1 (life 2537 min) and FPT = 169 min for
    Bearing 3-2 (life 2495 min) -- both physically implausible.

    The guards target that failure mode directly:
      * Persistence (n_consecutive): a transient blip cannot stay above the
        threshold for many consecutive samples; genuine degradation can.
      * Relative floor (min_relative_rise): the HI must rise by a meaningful
        FRACTION of its own baseline, not merely by a few tiny sigmas. This
        is what prevents a razor-thin threshold from being tripped by noise
        when sigma is very small.

    Args:
        hi: 1D health indicator array
        n_consecutive: consecutive exceedances required (persistence guard)
        min_relative_rise: fractional rise above baseline mean required
                           (0.10 = HI must exceed 110% of baseline mu)

    Returns:
        fpt_idx: index of FPT
        detected: True if a genuine crossing was found,
                  False if the 80%-of-life fallback was used
    """
    # Baseline statistics from the first 20% (healthy) of the trajectory
    baseline_end = max(int(len(hi) * 0.2), 10)
    mu = np.mean(hi[:baseline_end])
    sigma = np.std(hi[:baseline_end])

    # Both conditions must hold -> take the stricter (higher) threshold
    thr_sigma = mu + 3.0 * sigma
    thr_relative = mu * (1.0 + min_relative_rise)
    threshold = max(thr_sigma, thr_relative)

    # First run of n_consecutive samples above the threshold
    count = 0
    for i in range(len(hi)):
        if hi[i] > threshold:
            count += 1
            if count >= n_consecutive:
                return i - n_consecutive + 1, True
        else:
            count = 0

    # Never crossed: fall back to 80% of life, flagged as NOT detected
    return int(len(hi) * 0.8), False


def detect_acceleration_point(hi, fpt_idx, eol_idx, sigma_factor=1.0):
    """Detect the acceleration point within the degradation phase.

    The acceleration point is where the first-difference of HI
    exceeds mean + sigma_factor * std of the Stage 2 trend.

    Args:
        hi: 1D health indicator array
        fpt_idx: FPT index
        eol_idx: EOL index
        sigma_factor: multiplier for std threshold

    Returns:
        acc_idx: acceleration point index
    """
    if eol_idx - fpt_idx < 5:
        # Too short to split — put acceleration at midpoint
        return (fpt_idx + eol_idx) // 2

    # Compute first-difference in the degradation phase
    degrad_hi = hi[fpt_idx:eol_idx]
    first_diff = np.diff(degrad_hi)

    if len(first_diff) < 3:
        return (fpt_idx + eol_idx) // 2

    # Use first half of degradation as "early degradation" reference
    half = len(first_diff) // 2
    if half < 2:
        return (fpt_idx + eol_idx) // 2

    early_diff = first_diff[:half]
    mu_diff = np.mean(early_diff)
    sigma_diff = np.std(early_diff)
    threshold = mu_diff + sigma_factor * sigma_diff

    # Find first point in second half where diff exceeds threshold
    for i in range(half, len(first_diff)):
        if first_diff[i] > threshold:
            return fpt_idx + i

    # Default: 70% through degradation phase
    return fpt_idx + int(0.7 * (eol_idx - fpt_idx))


def assign_stage_labels(n_recordings, fpt_idx, acc_idx, eol_idx):
    """Assign stage labels to every recording.

    Stage 1 (healthy):          0 to fpt_idx - 1
    Stage 2 (early degradation): fpt_idx to acc_idx - 1
    Stage 3 (near-failure):      acc_idx to eol_idx

    Args:
        n_recordings: total number of recordings
        fpt_idx: First Prediction Time index
        acc_idx: acceleration point index
        eol_idx: End of Life index (= n_recordings - 1)

    Returns:
        labels: 1D integer array of shape (n_recordings,) with values 1, 2, or 3
    """
    labels = np.ones(n_recordings, dtype=np.int32)  # default Stage 1
    labels[fpt_idx:acc_idx] = 2
    labels[acc_idx:] = 3
    return labels


# ── Main pipeline ────────────────────────────────────────────────────

class HealthIndicatorPipeline:
    """Complete pipeline: raw signals → features + stage labels."""

    def __init__(self, hi_method='simple_rms', smoothing_window=3,
                 fpt_consecutive=5, fpt_min_relative_rise=0.10,
                 acceleration_sigma=1.0):
        """
        Args:
            hi_method: 'simple_rms' or 'velocity_rms'
            smoothing_window: moving average window for HI smoothing
            fpt_consecutive: consecutive exceedances for FPT detection
            fpt_min_relative_rise: fractional rise above baseline required
            acceleration_sigma: sigma factor for acceleration point
        """
        self.hi_method = hi_method
        self.smoothing_window = smoothing_window
        self.fpt_consecutive = fpt_consecutive
        self.fpt_min_relative_rise = fpt_min_relative_rise
        self.acceleration_sigma = acceleration_sigma

    def process_bearing(self, bearing_data, verbose=True):
        """Process a single bearing: extract features and assign stages.

        Args:
            bearing_data: dict from XJTUSYLoader.load_bearing()
            verbose: print progress

        Returns:
            dict with keys:
                - 'bearing_id': str
                - 'features': array (n_recordings, 20) multivariate features
                - 'feature_names': list of 20 feature names
                - 'hi_horiz': 1D HI (horizontal)
                - 'hi_vert': 1D HI (vertical)
                - 'hi_smoothed': 1D smoothed HI (used for stage detection)
                - 'stage_labels': 1D integer array (1, 2, or 3)
                - 'fpt_idx': FPT index
                - 'acc_idx': acceleration point index
                - 'eol_idx': EOL index
                - 'rul': 1D RUL array (countdown from EOL)
                - 'stage_counts': dict with count per stage
        """
        raw = bearing_data['raw_signals']
        bid = bearing_data['bearing_id']
        n_rec = raw.shape[0]

        if verbose:
            print(f"Processing {bid}...")

        # 1. Extract multivariate features (model input)
        features = extract_features_all(raw)

        # 2. Compute scalar HI (for stage detection only)
        if self.hi_method == 'velocity_rms':
            shaft_freq = bearing_data['speed_hz']
            hi_h, hi_v = compute_velocity_rms(raw, shaft_freq)
        else:
            hi_h, hi_v = compute_simple_rms(raw)

        # Use horizontal channel HI for stage detection
        # (following Lu et al. 2022 — horizontal is more responsive to radial load)
        hi_raw = hi_h.copy()
        hi_sm = smooth(hi_raw, self.smoothing_window)

        # 3. Detect FPT and EOL
        fpt_idx, fpt_detected = detect_fpt(
            hi_sm, self.fpt_consecutive, self.fpt_min_relative_rise)
        eol_idx = n_rec - 1  # last recording = failure

        # 4. Detect acceleration point
        acc_idx = detect_acceleration_point(
            hi_sm, fpt_idx, eol_idx, self.acceleration_sigma
        )

        # Ensure ordering: 0 <= fpt <= acc <= eol
        fpt_idx = max(0, min(fpt_idx, eol_idx - 2))
        acc_idx = max(fpt_idx + 1, min(acc_idx, eol_idx))

        # 5. Assign stage labels
        stage_labels = assign_stage_labels(n_rec, fpt_idx, acc_idx, eol_idx)

        # 6. Compute RUL (countdown from EOL)
        rul = np.arange(n_rec - 1, -1, -1, dtype=np.float32)

        # Stage distribution
        stage_counts = {
            1: int(np.sum(stage_labels == 1)),
            2: int(np.sum(stage_labels == 2)),
            3: int(np.sum(stage_labels == 3)),
        }

        if verbose:
            flag = "detected" if fpt_detected else "FALLBACK"
            print(f"  FPT: t={fpt_idx} min [{flag}] | Acc: t={acc_idx} min | EOL: t={eol_idx} min")
            print(f"  Stages: S1={stage_counts[1]}, S2={stage_counts[2]}, S3={stage_counts[3]}")

        return {
            'bearing_id': bid,
            'features': features,
            'feature_names': get_all_feature_names(),
            'hi_horiz': hi_h,
            'hi_vert': hi_v,
            'hi_smoothed': hi_sm,
            'stage_labels': stage_labels,
            'fpt_idx': fpt_idx,
            'fpt_detected': fpt_detected,
            'acc_idx': acc_idx,
            'eol_idx': eol_idx,
            'rul': rul,
            'stage_counts': stage_counts,
        }

    def process_all(self, all_bearing_data, verbose=True):
        """Process all bearings.

        Args:
            all_bearing_data: dict from XJTUSYLoader.load_all()

        Returns:
            dict mapping bearing_id → processed result
        """
        results = {}
        total_stages = {1: 0, 2: 0, 3: 0}

        for bid, bdata in all_bearing_data.items():
            result = self.process_bearing(bdata, verbose=verbose)
            results[bid] = result
            for s in [1, 2, 3]:
                total_stages[s] += result['stage_counts'][s]

        if verbose:
            total = sum(total_stages.values())
            print(f"\n=== GLOBAL STAGE DISTRIBUTION ===")
            for s in [1, 2, 3]:
                pct = 100 * total_stages[s] / total
                print(f"  Stage {s}: {total_stages[s]} ({pct:.1f}%)")
            print(f"  Total: {total}")
            n_fb = sum(1 for r in results.values() if not r['fpt_detected'])
            print(f"  FPT detected: {len(results) - n_fb}/{len(results)} bearings"
                  f"{' (' + str(n_fb) + ' fallback)' if n_fb else ''}")

        return results
