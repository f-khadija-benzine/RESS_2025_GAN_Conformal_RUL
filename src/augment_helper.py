"""
augment_helper.py — Step 6b GAN-augmentation utilities (notebook-side)

Kept out of src/ deliberately: this is experiment-orchestration glue for the
+GAN ablation row, not a core library component. It calls the frozen
generators (gan.py) and the baseline trainer (model.py) without modifying
either.

Central function: build_augmented_training_set(). It takes one fold's real
training tensors and a loaded StageGAN, synthesises near-failure windows,
assigns each a RUL label by a nearest-neighbour tie to a real Stage-3 window,
and returns augmented (X, y_rul, y_stage) ready for RULTrainer.fit.

Why a nearest-neighbour tie rather than independent marginal sampling:
The generator is conditioned on STAGE only, not on RUL, so a synthetic window
carries no intrinsic RUL. Sampling a RUL independently from the real Stage-3
marginal keeps the label distribution honest but severs any link between a
window's content and its label — pure label noise on the RUL axis, which gives
+GAN little reason to sharpen regression. Borrowing the RUL of the nearest real
Stage-3 window (in scaled feature space) restores a weak but genuine
content->label correlation at ~no cost and no GAN change, so the near-failure
RUL *slope* is at least partially preserved in the synthetic set.
"""

import numpy as np
from scipy.spatial import cKDTree

# Match the clip applied to real windows in windowing.apply_scaler (log_clip).
# The generator has no output activation, so its samples are not bounded to
# this range; real training windows are. Clipping synthetic windows to the
# same support (a) puts them in the range the encoder saw at baseline and
# (b) makes the kNN distances below meaningful, since the tree is built from
# clipped real windows.
CLIP_SIGMA = 5.0


def _flatten(X):
    """(N, T, F) -> (N, T*F) for distance computation."""
    return np.asarray(X, dtype=np.float32).reshape(len(X), -1)


def synth_rul_by_knn(X_syn, X_real_s3, y_rul_real_s3):
    """Assign a RUL to each synthetic window by nearest real Stage-3 window.

    Args:
        X_syn:        (M, T, F) synthetic near-failure windows (already clipped)
        X_real_s3:    (R, T, F) real Stage-3 training windows
        y_rul_real_s3:(R,)      their RUL labels

    Returns:
        (M,) RUL labels, each borrowed from the synthetic window's nearest
        real Stage-3 neighbour in flattened feature space.
    """
    if len(X_real_s3) == 0:
        raise ValueError("no real Stage-3 windows to borrow RUL from")
    tree = cKDTree(_flatten(X_real_s3))
    _, idx = tree.query(_flatten(X_syn), k=1)
    return np.asarray(y_rul_real_s3, dtype=np.float32)[idx]


def build_augmented_training_set(gan, X_train, y_rul_train, y_stage_train,
                                 augment_ratio=1.0, target_stage=2,
                                 seed=42, verbose=True):
    """Augment one fold's training set with synthetic near-failure windows.

    Args:
        gan:            a loaded StageGAN (generator in eval-ready state)
        X_train:        (N, T, F) scaled real training windows
        y_rul_train:    (N,) real RUL labels in [0, 1]
        y_stage_train:  (N,) 0-indexed stage labels {0,1,2}
        augment_ratio:  synthetic count = ratio * (# real target-stage windows)
        target_stage:   0-indexed stage to synthesise (2 = near-failure / S3)
        seed:           RNG seed for the final shuffle (reproducible ablation)

    Returns:
        X_aug, y_rul_aug, y_stage_aug — real + synthetic, shuffled together by
        one shared permutation so the three arrays stay aligned. When
        augment_ratio == 0 the inputs are returned unchanged (as arrays).
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    y_rul_train = np.asarray(y_rul_train, dtype=np.float32)
    y_stage_train = np.asarray(y_stage_train, dtype=np.int64)

    real_tgt = y_stage_train == target_stage
    n_real_tgt = int(real_tgt.sum())
    n_syn = int(round(augment_ratio * n_real_tgt))

    if n_syn == 0 or n_real_tgt == 0:
        if verbose:
            print(f"  augment: ratio={augment_ratio}, "
                  f"real S3={n_real_tgt} -> 0 synthetic (returning real only)")
        return X_train, y_rul_train, y_stage_train

    # 1. sample synthetic windows from the frozen generator
    X_syn = gan.sample(n_syn, target_stage)                     # (M, T, F)

    # 2. clip to the same support as the real (log_clip) windows
    X_syn = np.clip(X_syn, -CLIP_SIGMA, CLIP_SIGMA).astype(np.float32)

    # 3. borrow a RUL from the nearest real Stage-3 window
    y_rul_syn = synth_rul_by_knn(
        X_syn, X_train[real_tgt], y_rul_train[real_tgt])
    y_stage_syn = np.full(n_syn, target_stage, dtype=np.int64)

    # 4. concatenate real + synthetic
    X_aug = np.concatenate([X_train, X_syn], axis=0)
    y_rul_aug = np.concatenate([y_rul_train, y_rul_syn], axis=0)
    y_stage_aug = np.concatenate([y_stage_train, y_stage_syn], axis=0)

    # 5. one shared permutation keeps the three arrays aligned
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_aug))

    if verbose:
        print(f"  augment: ratio={augment_ratio}, real S3={n_real_tgt} "
              f"-> +{n_syn} synthetic  (train {len(X_train)} -> {len(X_aug)})")
        print(f"           synthetic RUL borrowed by kNN: "
              f"mean {y_rul_syn.mean():.3f}  "
              f"[{y_rul_syn.min():.3f}, {y_rul_syn.max():.3f}]  "
              f"| real S3 RUL mean {y_rul_train[real_tgt].mean():.3f}")

    return X_aug[perm], y_rul_aug[perm], y_stage_aug[perm]
