"""
gan.py — Stage-Conditioned WGAN-GP for Near-Failure Augmentation (C1)

Step 4 of the RESS pipeline.

Generates synthetic multivariate sensor windows conditioned on degradation
stage, to enrich the scarce near-failure regime in the training set.

Design choices and their reasons:

  * WGAN-GP loss. A plain GAN can mode-collapse — the generator finds one
    window that fools the discriminator and emits copies of it forever.
    That would be fatal here: 1000 near-identical synthetic windows add no
    information, augmentation would not improve the predictor, and C1 would
    fail in a way indistinguishable from "augmentation does not help". The
    Wasserstein loss with gradient penalty does not vanish as the critic
    improves and is strongly resistant to collapse, at the cost of ~5x more
    critic steps per generator step.

  * Conditioning. The generator receives a one-hot STAGE label alongside
    the noise, so it can be asked specifically for near-failure windows.
    An unconditional generator would mostly reproduce the majority
    (healthy) class — the opposite of what is needed.

  * Hybrid Conv-BiLSTM. 1D convolutions capture local shock/amplitude
    patterns across the 20 features; the BiLSTM captures the temporal
    ordering within the 32-step window. ~0.9M parameters total — trivial
    for a T4.

  * Trained per fold on that fold's TRAINING windows only. The GAN never
    sees calibration or test data, so synthetic samples cannot leak into
    the conformal calibration and cannot break the coverage guarantee.

Usage:
    from gan import StageGAN, GANConfig
    cfg = GANConfig(augment_ratio=1.0)          # ablation knob
    gan = StageGAN(cfg)
    gan.fit(X_train, y_stage_train)             # real windows + stage labels
    X_aug, y_stage_aug = gan.augment(X_train, y_stage_train, target_stage=3)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Configuration ────────────────────────────────────────────────────

@dataclass
class GANConfig:
    # data shape
    window_size: int = 32
    n_features: int = 20
    n_stages: int = 3

    # architecture
    z_dim: int = 64
    hidden: int = 128
    conv_channels: int = 128

    # training
    epochs: int = 300
    batch_size: int = 64
    lr: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.9
    n_critic: int = 5          # critic steps per generator step (WGAN-GP)
    gp_lambda: float = 10.0    # gradient-penalty weight

    # augmentation (ABLATION KNOB)
    #   augment_ratio = (synthetic S3 windows) / (real S3 windows)
    #   0.0 = no augmentation, 1.0 = double the S3 count, etc.
    augment_ratio: float = 1.0
    target_stage: int = 3      # which stage to synthesise (3 = near-failure)

    seed: int = 42
    device: str = 'cuda'       # falls back to cpu automatically


# ── Generator: (noise + stage) -> synthetic window ───────────────────

class Generator(nn.Module):
    """noise z + one-hot stage -> [window_size, n_features]."""

    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        T, C = cfg.window_size, cfg.conv_channels

        # Project (z, stage) into a seed sequence
        self.fc = nn.Linear(cfg.z_dim + cfg.n_stages, T * 64)

        # Local pattern extraction across the sequence
        self.conv = nn.Sequential(
            nn.Conv1d(64, C, kernel_size=5, padding=2),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C, C, kernel_size=5, padding=2),
            nn.BatchNorm1d(C), nn.LeakyReLU(0.2, inplace=True),
        )

        # Temporal modelling
        self.lstm = nn.LSTM(C, cfg.hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * cfg.hidden, cfg.n_features)

    def forward(self, z, stage_onehot):
        cfg = self.cfg
        x = self.fc(torch.cat([z, stage_onehot], dim=1))
        x = x.view(-1, 64, cfg.window_size)      # (B, 64, T)
        x = self.conv(x)                         # (B, C, T)
        x = x.transpose(1, 2)                    # (B, T, C)
        x, _ = self.lstm(x)                      # (B, T, 2H)
        return self.out(x)                       # (B, T, n_features)


# ── Critic (discriminator): (window + stage) -> real-ness score ──────

class Critic(nn.Module):
    """[window_size, n_features] + one-hot stage -> scalar score.

    Note: no sigmoid. WGAN critics output an unbounded real number (an
    estimate of the Wasserstein distance), not a probability. Also no
    BatchNorm — it breaks the gradient penalty's per-sample assumption;
    LayerNorm is used instead.
    """

    def __init__(self, cfg: GANConfig):
        super().__init__()
        self.cfg = cfg
        C = cfg.conv_channels
        in_ch = cfg.n_features + cfg.n_stages

        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, C, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(C, C, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.lstm = nn.LSTM(C, cfg.hidden, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(2 * cfg.hidden)
        self.out = nn.Linear(2 * cfg.hidden, 1)

    def forward(self, window, stage_onehot):
        # Broadcast the stage label across every time step and append it
        c = stage_onehot.unsqueeze(1).expand(-1, window.size(1), -1)
        x = torch.cat([window, c], dim=2)        # (B, T, feat+stage)
        x = x.transpose(1, 2)                    # (B, feat+stage, T)
        x = self.conv(x).transpose(1, 2)        # (B, T, C)
        x, _ = self.lstm(x)                      # (B, T, 2H)
        x = self.norm(x[:, -1])                  # last step
        return self.out(x)                       # (B, 1)


# ── Main pipeline ────────────────────────────────────────────────────

class StageGAN:
    """Stage-conditioned WGAN-GP: fit on real windows, sample synthetic ones."""

    def __init__(self, cfg: Optional[GANConfig] = None):
        self.cfg = cfg or GANConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else 'cpu')
        if self.cfg.device == 'cuda' and self.device.type == 'cpu':
            print("[StageGAN] CUDA not available — training on CPU (slow).")

        self.G = Generator(self.cfg).to(self.device)
        self.D = Critic(self.cfg).to(self.device)
        self.history = {'d_loss': [], 'g_loss': [], 'wasserstein': []}
        print(f"[StageGAN] Generator {self._count(self.G):,} params | "
              f"Critic {self._count(self.D):,} params | device={self.device}")

    @staticmethod
    def _count(m):
        return sum(p.numel() for p in m.parameters())

    def _onehot(self, stages):
        """0-indexed stage ints -> one-hot float tensor."""
        s = torch.as_tensor(stages, dtype=torch.long, device=self.device)
        return F.one_hot(s, self.cfg.n_stages).float()

    def _gradient_penalty(self, real, stage_oh):
        """Enforce the 1-Lipschitz constraint on the critic (WGAN-GP core).

        Interpolates between real and fake samples and penalises any
        deviation of the critic's gradient norm from 1. This is what keeps
        the Wasserstein estimate valid and the training stable.
        """
        b = real.size(0)
        eps = torch.rand(b, 1, 1, device=self.device)
        z = torch.randn(b, self.cfg.z_dim, device=self.device)
        fake = self.G(z, stage_oh).detach()
        inter = (eps * real + (1 - eps) * fake).requires_grad_(True)

        score = self.D(inter, stage_oh)
        grad = torch.autograd.grad(
            outputs=score, inputs=inter,
            grad_outputs=torch.ones_like(score),
            create_graph=True, retain_graph=True)[0]
        grad = grad.view(b, -1)
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X_train, y_stage_train, verbose=True):
        """Train the GAN on one fold's real training windows.

        Args:
            X_train: (N, window_size, n_features) — scaled training windows
            y_stage_train: (N,) — 0-indexed stage labels {0,1,2}
        """
        cfg = self.cfg
        X = torch.as_tensor(X_train, dtype=torch.float32)
        y = torch.as_tensor(y_stage_train, dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, y)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

        optG = torch.optim.Adam(self.G.parameters(), lr=cfg.lr,
                                betas=(cfg.beta1, cfg.beta2))
        optD = torch.optim.Adam(self.D.parameters(), lr=cfg.lr,
                                betas=(cfg.beta1, cfg.beta2))

        self.G.train(); self.D.train()
        crit_step = 0
        for epoch in range(cfg.epochs):
            ep_d, ep_g, ep_w, n_g = 0.0, 0.0, 0.0, 0

            for real, stage in dl:
                real = real.to(self.device)
                stage_oh = self._onehot(stage)

                # ---- Critic update (every batch) ----
                optD.zero_grad()
                z = torch.randn(real.size(0), cfg.z_dim, device=self.device)
                fake = self.G(z, stage_oh).detach()
                d_real = self.D(real, stage_oh).mean()
                d_fake = self.D(fake, stage_oh).mean()
                gp = self._gradient_penalty(real, stage_oh)
                # Critic maximises (d_real - d_fake); minimise the negative, plus GP
                d_loss = -(d_real - d_fake) + cfg.gp_lambda * gp
                d_loss.backward()
                optD.step()

                ep_d += d_loss.item()
                ep_w += (d_real - d_fake).item()
                crit_step += 1

                # ---- Generator update: once per n_critic critic steps ----
                if crit_step % cfg.n_critic == 0:
                    optG.zero_grad()
                    z = torch.randn(real.size(0), cfg.z_dim, device=self.device)
                    gen = self.G(z, stage_oh)
                    g_loss = -self.D(gen, stage_oh).mean()
                    g_loss.backward()
                    optG.step()
                    ep_g += g_loss.item(); n_g += 1

            n_batches = len(dl)
            self.history['d_loss'].append(ep_d / n_batches)
            self.history['g_loss'].append(ep_g / max(n_g, 1))
            self.history['wasserstein'].append(ep_w / n_batches)

            if verbose and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
                print(f"  epoch {epoch:4d} | D {ep_d/n_batches:8.3f} | "
                      f"G {ep_g/max(n_g,1):8.3f} | "
                      f"W-dist {ep_w/n_batches:8.4f}")

        return self

    @torch.no_grad()
    def sample(self, n, stage):
        """Generate `n` synthetic windows of a given (0-indexed) stage.

        Returns:
            (n, window_size, n_features) numpy array
        """
        self.G.eval()
        z = torch.randn(n, self.cfg.z_dim, device=self.device)
        stage_oh = self._onehot(np.full(n, stage))
        return self.G(z, stage_oh).cpu().numpy()

    def augment(self, X_train, y_stage_train, target_stage=None,
                augment_ratio=None):
        """Return training set augmented with synthetic near-failure windows.

        The number generated is augment_ratio * (real windows of target_stage),
        so augment_ratio is the ablation knob: 0 disables augmentation, 1
        doubles the target-stage count, etc.

        Args:
            X_train, y_stage_train: real training data
            target_stage: 0-indexed stage to synthesise (default cfg.target_stage-1)
            augment_ratio: override cfg.augment_ratio

        Returns:
            X_aug: (N + M, window_size, n_features)
            y_aug: (N + M,) stage labels, synthetic ones flagged with the
                   same stage label as the real target-stage windows
        """
        cfg = self.cfg
        ratio = cfg.augment_ratio if augment_ratio is None else augment_ratio
        # cfg.target_stage is 1-indexed (3 = near-failure); labels are 0-indexed
        tgt = (cfg.target_stage - 1) if target_stage is None else target_stage

        n_real_tgt = int((y_stage_train == tgt).sum())
        n_syn = int(round(ratio * n_real_tgt))

        if n_syn == 0:
            return np.asarray(X_train), np.asarray(y_stage_train)

        X_syn = self.sample(n_syn, tgt)
        y_syn = np.full(n_syn, tgt, dtype=np.int64)

        X_aug = np.concatenate([np.asarray(X_train), X_syn], axis=0)
        y_aug = np.concatenate([np.asarray(y_stage_train), y_syn], axis=0)

        # Shuffle so synthetic windows are not all at the end
        perm = np.random.permutation(len(X_aug))
        return X_aug[perm], y_aug[perm]
