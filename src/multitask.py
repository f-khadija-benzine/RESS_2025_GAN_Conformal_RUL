"""
multitask.py — Multi-task RUL + stage model (C2)

Extends the baseline by adding a stage-classification head to the shared
BiLSTM encoder. Both heads are trained jointly on all windows:

    L = L_RUL + lambda * L_stage

  * RUL head: regression, MSE, on all windows. The piecewise target makes
    healthy windows learnable (all carry RUL = 1.0), so no masking is needed.
  * Stage head: 3-class classification (healthy / early / near-failure),
    cross-entropy, on all windows. Healthy windows, which carry little RUL
    signal, still inform the shared encoder through this head.

The hypothesis is that forcing the encoder to also separate the degradation
stages yields a representation that improves the RUL head, particularly in
the near-failure regime. lambda balances the two tasks and is swept.

This module reuses BiLSTMEncoder, ModelConfig, and the metric functions from
model.py; it only adds the two-head model and its trainer.
"""

import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn

from model import ModelConfig, BiLSTMEncoder


@dataclass
class MTConfig(ModelConfig):
    """ModelConfig plus multi-task settings."""
    lambda_stage: float = 0.5     # weight on the stage-classification loss
    n_stages: int = 3
    stage_hidden: int = 48


class MultiTaskModel(nn.Module):
    """Shared encoder + RUL regression head + stage classification head."""

    def __init__(self, cfg: MTConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = BiLSTMEncoder(cfg)
        enc_dim = 2 * cfg.hidden

        self.rul_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.rul_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.rul_hidden, 1),
        )
        self.stage_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.stage_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.stage_hidden, cfg.n_stages),
        )

    def forward(self, x):
        context, _ = self.encoder(x)
        rul = self.rul_head(context).squeeze(-1).clamp(0.0, 1.0)
        stage_logits = self.stage_head(context)
        return rul, stage_logits


class MultiTaskTrainer:
    """Trains the multi-task model for one fold with early stopping on val RUL."""

    def __init__(self, cfg: MTConfig = None):
        self.cfg = cfg or MTConfig()
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else 'cpu')
        self.model = MultiTaskModel(self.cfg).to(self.device)
        self.history = {'train_rul': [], 'val_rul': [], 'train_stage_acc': []}

    def _loaders(self, X, y_rul, y_stage):
        X = torch.as_tensor(X, dtype=torch.float32)
        yr = torch.as_tensor(y_rul, dtype=torch.float32)
        ys = torch.as_tensor(np.asarray(y_stage), dtype=torch.long)
        ds = torch.utils.data.TensorDataset(X, yr, ys)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=True)

    def fit(self, X_train, y_rul_train, y_stage_train,
            X_val, y_rul_val, verbose=True):
        cfg = self.cfg
        tr = self._loaders(X_train, y_rul_train, y_stage_train)
        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr,
                               weight_decay=cfg.weight_decay)
        mse = nn.MSELoss()
        ce = nn.CrossEntropyLoss()

        best_val = float('inf'); best_state = None; bad = 0
        for epoch in range(cfg.epochs):
            self.model.train()
            se, correct, n = 0.0, 0, 0
            for xb, yrb, ysb in tr:
                xb, yrb, ysb = xb.to(self.device), yrb.to(self.device), ysb.to(self.device)
                opt.zero_grad()
                rul, logits = self.model(xb)
                loss = mse(rul, yrb) + cfg.lambda_stage * ce(logits, ysb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                opt.step()
                se += ((rul - yrb) ** 2).sum().item()
                correct += (logits.argmax(1) == ysb).sum().item(); n += len(yrb)
            train_rul = (se / n) ** 0.5
            train_stage_acc = correct / n

            val_rul = self.evaluate_rul_rmse(X_val, y_rul_val)
            self.history['train_rul'].append(train_rul)
            self.history['val_rul'].append(val_rul)
            self.history['train_stage_acc'].append(train_stage_acc)

            if val_rul < best_val - 1e-5:
                best_val = val_rul
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if verbose and (epoch % 10 == 0 or epoch == cfg.epochs - 1):
                print(f"  epoch {epoch:3d} | train RUL {train_rul:.4f} "
                      f"stage-acc {train_stage_acc:.3f} | val RUL {val_rul:.4f} "
                      f"| best {best_val:.4f}")
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def predict(self, X):
        """Returns (rul_pred, stage_pred)."""
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32).to(self.device)
        ruls, stages = [], []
        for i in range(0, len(X), self.cfg.batch_size):
            r, s = self.model(X[i:i + self.cfg.batch_size])
            ruls.append(r.cpu().numpy())
            stages.append(s.argmax(1).cpu().numpy())
        return (np.concatenate(ruls) if ruls else np.empty(0),
                np.concatenate(stages) if stages else np.empty(0))

    @torch.no_grad()
    def predict_rul(self, X):
        return self.predict(X)[0]

    def evaluate_rul_rmse(self, X, y):
        if len(X) == 0:
            return float('nan')
        yp = self.predict_rul(X)
        return float(np.sqrt(np.mean((yp - np.asarray(y)) ** 2)))

    def stage_accuracy(self, X, y_stage):
        if len(X) == 0:
            return float('nan')
        _, sp = self.predict(X)
        return float(np.mean(sp == np.asarray(y_stage)))
