"""训练器 — 标准化 Huber Loss + Early Stopping.

论文 Section 7.7:
  L_resp = Σ_m λ_m · Huber(r_hat_{t+h}^{(m)} - r_{t+h}^{(m)})
  所有响应变量先标准化.
"""

import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from src.training.metrics import compute_all_metrics, compute_per_variable_metrics

logger = logging.getLogger(__name__)

TARGET_NAMES = ["AdvanceRate", "Torque", "Thrust", "Penetration", "ShieldPressure"]


class StandardizedHuberLoss(nn.Module):
    """多任务标准化 Huber Loss.

    L_resp = Σ_m λ_m · Huber(ŷ_m - y_m)
    """

    def __init__(self, n_targets: int = 5, delta: float = 1.0,
                 weights: Optional[torch.Tensor] = None):
        super().__init__()
        if weights is None:
            weights = torch.ones(n_targets)
        self.register_buffer("weights", weights)
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        quad = torch.where(abs_diff <= self.delta,
                           0.5 * diff ** 2,
                           self.delta * (abs_diff - 0.5 * self.delta))
        weighted = quad * self.weights.unsqueeze(0)
        return weighted.sum(dim=1).mean()


# ── Trainer ───────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 30, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def train_epoch(model: nn.Module, dataloader: DataLoader,
                loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                device: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        x_seq, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        pred = model(x_seq)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train_graph_sequence_epoch(model: nn.Module, dataloader: DataLoader,
                               loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                               device: str, tau: float = 2.0) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        graph_seqs, monitoring_seq, y = batch[:3]
        monitoring_seq = monitoring_seq.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred, _ = model(graph_seqs, monitoring_seq, tau=tau)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model: nn.Module, dataloader: DataLoader,
               loss_fn: nn.Module, device: str) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in dataloader:
        x_seq, y = batch[0].to(device), batch[1].to(device)
        pred = model(x_seq)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(targets, preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def eval_graph_sequence_epoch(model: nn.Module, dataloader: DataLoader,
                              loss_fn: nn.Module, device: str,
                              tau: float = 2.0) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in dataloader:
        graph_seqs, monitoring_seq, y = batch[:3]
        monitoring_seq = monitoring_seq.to(device)
        y = y.to(device)

        pred, _ = model(graph_seqs, monitoring_seq, tau=tau)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(targets, preds)
    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def train_sequence_model(model: nn.Module,
                         train_loader: DataLoader,
                         val_loader: DataLoader,
                         epochs: int = 200,
                         lr: float = 0.001,
                         weight_decay: float = 1e-4,
                         patience: int = 30,
                         huber_delta: float = 1.0,
                         device: str = "cpu",
                         checkpoint_dir: Optional[Path] = None):
    """标准训练循环."""
    loss_fn = StandardizedHuberLoss(n_targets=5, delta=huber_delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    early_stop = EarlyStopping(patience=patience)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_results = eval_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_results["loss"])
        improved = early_stop.update(val_results["loss"])

        if improved:
            best_val_loss = val_results["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(best_state, checkpoint_dir / "best_model.pt")

        if epoch % 20 == 0 or early_stop.should_stop:
            logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                        f"val_loss={val_results['loss']:.4f} | "
                        f"val_mae={val_results['mae']:.4f}")

        if early_stop.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


# ── 静态图模型训练 (只用 G_t, 不用序列) ────────────────────────────

def train_static_graph_epoch(model: nn.Module, dataloader: DataLoader,
                             loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                             device: str, tau: float = 2.0) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        graph_seqs, monitoring_seq, y = batch[:3]
        y = y.to(device)
        # 遍历 batch 中每个样本, 取其序列的最后一个图快照
        batch_preds = []
        for sample_seqs in graph_seqs:
            last_snap = sample_seqs[-1]  # 每个样本序列的最后一步快照
            rock_attrs = last_snap.rock_attrs.to(device)
            tbm_comp = last_snap.tbm_components.to(device)
            pred = model(last_snap.hetero_data, rock_attrs, tbm_comp, tau=tau)
            batch_preds.append(pred.unsqueeze(0))  # (1, output_dim)
        preds = torch.cat(batch_preds, dim=0)  # (batch, output_dim)
        loss = loss_fn(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def eval_static_graph_epoch(model: nn.Module, dataloader: DataLoader,
                            loss_fn: nn.Module, device: str,
                            tau: float = 2.0) -> dict:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_targets = [], []
    for batch in dataloader:
        graph_seqs, monitoring_seq, y = batch[:3]
        y = y.to(device)
        # 遍历 batch 中每个样本, 取其序列的最后一个图快照
        batch_preds = []
        for sample_seqs in graph_seqs:
            last_snap = sample_seqs[-1]
            rock_attrs = last_snap.rock_attrs.to(device)
            tbm_comp = last_snap.tbm_components.to(device)
            pred = model(last_snap.hetero_data, rock_attrs, tbm_comp, tau=tau)
            batch_preds.append(pred.unsqueeze(0))  # (1, output_dim)
        preds = torch.cat(batch_preds, dim=0)  # (batch, output_dim)
        loss = loss_fn(preds, y)
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(targets, preds)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def train_static_graph_model(model: nn.Module,
                             train_loader: DataLoader,
                             val_loader: DataLoader,
                             epochs: int = 200,
                             lr: float = 0.001,
                             weight_decay: float = 1e-4,
                             patience: int = 30,
                             huber_delta: float = 1.0,
                             tau: float = 2.0,
                             device: str = "cpu",
                             checkpoint_dir: Optional[Path] = None,
                             checkpoint_name: str = "best_static_graph.pt"):
    """Training loop for static graph models (single G_t, no sequence)."""
    loss_fn = StandardizedHuberLoss(n_targets=5, delta=huber_delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    early_stop = EarlyStopping(patience=patience)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_static_graph_epoch(
            model, train_loader, loss_fn, optimizer, device, tau=tau
        )
        val_results = eval_static_graph_epoch(
            model, val_loader, loss_fn, device, tau=tau
        )
        scheduler.step(val_results["loss"])
        improved = early_stop.update(val_results["loss"])

        if improved:
            best_val_loss = val_results["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(best_state, checkpoint_dir / checkpoint_name)

        if epoch % 20 == 0 or early_stop.should_stop:
            logger.info(f"[Static] Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                        f"val_loss={val_results['loss']:.4f} | "
                        f"val_mae={val_results['mae']:.4f}")

        if early_stop.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss


def train_graph_sequence_model(model: nn.Module,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               epochs: int = 200,
                               lr: float = 0.001,
                               weight_decay: float = 1e-4,
                               patience: int = 30,
                               huber_delta: float = 1.0,
                               tau: float = 2.0,
                               device: str = "cpu",
                               checkpoint_dir: Optional[Path] = None,
                               checkpoint_name: str = "best_graph_model.pt"):
    """Training loop for graph-sequence models."""
    loss_fn = StandardizedHuberLoss(n_targets=5, delta=huber_delta)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    early_stop = EarlyStopping(patience=patience)

    best_state = None
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_graph_sequence_epoch(
            model, train_loader, loss_fn, optimizer, device, tau=tau
        )
        val_results = eval_graph_sequence_epoch(
            model, val_loader, loss_fn, device, tau=tau
        )
        scheduler.step(val_results["loss"])
        improved = early_stop.update(val_results["loss"])

        if improved:
            best_val_loss = val_results["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_dir:
                torch.save(best_state, checkpoint_dir / checkpoint_name)

        if epoch % 20 == 0 or early_stop.should_stop:
            logger.info(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                        f"val_loss={val_results['loss']:.4f} | "
                        f"val_mae={val_results['mae']:.4f}")

        if early_stop.should_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_loss
