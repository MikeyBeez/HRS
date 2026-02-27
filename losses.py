"""Loss functions for HRS training (v1 and v2).

Includes:
- CrossEntropyLoss: standard next-token prediction
- LocalityLoss: InfoNCE contrastive objective on local windows
- Combined HRS loss integrating all objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LocalityConfig


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss for next-token prediction with optional label smoothing."""

    def __init__(self, ignore_index: int = -100, label_smoothing: float = 0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )


class LocalityLoss(nn.Module):
    """InfoNCE contrastive loss on local windows (Head B objective).

    Tokens within a window of size W are positives (should have similar
    representations). Tokens beyond 2W are negatives (should be dissimilar).
    """

    def __init__(self, cfg: LocalityConfig):
        super().__init__()
        self.window_size = cfg.window_size
        self.neg_distance = cfg.neg_distance
        self.temperature = cfg.temperature
        self.n_negatives = cfg.n_negatives

    def forward(
        self, hidden_states: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        B, T, D = hidden_states.shape
        W = self.window_size

        if T < 2 * self.neg_distance:
            return torch.tensor(0.0, device=hidden_states.device)

        n_anchors = min(64, T - 2 * W)
        anchor_idx = torch.randint(W, T - W, (n_anchors,), device=hidden_states.device)

        anchors = hidden_states[:, anchor_idx]

        pos_offsets = torch.randint(-W // 2, W // 2 + 1, (n_anchors,), device=hidden_states.device)
        pos_idx = (anchor_idx + pos_offsets).clamp(0, T - 1)
        positives = hidden_states[:, pos_idx]

        n_neg = self.n_negatives
        neg_idx = torch.randint(0, T, (n_neg,), device=hidden_states.device)
        negatives = hidden_states[:, neg_idx]

        anchors_n = F.normalize(anchors, dim=-1)
        positives_n = F.normalize(positives, dim=-1)
        negatives_n = F.normalize(negatives, dim=-1)

        pos_sim = (anchors_n * positives_n).sum(dim=-1) / self.temperature
        neg_sim = torch.einsum('bad,bnd->ban', anchors_n, negatives_n) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(B * n_anchors, dtype=torch.long, device=hidden_states.device)
        loss = F.cross_entropy(logits.reshape(B * n_anchors, -1), labels)

        return loss


class SelfAdaptiveLossWeight(nn.Module):
    """Self-adaptive loss weighting via sigmoid-activated learned scalars.

    Each auxiliary loss gets a learned scalar that passes through sigmoid
    to produce a weight in [0, 1]. This lets the model learn the relative
    importance of each loss term during training.
    """

    def __init__(self, n_losses: int, init_value: float = 0.0):
        super().__init__()
        self.raw_weights = nn.Parameter(torch.full((n_losses,), init_value))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_weights)


class CombinedHRSLoss(nn.Module):
    """Combined loss for HRS training (v1 and v2).

    v1: L_CE + L_locality + L_balance + L_entropy + L_flops + L_recon
    v2: L_CE + L_locality + L_recon (no routing losses)
    """

    def __init__(self, locality_cfg: LocalityConfig = None, label_smoothing: float = 0.0):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(label_smoothing=label_smoothing)
        self.locality_loss = LocalityLoss(locality_cfg) if locality_cfg and locality_cfg.enabled else None
        self.locality_weight = locality_cfg.loss_weight if locality_cfg and locality_cfg.enabled else 0.0

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        layer_representations: list = None,
        routing_weights: list = None,
        routing_balance_loss_val: torch.Tensor = None,
        balance_weight: float = 0.01,
        routing_entropy_loss_val: torch.Tensor = None,
        entropy_weight: float = 0.0,
        routing_flops_loss_val: torch.Tensor = None,
        flops_weight: float = 0.0,
        engram_recon_loss: torch.Tensor = None,
        recon_weight: float = 0.1,
    ) -> dict:
        ce = self.ce_loss(logits, targets)

        total = ce
        result = {"loss": ce, "ce_loss": ce.detach()}

        # Locality loss (averaged across selected layers)
        if self.locality_loss is not None and layer_representations:
            loc_total = torch.tensor(0.0, device=ce.device)
            for i, rep in enumerate(layer_representations):
                loc_total = loc_total + self.locality_loss(rep, layer_idx=i)
            loc_avg = loc_total / max(len(layer_representations), 1)
            total = total + self.locality_weight * loc_avg
            result["locality_loss"] = loc_avg.detach()

        # Routing balance loss (v1 only)
        if routing_balance_loss_val is not None:
            total = total + balance_weight * routing_balance_loss_val
            result["balance_loss"] = routing_balance_loss_val.detach()

        # Routing entropy regularization (v1 only)
        if routing_entropy_loss_val is not None and entropy_weight > 0:
            total = total + entropy_weight * routing_entropy_loss_val
            result["entropy_loss"] = routing_entropy_loss_val.detach()

        # Routing FLOPs cost loss (v1 only)
        if routing_flops_loss_val is not None and flops_weight > 0:
            total = total + flops_weight * routing_flops_loss_val
            result["flops_loss"] = routing_flops_loss_val.detach()

        # Engram reconstruction loss (v1 and v2)
        if engram_recon_loss is not None:
            total = total + recon_weight * engram_recon_loss
            result["recon_loss"] = engram_recon_loss.detach()

        result["loss"] = total
        return result
