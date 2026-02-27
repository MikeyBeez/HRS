"""Loss functions for HRS training.

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
    This shapes geometry so nearby tokens cluster in representation space.
    """

    def __init__(self, cfg: LocalityConfig):
        super().__init__()
        self.window_size = cfg.window_size      # W
        self.neg_distance = cfg.neg_distance    # 2W
        self.temperature = cfg.temperature
        self.n_negatives = cfg.n_negatives

    def forward(
        self, hidden_states: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """Compute InfoNCE locality loss on hidden states.

        Args:
            hidden_states: (B, T, D) hidden states from a specific layer
            layer_idx: which layer (unused, for logging)

        Returns:
            Scalar InfoNCE loss
        """
        B, T, D = hidden_states.shape
        W = self.window_size

        if T < 2 * self.neg_distance:
            return torch.tensor(0.0, device=hidden_states.device)

        # Subsample anchor positions for efficiency
        n_anchors = min(64, T - 2 * W)
        anchor_idx = torch.randint(W, T - W, (n_anchors,), device=hidden_states.device)

        # Get anchor representations: (B, n_anchors, D)
        anchors = hidden_states[:, anchor_idx]

        # Positive samples: random token within window of each anchor
        pos_offsets = torch.randint(-W // 2, W // 2 + 1, (n_anchors,), device=hidden_states.device)
        pos_idx = (anchor_idx + pos_offsets).clamp(0, T - 1)
        positives = hidden_states[:, pos_idx]  # (B, n_anchors, D)

        # Negative samples: random tokens (most will be far enough)
        # For efficiency, just sample random positions — in a 512-token
        # sequence with W=16, most random pairs are naturally >2W apart
        n_neg = self.n_negatives
        neg_idx = torch.randint(0, T, (n_neg,), device=hidden_states.device)
        negatives = hidden_states[:, neg_idx]  # (B, n_neg, D)

        # Normalize for cosine similarity
        anchors_n = F.normalize(anchors, dim=-1)       # (B, n_anchors, D)
        positives_n = F.normalize(positives, dim=-1)    # (B, n_anchors, D)
        negatives_n = F.normalize(negatives, dim=-1)    # (B, n_neg, D)

        # Positive similarity: (B, n_anchors)
        pos_sim = (anchors_n * positives_n).sum(dim=-1) / self.temperature

        # Negative similarities: (B, n_anchors, n_neg)
        # einsum: batch, anchor, dim x batch, neg, dim -> batch, anchor, neg
        neg_sim = torch.einsum('bad,bnd->ban', anchors_n, negatives_n) / self.temperature

        # InfoNCE: classify positive as index 0 among [pos, neg1, neg2, ...]
        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (B, n_anchors, 1+n_neg)
        labels = torch.zeros(B * n_anchors, dtype=torch.long, device=hidden_states.device)
        loss = F.cross_entropy(logits.reshape(B * n_anchors, -1), labels)

        return loss


class CombinedHRSLoss(nn.Module):
    """Combined loss for HRS training.

    L_total = L_CE + locality_weight * L_locality + balance_weight * L_balance + recon_weight * L_recon
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
        """Compute combined HRS loss.

        Returns dict with individual loss components for logging.
        """
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

        # Routing balance loss
        if routing_balance_loss_val is not None:
            total = total + balance_weight * routing_balance_loss_val
            result["balance_loss"] = routing_balance_loss_val.detach()

        # Routing entropy regularization (encourages exploration)
        if routing_entropy_loss_val is not None and entropy_weight > 0:
            total = total + entropy_weight * routing_entropy_loss_val
            result["entropy_loss"] = routing_entropy_loss_val.detach()

        # Routing FLOPs cost loss
        if routing_flops_loss_val is not None and flops_weight > 0:
            total = total + flops_weight * routing_flops_loss_val
            result["flops_loss"] = routing_flops_loss_val.detach()

        # Engram reconstruction loss
        if engram_recon_loss is not None:
            total = total + recon_weight * engram_recon_loss
            result["recon_loss"] = engram_recon_loss.detach()

        result["loss"] = total
        return result
