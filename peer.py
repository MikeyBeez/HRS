"""PEER: Parameter Efficient Expert Retrieval.

Product-key retrieval over ~262K single-neuron experts.
Replaces standard MLP/FFN in all transformer layers.

Architecture:
- 2 sub-key sets x 512 keys = 262,144 addressable experts
- 8 heads x 16 experts/head = 128 active experts per token
- Each expert: sigma(u_i^T x) * v_i (single neuron)
- O(sqrt(N)) retrieval via product keys

Reference: Lample et al., "Large Memory Layers with Product Keys" (2019)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import PEERConfig, ModelConfig


class PEER(nn.Module):
    """Parameter Efficient Expert Retrieval FFN.

    Uses product keys to index into a large table of single-neuron experts.
    Each head independently retrieves top-k experts, and outputs are summed.
    """

    def __init__(self, model_cfg: ModelConfig, peer_cfg: PEERConfig):
        super().__init__()
        d = model_cfg.d_model
        self.n_heads = peer_cfg.n_heads
        self.top_k = peer_cfg.top_k
        self.n_sub_keys = peer_cfg.n_sub_keys
        self.head_dim = d // peer_cfg.n_heads

        n_total = peer_cfg.n_sub_keys ** 2  # 512^2 = 262144

        # Input projection: split input into heads
        self.input_proj = nn.Linear(d, d, bias=False)

        # Product keys: two sets of sub-keys per head
        # Each head has its own key tables
        self.keys_a = nn.Parameter(torch.empty(self.n_heads, peer_cfg.n_sub_keys, self.head_dim))
        self.keys_b = nn.Parameter(torch.empty(self.n_heads, peer_cfg.n_sub_keys, self.head_dim))

        # Expert parameters: single-neuron experts (u_i, v_i)
        # u_i: input weight (head_dim,), v_i: output weight (head_dim,)
        # Shared across heads to save memory, indexed by product key
        self.expert_u = nn.Parameter(torch.empty(n_total, self.head_dim))
        self.expert_v = nn.Parameter(torch.empty(n_total, self.head_dim))

        # Output projection: merge heads back to d_model
        self.output_proj = nn.Linear(d, d, bias=False)

        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(model_cfg.dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.input_proj.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.normal_(self.keys_a, std=0.02)
        nn.init.normal_(self.keys_b, std=0.02)
        # Expert weights: small init for stability
        nn.init.normal_(self.expert_u, std=0.01)
        nn.init.normal_(self.expert_v, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PEER forward pass with product-key retrieval.

        Args:
            x: (B, T, d_model)
        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape
        K = self.top_k
        H = self.n_heads
        S = self.n_sub_keys
        hd = self.head_dim

        # Project and split into heads: (B, T, H, hd)
        h = self.input_proj(x).reshape(B, T, H, hd)

        # Product key retrieval per head
        # Score against sub-key set A: (B, T, H, S)
        scores_a = torch.einsum('bthd,hsd->bths', h, self.keys_a)
        # Score against sub-key set B: (B, T, H, S)
        scores_b = torch.einsum('bthd,hsd->bths', h, self.keys_b)

        # Top-k from each sub-key set
        top_a_scores, top_a_idx = scores_a.topk(K, dim=-1)  # (B, T, H, K)
        top_b_scores, top_b_idx = scores_b.topk(K, dim=-1)  # (B, T, H, K)

        # Product keys: combine top-k from A and B
        # Product scores: outer product of top scores -> (B, T, H, K, K)
        product_scores = top_a_scores.unsqueeze(-1) + top_b_scores.unsqueeze(-2)
        # Product indices: a_idx * S + b_idx -> (B, T, H, K, K)
        product_idx = top_a_idx.unsqueeze(-1) * S + top_b_idx.unsqueeze(-2)

        # Flatten to K^2 candidates, take top-K
        product_scores_flat = product_scores.reshape(B, T, H, K * K)
        product_idx_flat = product_idx.reshape(B, T, H, K * K)

        top_scores, top_pos = product_scores_flat.topk(K, dim=-1)  # (B, T, H, K)
        top_expert_idx = product_idx_flat.gather(-1, top_pos)  # (B, T, H, K)

        # Softmax over selected expert scores for weighted combination
        top_weights = F.softmax(top_scores, dim=-1)  # (B, T, H, K)

        # Gather expert parameters for selected experts
        # expert_u/expert_v: (N_total, hd)
        # top_expert_idx: (B, T, H, K) -> flatten to gather
        flat_idx = top_expert_idx.reshape(-1)  # (B*T*H*K,)
        sel_u = self.expert_u[flat_idx].reshape(B, T, H, K, hd)  # (B, T, H, K, hd)
        sel_v = self.expert_v[flat_idx].reshape(B, T, H, K, hd)  # (B, T, H, K, hd)

        # Single-neuron expert computation: sigma(u_i^T x) * v_i
        # h: (B, T, H, hd) -> (B, T, H, 1, hd)
        h_expanded = h.unsqueeze(3)
        # Activation: sigmoid(u_i^T x) for each expert
        activations = torch.sigmoid((h_expanded * sel_u).sum(dim=-1))  # (B, T, H, K)
        # Weighted output: sum_i w_i * activation_i * v_i
        weighted_acts = (top_weights * activations).unsqueeze(-1)  # (B, T, H, K, 1)
        expert_out = (weighted_acts * sel_v).sum(dim=3)  # (B, T, H, hd)

        # Merge heads: (B, T, D)
        merged = expert_out.reshape(B, T, D)
        out = self.dropout(self.norm(self.output_proj(merged)))

        return out

    def expert_utilization(self) -> dict:
        """Return statistics about expert usage (call after forward pass).

        This is a placeholder — actual tracking requires storing indices
        from the last forward pass. See metrics.py for evaluation-time tracking.
        """
        n_total = self.n_sub_keys ** 2
        return {
            "n_total_experts": n_total,
            "n_active_per_token": self.n_heads * self.top_k,
            "sparsity": 1.0 - (self.n_heads * self.top_k) / n_total,
        }
