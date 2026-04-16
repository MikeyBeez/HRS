"""Phase D: MetadataEmbedder — converts tagger output to dense vectors.

Produces a per-token metadata embedding the same dimension as the main
model's hidden states (1024). These are added to the token embeddings
before the first transformer block sees them.

The tagger is frozen; only the MetadataEmbedder parameters are trained
alongside the main model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from experiments.identity_ae.preprocessing.train_tagger import (
    PromptTagger, CONTENT_TYPES, POS_TAGS, ENTITY_TYPES, GRAM_ROLES,
    QUERY_TYPES,
)


class MetadataEmbedder(nn.Module):
    """Convert per-token metadata predictions to dense embeddings."""

    def __init__(self, main_dim=1024, meta_dim=128):
        super().__init__()
        self.ct_embed = nn.Embedding(len(CONTENT_TYPES), meta_dim)
        self.pos_embed = nn.Embedding(len(POS_TAGS), meta_dim)
        self.et_embed = nn.Embedding(len(ENTITY_TYPES), meta_dim)
        self.gr_embed = nn.Embedding(len(GRAM_ROLES), meta_dim)
        self.sal_proj = nn.Linear(1, meta_dim)
        self.qt_embed = nn.Embedding(len(QUERY_TYPES), meta_dim)

        self.proj = nn.Linear(6 * meta_dim, main_dim)
        # Learnable mixing scalar, initialized small so metadata is a
        # perturbation at the start of training.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, ct, pos, et, gr, sal, qt):
        """
        Args:
            ct:  (B, T) int — content_type predictions
            pos: (B, T) int — POS predictions
            et:  (B, T) int — entity_type predictions
            gr:  (B, T) int — grammatical role predictions
            sal: (B, T) float — salience predictions
            qt:  (B,) int — query_type prediction (broadcast to all tokens)
        Returns:
            meta_embed: (B, T, main_dim)
        """
        ct_e = self.ct_embed(ct)                           # (B, T, meta_dim)
        pos_e = self.pos_embed(pos)
        et_e = self.et_embed(et)
        gr_e = self.gr_embed(gr)
        sal_e = self.sal_proj(sal.unsqueeze(-1))
        qt_e = self.qt_embed(qt).unsqueeze(1).expand(
            -1, ct.size(1), -1)                            # (B, T, meta_dim)

        cat = torch.cat([ct_e, pos_e, et_e, gr_e, sal_e, qt_e], dim=-1)
        return self.alpha * self.proj(cat)


class TaggerPipeline(nn.Module):
    """Frozen tagger + trainable MetadataEmbedder in a single module.

    Input: BPE token IDs (B, T)
    Output: metadata embeddings (B, T, main_dim) to be added to token
            embeddings before the main model's first block.
    """

    def __init__(self, tagger_path, main_dim=1024, meta_dim=128, device="cpu"):
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tagger = PromptTagger(tokenizer.vocab_size, embed_dim=256,
                                   hidden_dim=512)
        ckpt = torch.load(tagger_path, map_location=device, weights_only=False)
        self.tagger.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Freeze tagger
        for p in self.tagger.parameters():
            p.requires_grad = False
        self.tagger.eval()

        self.embedder = MetadataEmbedder(main_dim=main_dim,
                                          meta_dim=meta_dim)

    @torch.no_grad()
    def _tag(self, ids):
        """Run the frozen tagger on CPU and return hard predictions on
        the same device as the embedder."""
        ids_cpu = ids.cpu()
        ct_l, pos_l, et_l, gr_l, sal_p, qt_l = self.tagger(ids_cpu)
        target_device = next(self.embedder.parameters()).device
        ct = ct_l.argmax(-1).to(target_device)
        pos = pos_l.argmax(-1).to(target_device)
        et = et_l.argmax(-1).to(target_device)
        gr = gr_l.argmax(-1).to(target_device)
        sal = sal_p.to(target_device)
        qt = qt_l.argmax(-1).to(target_device)
        return ct, pos, et, gr, sal, qt

    def forward(self, ids):
        """
        Args:
            ids: (B, T) int — GPT-2 BPE token IDs
        Returns:
            meta_embed: (B, T, D) on the embedder's device
        """
        ct, pos, et, gr, sal, qt = self._tag(ids)
        return self.embedder(ct, pos, et, gr, sal, qt)
