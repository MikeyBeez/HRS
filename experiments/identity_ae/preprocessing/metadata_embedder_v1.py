"""V1 MetadataEmbedder (5 fields, no gram_role).

Used by Phase 49b training. Keep this file for loading 49b checkpoints.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer

CONTENT_TYPES_V1 = ["content", "template", "punctuation"]
POS_TAGS_V1 = ["noun", "verb", "adj", "adv", "det", "prep", "conj", "pron", "other"]
ENTITY_TYPES_V1 = ["none", "person", "location", "org", "artifact", "number", "technical_term"]
QUERY_TYPES_V1 = ["factual", "numeric", "entity", "technical", "procedural", "compositional", "declarative", "other"]


class MetadataEmbedderV1(nn.Module):
    def __init__(self, main_dim=1024, meta_dim=128):
        super().__init__()
        self.ct_embed = nn.Embedding(len(CONTENT_TYPES_V1), meta_dim)
        self.pos_embed = nn.Embedding(len(POS_TAGS_V1), meta_dim)
        self.et_embed = nn.Embedding(len(ENTITY_TYPES_V1), meta_dim)
        self.sal_proj = nn.Linear(1, meta_dim)
        self.qt_embed = nn.Embedding(len(QUERY_TYPES_V1), meta_dim)
        self.proj = nn.Linear(5 * meta_dim, main_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, ct, pos, et, sal, qt):
        ct_e = self.ct_embed(ct)
        pos_e = self.pos_embed(pos)
        et_e = self.et_embed(et)
        sal_e = self.sal_proj(sal.unsqueeze(-1))
        qt_e = self.qt_embed(qt).unsqueeze(1).expand(-1, ct.size(1), -1)
        cat = torch.cat([ct_e, pos_e, et_e, sal_e, qt_e], dim=-1)
        return self.alpha * self.proj(cat)
