"""Engram Store: store and retrieve engram vectors with associated text.

Stores engram vectors (mean-pooled hidden states from V18's extraction layer)
as keys, with original prompt text as values. Supports cosine similarity
retrieval and persistence to disk.
"""

import torch
import torch.nn.functional as F
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EngramEntry:
    """Metadata for a single stored engram."""
    text: str                    # original text segment
    mean_entropy: float          # mean entropy at storage time
    condition: str               # 'full_context' or 'isolated'
    source: str = ""             # document/segment identifier


class EngramStore:
    """Store and retrieve engram vectors via cosine similarity.

    Keys are engram vectors (d_model,), values are text segments.
    Retrieval is brute-force cosine similarity — fast enough for <100K entries.
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.keys: Optional[torch.Tensor] = None   # (N, d_model)
        self.entries: List[EngramEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def store(self, engram: torch.Tensor, entry: EngramEntry):
        """Store an engram vector with its metadata.

        Args:
            engram: (d_model,) engram vector
            entry: associated metadata
        """
        engram = engram.detach().cpu().float()
        if engram.dim() == 2:
            engram = engram.squeeze(0)
        assert engram.shape == (self.d_model,), f"Expected ({self.d_model},), got {engram.shape}"

        # Normalize for cosine similarity
        engram = F.normalize(engram, dim=0)

        if self.keys is None:
            self.keys = engram.unsqueeze(0)
        else:
            self.keys = torch.cat([self.keys, engram.unsqueeze(0)], dim=0)
        self.entries.append(entry)

    def retrieve(
        self, query: torch.Tensor, top_k: int = 1, min_similarity: float = 0.0,
    ) -> List[Tuple[float, EngramEntry, torch.Tensor]]:
        """Retrieve nearest engrams by cosine similarity.

        Args:
            query: (d_model,) query engram vector
            top_k: number of results to return
            min_similarity: minimum cosine similarity threshold

        Returns:
            List of (similarity, entry, engram_vector) tuples, sorted by similarity descending.
            Returns empty list if store is empty or no match above threshold.
        """
        if self.keys is None or len(self.entries) == 0:
            return []

        query = query.detach().cpu().float()
        if query.dim() == 2:
            query = query.squeeze(0)
        query = F.normalize(query, dim=0)

        # Cosine similarity (keys are already normalized)
        similarities = self.keys @ query  # (N,)

        # Filter by minimum similarity
        mask = similarities >= min_similarity
        if not mask.any():
            return []

        # Top-K
        k = min(top_k, mask.sum().item())
        values, indices = similarities.topk(k)

        results = []
        for sim, idx in zip(values.tolist(), indices.tolist()):
            results.append((sim, self.entries[idx], self.keys[idx]))
        return results

    def save(self, path: str):
        """Save store to disk (keys tensor + entries JSON)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save keys tensor
        if self.keys is not None:
            torch.save(self.keys, path / "engram_keys.pt")

        # Save entries as JSON
        entries_data = []
        for e in self.entries:
            entries_data.append({
                "text": e.text,
                "mean_entropy": e.mean_entropy,
                "condition": e.condition,
                "source": e.source,
            })
        with open(path / "engram_entries.json", "w") as f:
            json.dump({
                "d_model": self.d_model,
                "n_entries": len(self.entries),
                "entries": entries_data,
            }, f, indent=2)

        print(f"Saved {len(self.entries)} engrams to {path}")

    @classmethod
    def load(cls, path: str) -> "EngramStore":
        """Load store from disk."""
        path = Path(path)

        with open(path / "engram_entries.json") as f:
            data = json.load(f)

        store = cls(d_model=data["d_model"])

        if (path / "engram_keys.pt").exists():
            store.keys = torch.load(path / "engram_keys.pt", weights_only=True)

        for e in data["entries"]:
            store.entries.append(EngramEntry(
                text=e["text"],
                mean_entropy=e["mean_entropy"],
                condition=e["condition"],
                source=e.get("source", ""),
            ))

        print(f"Loaded {len(store.entries)} engrams from {path}")
        return store

    def stats(self) -> dict:
        """Return summary statistics about the store."""
        if not self.entries:
            return {"n_entries": 0}

        entropies = [e.mean_entropy for e in self.entries]
        conditions = {}
        for e in self.entries:
            conditions[e.condition] = conditions.get(e.condition, 0) + 1

        return {
            "n_entries": len(self.entries),
            "mean_entropy": sum(entropies) / len(entropies),
            "min_entropy": min(entropies),
            "max_entropy": max(entropies),
            "conditions": conditions,
        }
