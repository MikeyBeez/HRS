"""Topic-Filtered Context Manager for V18.

Online clustering system that organizes prompts by topic using engram
similarity. Maintains clusters with evolving centroids and linked-list
prompt chains. Two-slot active buffer: current topic + previous topic.

The context window is filled with prompts from the active cluster(s)
instead of naive recency-based sliding window.
"""

import re
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PromptNode:
    """A single prompt in a topic cluster's linked list."""
    text: str
    token_ids: torch.Tensor       # (T,) token ids
    engram: torch.Tensor           # (D,) engram vector
    timestamp: int                 # insertion order


# Stop words for keyword extraction
_STOP_WORDS = frozenset(
    "the a an is was were be been being have has had do does did will would "
    "shall should may might can could to of in for on with at by from as into "
    "through during before after above below between out off over under again "
    "further then once here there when where why how all each every both few "
    "more most other some such no nor not only own same so than too very and "
    "but if or because until while that this these those it its he she they "
    "his her their him them what which who whom whose are am about up also "
    "just don didn doesn like get got much many well still even back".split()
)


def _extract_title(texts: List[str], max_words: int = 4) -> str:
    """Extract a short descriptive title from a collection of texts.

    Uses term frequency to find distinctive keywords, filters stop words
    and short tokens, returns the top keywords joined as a title.
    """
    word_counts = Counter()
    for text in texts:
        words = re.findall(r'[A-Za-z][a-z]{2,}', text)  # 3+ char words starting with letter
        for w in words:
            wl = w.lower()
            if wl not in _STOP_WORDS:
                word_counts[wl] += 1

    if not word_counts:
        return texts[0][:30] if texts else "Untitled"

    # Take top keywords, capitalize
    top = [word for word, _ in word_counts.most_common(max_words)]
    return " / ".join(w.capitalize() for w in top)


class TopicCluster:
    """A topic cluster with an evolving centroid and ordered prompt chain."""

    def __init__(self, cluster_id: int, first_engram: torch.Tensor, first_node: PromptNode):
        self.cluster_id = cluster_id
        self.centroid = F.normalize(first_engram.float(), dim=0)
        self.prompts: List[PromptNode] = [first_node]
        self.n_updates = 1
        self.title: str = _extract_title([first_node.text])
        self.user_enabled: bool = True  # can be toggled by UI

    def add(self, node: PromptNode):
        """Add a prompt and update the centroid and title."""
        self.prompts.append(node)
        self.n_updates += 1
        # Running mean: centroid = (centroid * (n-1) + new) / n
        new_engram = F.normalize(node.engram.float(), dim=0)
        self.centroid = (self.centroid * (self.n_updates - 1) + new_engram) / self.n_updates
        self.centroid = F.normalize(self.centroid, dim=0)
        # Refresh title from all prompts
        self.title = _extract_title([p.text for p in self.prompts])

    def get_context_tokens(self, max_tokens: int) -> torch.Tensor:
        """Get token ids from this cluster's prompts, most recent first.

        Packs as many recent prompts as fit within max_tokens.

        Returns:
            (T,) token ids, or empty tensor if no prompts
        """
        chunks = []
        total = 0
        # Walk backwards (most recent first)
        for node in reversed(self.prompts):
            n = node.token_ids.shape[0]
            if total + n > max_tokens:
                # Take partial from this prompt to fill remaining space
                remaining = max_tokens - total
                if remaining > 0:
                    chunks.append(node.token_ids[-remaining:])
                break
            chunks.append(node.token_ids)
            total += n

        if not chunks:
            return torch.tensor([], dtype=torch.long)

        # Reverse to restore chronological order
        chunks.reverse()
        return torch.cat(chunks)

    def similarity(self, engram: torch.Tensor) -> float:
        """Cosine similarity between an engram and this cluster's centroid."""
        engram = F.normalize(engram.float(), dim=0)
        return (self.centroid @ engram).item()

    def __len__(self):
        return len(self.prompts)

    def total_tokens(self) -> int:
        return sum(n.token_ids.shape[0] for n in self.prompts)


class TopicContextManager:
    """Manages topic-clustered context with a two-slot active buffer.

    Active buffer holds N most recently used topic clusters (configurable,
    default 2). When a new topic activates, the least recently used slot
    is evicted. If a prompt matches an already-active cluster, that cluster
    moves to the front (MRU position).

    Args:
        model: V18 model (for engram extraction)
        tokenizer: GPT-2 tokenizer
        similarity_threshold: min cosine similarity to join existing cluster
        max_context_tokens: max tokens to load into context window
        extract_layer: which layer to extract engrams from (0-indexed)
        max_active: max number of simultaneously active topic clusters
    """

    def __init__(
        self,
        model,
        tokenizer,
        similarity_threshold: float = 0.4,
        max_context_tokens: int = 512,
        extract_layer: int = -2,
        max_active: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.similarity_threshold = similarity_threshold
        self.max_context_tokens = max_context_tokens
        self.max_active = max_active
        self.device = next(model.parameters()).device

        n_layers = model.cfg.model.n_layers
        self.extract_layer = extract_layer if extract_layer >= 0 else n_layers + extract_layer

        # Merge threshold: if two centroids are this similar, merge them
        self.merge_threshold = min(0.85, similarity_threshold + 0.35)

        # All clusters (persistent — never deleted, but may be merged)
        self.clusters: List[TopicCluster] = []
        self._next_cluster_id = 0
        self._timestamp = 0

        # Active buffer: ordered list, index 0 = most recently used
        self.active: List[TopicCluster] = []

    @torch.no_grad()
    def extract_engram(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Extract engram from token ids via forward pass.

        Args:
            token_ids: (T,) or (1, T) token ids

        Returns:
            (D,) engram vector
        """
        self.model.eval()
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        token_ids = token_ids[:, :512].to(self.device)  # truncate to max_seq_len

        # Hook to capture hidden states at extraction layer
        captured = {}
        def hook_fn(module, inp, out):
            captured['h'] = out[0].detach()

        handle = self.model.blocks[self.extract_layer].register_forward_hook(hook_fn)
        _ = self.model(token_ids, step=0)
        handle.remove()

        # Mean-pool across positions
        engram = captured['h'].mean(dim=1).squeeze(0).cpu()  # (D,)
        return engram

    def find_nearest_cluster(self, engram: torch.Tensor) -> Tuple[Optional[TopicCluster], float]:
        """Find the cluster with highest similarity to an engram.

        Returns:
            (cluster, similarity) or (None, 0.0) if no cluster above threshold
        """
        if not self.clusters:
            return None, 0.0

        best_cluster = None
        best_sim = 0.0

        for cluster in self.clusters:
            sim = cluster.similarity(engram)
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_sim >= self.similarity_threshold:
            return best_cluster, best_sim
        return None, best_sim

    def process_prompt(self, text: str) -> dict:
        """Process an incoming prompt: classify, cluster, update active buffer.

        Args:
            text: the prompt text

        Returns:
            dict with routing info (cluster_id, is_new, similarity, active slots)
        """
        # Tokenize
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        token_ids = torch.tensor(ids, dtype=torch.long)

        # Extract engram
        engram = self.extract_engram(token_ids)

        # Create prompt node
        self._timestamp += 1
        node = PromptNode(
            text=text,
            token_ids=token_ids,
            engram=engram,
            timestamp=self._timestamp,
        )

        # Find nearest cluster
        nearest, sim = self.find_nearest_cluster(engram)

        if nearest is not None:
            # Join existing cluster
            nearest.add(node)
            is_new = False

            # Move to front of active buffer (MRU)
            if nearest in self.active:
                self.active.remove(nearest)
            self.active.insert(0, nearest)
            # Evict LRU if over capacity
            if len(self.active) > self.max_active:
                self.active = self.active[:self.max_active]
        else:
            # New topic — create new cluster
            nearest = TopicCluster(
                cluster_id=self._next_cluster_id,
                first_engram=engram,
                first_node=node,
            )
            self._next_cluster_id += 1
            self.clusters.append(nearest)
            is_new = True

            # Insert at front, evict LRU if needed
            self.active.insert(0, nearest)
            if len(self.active) > self.max_active:
                self.active = self.active[:self.max_active]
            sim = 1.0

        # Check if any clusters should be merged
        self._check_merge()

        return {
            "cluster_id": nearest.cluster_id,
            "is_new_topic": is_new,
            "similarity": sim,
            "cluster_size": len(nearest),
            "cluster_tokens": nearest.total_tokens(),
            "n_clusters": len(self.clusters),
            "active_ids": [c.cluster_id for c in self.active],
        }

    def _check_merge(self):
        """Check if any two clusters are close enough to merge.

        If two centroids have cosine similarity above merge_threshold,
        merge the smaller into the larger. Updates active buffer refs.
        """
        if len(self.clusters) < 2:
            return

        merged = True
        while merged:
            merged = False
            for i in range(len(self.clusters)):
                for j in range(i + 1, len(self.clusters)):
                    ci, cj = self.clusters[i], self.clusters[j]
                    sim = (ci.centroid @ cj.centroid).item()
                    if sim >= self.merge_threshold:
                        # Merge smaller into larger
                        if len(ci) >= len(cj):
                            keeper, absorbed = ci, cj
                        else:
                            keeper, absorbed = cj, ci

                        # Move all prompts
                        for node in absorbed.prompts:
                            keeper.add(node)
                        keeper.user_enabled = keeper.user_enabled or absorbed.user_enabled

                        # Update active buffer refs
                        self.active = [keeper if c is absorbed else c for c in self.active]
                        # Deduplicate active
                        seen = set()
                        deduped = []
                        for c in self.active:
                            if id(c) not in seen:
                                seen.add(id(c))
                                deduped.append(c)
                        self.active = deduped[:self.max_active]

                        # Remove absorbed
                        self.clusters.remove(absorbed)
                        merged = True
                        break
                if merged:
                    break

    def set_cluster_enabled(self, cluster_id: int, enabled: bool):
        """Toggle a cluster on/off (for UI control)."""
        for c in self.clusters:
            if c.cluster_id == cluster_id:
                c.user_enabled = enabled
                return
        raise ValueError(f"No cluster with id {cluster_id}")

    def list_topics(self) -> List[dict]:
        """List all clusters with their titles and status (for UI display).

        Returns list of dicts suitable for rendering a topic selector.
        """
        active_ids = {id(c) for c in self.active}
        return [
            {
                "cluster_id": c.cluster_id,
                "title": c.title,
                "enabled": c.user_enabled,
                "active": id(c) in active_ids,
                "n_prompts": len(c),
                "n_tokens": c.total_tokens(),
            }
            for c in self.clusters
        ]

    def get_context(self) -> torch.Tensor:
        """Get context tokens from active, user-enabled clusters.

        Budget is distributed with exponential decay by recency:
        MRU cluster gets 50% of remaining budget, next gets 50% of what's
        left, etc. This gives the current topic the most space while still
        including secondary topics. Clusters disabled by the user are skipped.

        Returns:
            (T,) token ids for context, up to max_context_tokens
        """
        enabled_active = [c for c in self.active if c.user_enabled]

        if not enabled_active:
            return torch.tensor([], dtype=torch.long)

        if len(enabled_active) == 1:
            return enabled_active[0].get_context_tokens(self.max_context_tokens)

        # Distribute budget: each active cluster gets half of remaining
        chunks = []
        remaining = self.max_context_tokens
        budgets = []
        for i, cluster in enumerate(enabled_active):
            if i == len(enabled_active) - 1:
                budget = remaining
            else:
                budget = remaining // 2
            budgets.append(budget)
            remaining -= budget

        # Gather tokens (reverse order so older context comes first)
        token_chunks = []
        for cluster, budget in reversed(list(zip(enabled_active, budgets))):
            tokens = cluster.get_context_tokens(budget)
            if tokens.shape[0] > 0:
                token_chunks.append(tokens)

        if not token_chunks:
            return torch.tensor([], dtype=torch.long)

        return torch.cat(token_chunks)

    def get_context_engram(self) -> Optional[torch.Tensor]:
        """Get the MRU cluster's centroid for cross-attention injection.

        Returns:
            (D,) centroid vector, or None if no active cluster
        """
        if not self.active:
            return None
        return self.active[0].centroid

    def stats(self) -> dict:
        """Return summary statistics."""
        return {
            "n_clusters": len(self.clusters),
            "total_prompts": sum(len(c) for c in self.clusters),
            "total_tokens": sum(c.total_tokens() for c in self.clusters),
            "cluster_sizes": {c.cluster_id: len(c) for c in self.clusters},
            "active_ids": [c.cluster_id for c in self.active],
            "max_active": self.max_active,
            "threshold": self.similarity_threshold,
        }

    def describe_clusters(self) -> str:
        """Human-readable summary of all clusters."""
        active_ids = {id(c) for c in self.active}
        lines = []
        for c in self.clusters:
            marker = ""
            if id(c) in active_ids:
                pos = self.active.index(c)
                marker = f" [ACTIVE #{pos}]"
            enabled = "ON" if c.user_enabled else "OFF"
            lines.append(
                f"  [{enabled}] Cluster {c.cluster_id} \"{c.title}\"{marker}: "
                f"{len(c)} prompts, {c.total_tokens()} tokens"
            )
        return "\n".join(lines)
