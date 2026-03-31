"""Retrieval Engine: entropy-gated engram retrieval for V18 inference.

Ties together the EntropyMonitor, EngramStore, and V18 model to provide
entropy-gated engram storage and retrieval during inference.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from config import ExperimentConfig, AblationConfig
from model import HRSTransformer
from engram_store import EngramStore, EngramEntry
from entropy_monitor import EntropyMonitor


class RetrievalEngine:
    """Entropy-gated engram retrieval for V18 inference.

    Write path: Process text, compute entropy, store engram if high-entropy.
    Read path: During generation, monitor entropy, retrieve engram when confused.
    """

    def __init__(
        self,
        model: HRSTransformer,
        store: EngramStore,
        write_threshold: float = 4.0,
        read_threshold: float = 4.0,
        read_window: int = 10,
        top_k: int = 1,
        min_similarity: float = 0.3,
        extract_layer: int = -2,
    ):
        self.model = model
        self.store = store
        self.monitor = EntropyMonitor(
            write_threshold=write_threshold,
            read_threshold=read_threshold,
            read_window=read_window,
        )
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.device = next(model.parameters()).device

        # Resolve extract layer index
        n_layers = model.cfg.model.n_layers
        self.extract_layer = extract_layer if extract_layer >= 0 else n_layers + extract_layer

        # Stats
        self.n_stores = 0
        self.n_retrieval_triggers = 0
        self.n_retrievals_found = 0

    @torch.no_grad()
    def compute_engram(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass and extract engram + logits.

        Args:
            input_ids: (1, T) or (B, T) token ids

        Returns:
            (engram, logits) where engram is (B, d_model) and logits is (B, T, V)
        """
        self.model.eval()
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        # Hook to capture hidden states at extraction layer
        hidden_at_layer = {}

        def hook_fn(module, input, output):
            # HRSBlock forward returns (x, routing_w, attn_w, kv_cache)
            x = output[0]
            hidden_at_layer['h'] = x.detach()

        block = self.model.blocks[self.extract_layer]
        handle = block.register_forward_hook(hook_fn)

        output = self.model(input_ids, step=0)

        handle.remove()

        # Mean-pool hidden states across token positions: (B, D)
        h = hidden_at_layer['h']
        engram = h.mean(dim=1)  # (B, D)

        return engram, output.logits

    def process_segment(
        self,
        text: str,
        tokenizer,
        condition: str = "full_context",
        source: str = "",
        context_ids: Optional[torch.Tensor] = None,
    ) -> tuple[bool, float]:
        """Process a text segment: compute entropy, optionally store engram.

        Args:
            text: text segment to process
            tokenizer: tokenizer for encoding
            condition: 'full_context' or 'isolated'
            source: source identifier
            context_ids: preceding context token ids (for full_context condition)

        Returns:
            (was_stored, mean_entropy)
        """
        # Tokenize
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)

        if condition == "full_context" and context_ids is not None:
            # Prepend context, but only use last 512 tokens total
            full_ids = torch.cat([context_ids.squeeze(0), ids_tensor.squeeze(0)])[-512:]
            full_ids = full_ids.unsqueeze(0)
        else:
            # Isolated: just the segment itself
            full_ids = ids_tensor[:, :512]

        # Forward pass
        engram, logits = self.compute_engram(full_ids)

        # Compute entropy over the segment tokens (not context)
        if condition == "full_context" and context_ids is not None:
            ctx_len = min(context_ids.shape[-1], 512 - len(ids))
            segment_logits = logits[:, ctx_len:, :]
        else:
            segment_logits = logits

        should_store, mean_entropy = self.monitor.should_store(segment_logits)

        if should_store:
            entry = EngramEntry(
                text=text,
                mean_entropy=mean_entropy,
                condition=condition,
                source=source,
            )
            self.store.store(engram.squeeze(0), entry)
            self.n_stores += 1

        return should_store, mean_entropy

    @torch.no_grad()
    def generate_with_retrieval(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.9,
        top_k_sampling: int = 50,
    ) -> tuple[torch.Tensor, dict]:
        """Generate tokens with entropy-gated engram retrieval.

        When rolling entropy exceeds the read threshold, retrieves the nearest
        engram from the store and injects it into V18's cross-attention buffer.

        Args:
            prompt_ids: (1, T) or (T,) prompt token ids
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k_sampling: top-k filtering for sampling

        Returns:
            (generated_ids, stats) where stats contains retrieval info
        """
        self.model.eval()
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_ids = prompt_ids.to(self.device)

        input_ids = prompt_ids.clone()
        max_seq_len = 512

        self.monitor.reset_rolling()
        stats = {
            "triggers": [],
            "n_triggers": 0,
            "n_retrievals": 0,
            "original_buffer": self.model.engram_buffer.clone() if self.model._engram_buffer_initialized else None,
        }

        # Save original engram buffer to restore after retrieval windows
        original_buffer = self.model.engram_buffer.clone() if self.model._engram_buffer_initialized else None
        retrieval_active = False
        retrieval_cooldown = 0

        for step in range(max_new_tokens):
            # Truncate to last max_seq_len tokens (sliding window, matching training)
            idx = input_ids[:, -max_seq_len:]

            output = self.model(idx, step=0)
            logits = output.logits[:, -1:, :]  # (1, 1, V)

            # Compute entropy of this token's distribution
            token_ent = EntropyMonitor.token_entropy(logits).item()
            self.monitor.update_rolling(token_ent)

            # Check if we should trigger retrieval
            if retrieval_cooldown > 0:
                retrieval_cooldown -= 1
            else:
                should_retrieve, rolling_ent = self.monitor.should_retrieve()
                if should_retrieve and not retrieval_active:
                    # Compute current context engram for query
                    context_engram = self._compute_context_engram(idx)

                    # Search store
                    results = self.store.retrieve(
                        context_engram, top_k=self.top_k,
                        min_similarity=self.min_similarity,
                    )

                    if results:
                        # Inject retrieved engram(s) into cross-attention buffer
                        retrieved_engrams = torch.stack([r[2] for r in results], dim=0)  # (K, D)
                        # Expand to match buffer shape (1, N, D) by repeating
                        n_slots = self.model.engram_buffer.shape[1]
                        if retrieved_engrams.shape[0] < n_slots:
                            n_repeats = (n_slots + retrieved_engrams.shape[0] - 1) // retrieved_engrams.shape[0]
                            retrieved_engrams = retrieved_engrams.repeat(n_repeats, 1)[:n_slots]
                        self.model.engram_buffer.data = retrieved_engrams.unsqueeze(0).to(self.device)
                        self.model._engram_buffer_initialized = True
                        retrieval_active = True
                        retrieval_cooldown = self.monitor.read_window  # cooldown after trigger

                        stats["triggers"].append({
                            "step": step,
                            "rolling_entropy": rolling_ent,
                            "token_entropy": token_ent,
                            "similarities": [r[0] for r in results],
                            "retrieved_texts": [r[1].text[:100] for r in results],
                        })
                        stats["n_triggers"] += 1
                        stats["n_retrievals"] += 1
                        self.n_retrieval_triggers += 1
                        self.n_retrievals_found += 1

            # Restore original buffer after cooldown expires
            if retrieval_active and retrieval_cooldown == 0:
                if original_buffer is not None:
                    self.model.engram_buffer.data = original_buffer.clone()
                else:
                    self.model._engram_buffer_initialized = False
                retrieval_active = False

            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            if top_k_sampling > 0:
                v, _ = torch.topk(next_logits, top_k_sampling)
                next_logits[next_logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Restore original buffer
        if original_buffer is not None:
            self.model.engram_buffer.data = original_buffer
        stats["trigger_rate"] = stats["n_triggers"] / max_new_tokens if max_new_tokens > 0 else 0

        return input_ids, stats

    def _compute_context_engram(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Compute engram from current context for retrieval query.

        Args:
            context_ids: (1, T) current context

        Returns:
            (d_model,) engram vector
        """
        engram, _ = self.compute_engram(context_ids)
        return engram.squeeze(0)  # (d_model,)

    def get_stats(self) -> dict:
        """Return cumulative statistics."""
        return {
            "n_stores": self.n_stores,
            "n_retrieval_triggers": self.n_retrieval_triggers,
            "n_retrievals_found": self.n_retrievals_found,
            "store_size": len(self.store),
            "store_stats": self.store.stats(),
        }
