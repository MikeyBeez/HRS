"""Data loading for WikiText-103 with GPT-2 tokenization.

Memory-efficient: tokenizes in chunks to avoid OOM on WikiText-103's
~500MB training set.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText dataset chunked into fixed-length sequences."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        n_seqs = (len(tokens) - 1) // seq_len
        self.tokens = tokens[: n_seqs * seq_len + 1]

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


def _tokenize_chunked(texts: list, tokenizer, chunk_size: int = 10000) -> list:
    """Tokenize a list of text strings in chunks to avoid OOM.

    Args:
        texts: list of text strings from the dataset
        tokenizer: HuggingFace tokenizer
        chunk_size: number of text rows to process at once

    Returns:
        list of token ids
    """
    all_ids = []
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        text = "\n".join(chunk)
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
    return all_ids


def load_wikitext(name: str = "wikitext-103", seq_len: int = 512):
    """Load and tokenize WikiText, returning train/val/test datasets.

    Uses chunked tokenization to handle WikiText-103's large training set
    without running out of memory.
    """
    from datasets import load_dataset

    ds_name = "wikitext"
    ds_config = "wikitext-2-raw-v1" if name == "wikitext-2" else "wikitext-103-raw-v1"

    print(f"  Loading dataset {ds_name}/{ds_config}...")
    raw = load_dataset(ds_name, ds_config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    splits = {}
    for split_name in ["train", "validation", "test"]:
        print(f"  Tokenizing {split_name}...")
        texts = raw[split_name]["text"]
        token_ids = _tokenize_chunked(texts, tokenizer, chunk_size=10000)
        tokens = torch.tensor(token_ids, dtype=torch.long)
        splits[split_name] = WikiTextDataset(tokens, seq_len)
        print(f"  {split_name}: {len(splits[split_name])} sequences of length {seq_len} ({len(tokens):,} tokens)")

    return splits, tokenizer


def build_dataloaders(splits: dict, batch_size: int = 24, num_workers: int = 4):
    """Wrap datasets in DataLoaders."""
    loaders = {}
    for name, ds in splits.items():
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(name == "train"),
        )
    return loaders
