"""Data loading for WikiText-103 with GPT-2 tokenization.

Memory-efficient: tokenizes in chunks to avoid OOM on WikiText-103's
~500MB training set.
"""

import re

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class WikiTextDataset(Dataset):
    """Pre-tokenized WikiText dataset chunked into fixed-length sequences."""

    def __init__(self, tokens: torch.Tensor, seq_len: int, category_ids: torch.Tensor = None):
        self.seq_len = seq_len
        n_seqs = (len(tokens) - 1) // seq_len
        self.tokens = tokens[: n_seqs * seq_len + 1]
        self.category_ids = category_ids  # (n_tokens,) or None

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        if self.category_ids is not None:
            # Majority category in this sequence window
            cat_slice = self.category_ids[start : start + self.seq_len]
            cat_id = cat_slice.mode().values.item()
            return x, y, cat_id
        return x, y


def _extract_articles(texts: list) -> list:
    """Extract article boundaries and texts from WikiText-103.

    WikiText-103 marks article titles with ' = Title = \\n' pattern (single =, space-padded).
    Section headers use ' = = Section = = \\n' (double+ =).
    Returns list of (title, text) tuples.
    """
    # Match top-level titles only (single =, not == or ===)
    title_pattern = re.compile(r'^\s*=\s+([^=]+?)\s+=\s*$')
    articles = []
    current_title = None
    current_lines = []

    for line in texts:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip section headers (== or ===)
        if stripped.startswith('= =') or stripped.startswith('= = ='):
            current_lines.append(stripped)
            continue
        m = title_pattern.match(stripped)
        if m:
            if current_title is not None and current_lines:
                articles.append((current_title.strip(), "\n".join(current_lines)))
            current_title = m.group(1)
            current_lines = []
        else:
            current_lines.append(stripped)

    if current_title is not None and current_lines:
        articles.append((current_title.strip(), "\n".join(current_lines)))

    return articles


def _cluster_articles(articles: list, n_clusters: int = 50) -> list:
    """Cluster articles into topic categories using TF-IDF + KMeans.

    Returns list of integer category IDs (one per article).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import MiniBatchKMeans

    texts = [text for _, text in articles]

    # Use a lightweight TF-IDF (max 5000 features, bigrams)
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        max_df=0.95,
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(texts)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
    labels = kmeans.fit_predict(tfidf)
    return labels.tolist()


def _build_category_ids(
    texts: list, tokenizer, articles: list, article_labels: list, seq_len: int,
    chunk_size: int = 10000,
) -> torch.Tensor:
    """Map each token position to its article's category ID.

    Processes text in chunks (matching _tokenize_chunked) and assigns each
    token the category of its enclosing article.
    Returns (n_tokens,) tensor of category IDs.
    """
    title_pattern = re.compile(r'^\s*=\s+([^=]+?)\s+=\s*$')
    article_title_to_idx = {title: i for i, (title, _) in enumerate(articles)}

    # Build per-line category assignment (fast — no tokenization)
    line_labels = []
    current_label = 0
    for line in texts:
        stripped = line.strip()
        if stripped and not stripped.startswith('= ='):
            m = title_pattern.match(stripped)
            if m:
                title = m.group(1).strip()
                if title in article_title_to_idx:
                    idx = article_title_to_idx[title]
                    current_label = article_labels[idx]
        line_labels.append(current_label)

    # Tokenize in chunks matching _tokenize_chunked, tracking category per token
    all_cat_ids = []
    for i in range(0, len(texts), chunk_size):
        chunk_texts = texts[i : i + chunk_size]
        chunk_labels = line_labels[i : i + chunk_size]
        text = "\n".join(chunk_texts)
        ids = tokenizer.encode(text, add_special_tokens=False)

        # Approximate: assign each token the category of the dominant line in the chunk
        # More accurate: since lines join with \n, assign per-line category proportionally
        if len(chunk_texts) == 1:
            all_cat_ids.extend([chunk_labels[0]] * len(ids))
        else:
            # Tokenize each line to get token counts per line
            # Use fast approximation: character ratio
            line_char_counts = [len(t) + 1 for t in chunk_texts]  # +1 for \n
            total_chars = sum(line_char_counts)
            token_idx = 0
            for j, (chars, label) in enumerate(zip(line_char_counts, chunk_labels)):
                n_tokens_for_line = max(1, round(len(ids) * chars / total_chars))
                if j == len(chunk_texts) - 1:
                    n_tokens_for_line = len(ids) - token_idx  # assign remaining
                all_cat_ids.extend([label] * n_tokens_for_line)
                token_idx += n_tokens_for_line

    return torch.tensor(all_cat_ids, dtype=torch.long)


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


def load_wikitext(name: str = "wikitext-103", seq_len: int = 512, with_categories: bool = False, n_categories: int = 50):
    """Load and tokenize WikiText, returning train/val/test datasets.

    Uses chunked tokenization to handle WikiText-103's large training set
    without running out of memory.

    Args:
        name: dataset name ("wikitext-103" or "wikitext-2")
        seq_len: sequence length for chunking
        with_categories: if True, compute per-token topic category labels
        n_categories: number of topic clusters (only used if with_categories=True)
    """
    from datasets import load_dataset

    ds_name = "wikitext"
    ds_config = "wikitext-2-raw-v1" if name == "wikitext-2" else "wikitext-103-raw-v1"

    print(f"  Loading dataset {ds_name}/{ds_config}...")
    raw = load_dataset(ds_name, ds_config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Pre-compute category labels from training split if needed
    articles = None
    article_labels = None
    if with_categories:
        print(f"  Extracting articles for category labeling...")
        train_texts = raw["train"]["text"]
        articles = _extract_articles(train_texts)
        print(f"  Found {len(articles)} articles, clustering into {n_categories} categories...")
        article_labels = _cluster_articles(articles, n_clusters=n_categories)
        print(f"  Category distribution: {len(set(article_labels))} unique categories")

    splits = {}
    for split_name in ["train", "validation", "test"]:
        print(f"  Tokenizing {split_name}...")
        texts = raw[split_name]["text"]
        token_ids = _tokenize_chunked(texts, tokenizer, chunk_size=10000)
        tokens = torch.tensor(token_ids, dtype=torch.long)

        cat_ids = None
        if with_categories and articles is not None:
            print(f"  Building category IDs for {split_name}...")
            cat_ids = _build_category_ids(texts, tokenizer, articles, article_labels, seq_len)
            # Trim/pad to match token count
            if len(cat_ids) > len(tokens):
                cat_ids = cat_ids[:len(tokens)]
            elif len(cat_ids) < len(tokens):
                # Pad with last known category
                pad_val = cat_ids[-1].item() if len(cat_ids) > 0 else 0
                cat_ids = torch.cat([cat_ids, torch.full((len(tokens) - len(cat_ids),), pad_val, dtype=torch.long)])

        splits[split_name] = WikiTextDataset(tokens, seq_len, category_ids=cat_ids)
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
