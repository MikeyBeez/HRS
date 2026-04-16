"""Phase C: Train a small per-token tagger on the R1-generated ground truth.

Architecture: GPT-2 BPE embeddings → 3-layer 1D convolutions → per-token
classification heads (content_type, pos, entity_type, salience) + a
sequence-level query_type head.

The tagger maps GPT-2 BPE tokens to metadata annotations. Ground truth
annotations are word-level; we expand them to BPE subtoken level by
assigning each subtoken the annotation of its parent word.

Usage:
    PYTHONPATH=/mnt/data/Code/HRS .venv/bin/python experiments/identity_ae/preprocessing/train_tagger.py
"""

import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

# -------------------------------------------------------------------
# Label vocabularies
# -------------------------------------------------------------------

CONTENT_TYPES = ["content", "template", "punctuation"]
POS_TAGS = ["noun", "verb", "adj", "adv", "det", "prep", "conj", "pron", "other"]
ENTITY_TYPES = ["none", "person", "location", "org", "artifact", "number",
                "technical_term"]
GRAM_ROLES = ["subject", "object", "predicate", "modifier", "root", "other"]
QUERY_TYPES = ["factual", "numeric", "entity", "technical", "procedural",
               "compositional", "declarative", "other"]

CT2I = {v: i for i, v in enumerate(CONTENT_TYPES)}
POS2I = {v: i for i, v in enumerate(POS_TAGS)}
ET2I = {v: i for i, v in enumerate(ENTITY_TYPES)}
GR2I = {v: i for i, v in enumerate(GRAM_ROLES)}
QT2I = {v: i for i, v in enumerate(QUERY_TYPES)}


# -------------------------------------------------------------------
# Data loading + BPE alignment
# -------------------------------------------------------------------

def load_ground_truth(gt_dir):
    """Load ground-truth JSON files and return a list of annotation dicts."""
    gt_dir = Path(gt_dir)
    manifest = gt_dir / "manifest.csv"
    entries = []
    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath = gt_dir / row["filename"]
            if fpath.exists():
                with open(fpath) as jf:
                    entries.append(json.load(jf))
    return entries


def align_to_bpe(text, annotation, tokenizer):
    """Align word-level annotations to GPT-2 BPE tokens.

    Returns:
        bpe_ids: list of int (token ids)
        bpe_ct: list of int (content_type labels)
        bpe_pos: list of int (POS labels)
        bpe_et: list of int (entity_type labels)
        bpe_sal: list of float (salience values)
        query_type: int
    """
    words = annotation.get("tokens", [])
    word_anns = annotation.get("token_annotations", [])
    query_type_str = annotation.get("query_type", "other")
    qt = QT2I.get(query_type_str, QT2I["other"])

    if not words or len(words) != len(word_anns):
        return None

    # Tokenize each word with GPT-2 BPE and propagate word-level labels
    bpe_ids = []
    bpe_ct = []
    bpe_pos = []
    bpe_et = []
    bpe_gr = []
    bpe_sal = []

    for w_idx, (word, ann) in enumerate(zip(words, word_anns)):
        # R1 sometimes returns non-string tokens (ints, None)
        if not isinstance(word, str):
            word = str(word)
        # Add a leading space for non-first words (GPT-2 convention)
        w_text = f" {word}" if w_idx > 0 else word
        subtokens = tokenizer.encode(w_text, add_special_tokens=False)
        if not subtokens:
            continue

        ct = CT2I.get(ann.get("content_type", "template"), CT2I["template"])
        pos = POS2I.get(ann.get("pos", "other"), POS2I["other"])
        et = ET2I.get(ann.get("entity_type", "none"), ET2I["none"])
        gr = GR2I.get(ann.get("gram_role", "other"), GR2I["other"])
        sal = float(ann.get("salience", 0.0))
        # Distribute salience evenly across subtokens
        sal_per = sal / len(subtokens)

        for st in subtokens:
            bpe_ids.append(st)
            bpe_ct.append(ct)
            bpe_pos.append(pos)
            bpe_et.append(et)
            bpe_gr.append(gr)
            bpe_sal.append(sal_per)

    if not bpe_ids:
        return None

    return {
        "bpe_ids": bpe_ids,
        "bpe_ct": bpe_ct,
        "bpe_pos": bpe_pos,
        "bpe_et": bpe_et,
        "bpe_gr": bpe_gr,
        "bpe_sal": bpe_sal,
        "query_type": qt,
    }


def prepare_dataset(entries, tokenizer, max_len=256):
    """Convert ground-truth entries to aligned BPE training examples."""
    examples = []
    n_skip = 0
    for entry in entries:
        ann = entry.get("annotation")
        if ann is None:
            n_skip += 1
            continue
        aligned = align_to_bpe(entry["text"], ann, tokenizer)
        if aligned is None:
            n_skip += 1
            continue
        # Truncate
        L = min(len(aligned["bpe_ids"]), max_len)
        examples.append({
            "ids": aligned["bpe_ids"][:L],
            "ct": aligned["bpe_ct"][:L],
            "pos": aligned["bpe_pos"][:L],
            "et": aligned["bpe_et"][:L],
            "gr": aligned["bpe_gr"][:L],
            "sal": aligned["bpe_sal"][:L],
            "qt": aligned["query_type"],
        })
    if n_skip:
        print(f"  skipped {n_skip} entries with bad annotations")
    return examples


def collate(batch, device):
    """Pad and stack a batch of examples."""
    max_len = max(len(ex["ids"]) for ex in batch)
    B = len(batch)
    ids = torch.zeros(B, max_len, dtype=torch.long, device=device)
    ct = torch.zeros(B, max_len, dtype=torch.long, device=device)
    pos = torch.zeros(B, max_len, dtype=torch.long, device=device)
    et = torch.zeros(B, max_len, dtype=torch.long, device=device)
    gr = torch.zeros(B, max_len, dtype=torch.long, device=device)
    sal = torch.zeros(B, max_len, device=device)
    qt = torch.zeros(B, dtype=torch.long, device=device)
    mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

    for i, ex in enumerate(batch):
        L = len(ex["ids"])
        ids[i, :L] = torch.tensor(ex["ids"])
        ct[i, :L] = torch.tensor(ex["ct"])
        pos[i, :L] = torch.tensor(ex["pos"])
        et[i, :L] = torch.tensor(ex["et"])
        gr[i, :L] = torch.tensor(ex["gr"])
        sal[i, :L] = torch.tensor(ex["sal"])
        qt[i] = ex["qt"]
        mask[i, :L] = True

    return ids, ct, pos, et, gr, sal, qt, mask


# -------------------------------------------------------------------
# Model
# -------------------------------------------------------------------

class PromptTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, 5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2)

        self.ct_head = nn.Linear(hidden_dim, len(CONTENT_TYPES))
        self.pos_head = nn.Linear(hidden_dim, len(POS_TAGS))
        self.et_head = nn.Linear(hidden_dim, len(ENTITY_TYPES))
        self.gr_head = nn.Linear(hidden_dim, len(GRAM_ROLES))
        self.sal_head = nn.Linear(hidden_dim, 1)
        self.qt_head = nn.Linear(hidden_dim, len(QUERY_TYPES))

    def forward(self, token_ids):
        x = self.embed(token_ids)                  # (B, T, E)
        x = x.transpose(1, 2)                      # (B, E, T)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.transpose(1, 2)                      # (B, T, H)

        ct_logits = self.ct_head(x)                # (B, T, 3)
        pos_logits = self.pos_head(x)              # (B, T, 9)
        et_logits = self.et_head(x)                # (B, T, 7)
        gr_logits = self.gr_head(x)                # (B, T, 6)
        sal = torch.sigmoid(self.sal_head(x))      # (B, T, 1)
        qt_logits = self.qt_head(x.mean(dim=1))   # (B, 8)

        return ct_logits, pos_logits, et_logits, gr_logits, sal.squeeze(-1), qt_logits


# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------

def train_epoch(model, examples, device, optimizer, batch_size=32):
    model.train()
    random.shuffle(examples)
    total_loss = 0
    n_batches = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        ids, ct, pos, et, gr, sal, qt, mask = collate(batch, device)
        ct_l, pos_l, et_l, gr_l, sal_p, qt_l = model(ids)

        # Masked per-token losses
        B, T = ids.shape
        ct_loss = F.cross_entropy(ct_l.reshape(-1, ct_l.size(-1)),
                                  ct.reshape(-1), reduction='none')
        ct_loss = (ct_loss.reshape(B, T) * mask).sum() / mask.sum()

        pos_loss = F.cross_entropy(pos_l.reshape(-1, pos_l.size(-1)),
                                   pos.reshape(-1), reduction='none')
        pos_loss = (pos_loss.reshape(B, T) * mask).sum() / mask.sum()

        et_loss = F.cross_entropy(et_l.reshape(-1, et_l.size(-1)),
                                  et.reshape(-1), reduction='none')
        et_loss = (et_loss.reshape(B, T) * mask).sum() / mask.sum()

        gr_loss = F.cross_entropy(gr_l.reshape(-1, gr_l.size(-1)),
                                  gr.reshape(-1), reduction='none')
        gr_loss = (gr_loss.reshape(B, T) * mask).sum() / mask.sum()

        sal_loss = F.mse_loss(sal_p * mask, sal * mask)

        qt_loss = F.cross_entropy(qt_l, qt)

        loss = ct_loss + pos_loss + et_loss + gr_loss + sal_loss + qt_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, examples, device, batch_size=64):
    model.eval()
    ct_correct = ct_total = 0
    pos_correct = pos_total = 0
    et_correct = et_total = 0
    gr_correct = gr_total = 0
    qt_correct = qt_total = 0
    sal_se = 0.0
    sal_n = 0

    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        ids, ct, pos, et, gr, sal, qt, mask = collate(batch, device)
        ct_l, pos_l, et_l, gr_l, sal_p, qt_l = model(ids)

        m = mask.float()
        ct_pred = ct_l.argmax(-1)
        ct_correct += ((ct_pred == ct) * mask).sum().item()
        ct_total += mask.sum().item()

        pos_pred = pos_l.argmax(-1)
        pos_correct += ((pos_pred == pos) * mask).sum().item()
        pos_total += mask.sum().item()

        et_pred = et_l.argmax(-1)
        et_correct += ((et_pred == et) * mask).sum().item()
        et_total += mask.sum().item()

        gr_pred = gr_l.argmax(-1)
        gr_correct += ((gr_pred == gr) * mask).sum().item()
        gr_total += mask.sum().item()

        qt_pred = qt_l.argmax(-1)
        qt_correct += (qt_pred == qt).sum().item()
        qt_total += qt.shape[0]

        sal_se += ((sal_p - sal) ** 2 * m).sum().item()
        sal_n += mask.sum().item()

    return {
        "ct_acc": ct_correct / max(ct_total, 1),
        "pos_acc": pos_correct / max(pos_total, 1),
        "et_acc": et_correct / max(et_total, 1),
        "gr_acc": gr_correct / max(gr_total, 1),
        "qt_acc": qt_correct / max(qt_total, 1),
        "sal_rmse": (sal_se / max(sal_n, 1)) ** 0.5,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gt_dir = Path(__file__).parent / "ground_truth"
    model_path = Path(__file__).parent / "tagger_model.pt"

    print("Loading ground truth...")
    entries = load_ground_truth(gt_dir)
    print(f"  loaded {len(entries)} annotations")

    print("Aligning to BPE...")
    examples = prepare_dataset(entries, tokenizer)
    print(f"  usable examples: {len(examples)}")

    if len(examples) < 20:
        print("ERROR: too few valid examples to train. Check ground truth.")
        return

    # Train/val split
    random.seed(42)
    random.shuffle(examples)
    split = int(len(examples) * 0.8)
    train_ex = examples[:split]
    val_ex = examples[split:]
    print(f"  train: {len(train_ex)}, val: {len(val_ex)}")

    # Model
    model = PromptTagger(tokenizer.vocab_size, embed_dim=256,
                         hidden_dim=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  tagger params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    best_val_loss = float("inf")
    patience = 0
    max_patience = 25

    print("\nTraining...")
    for epoch in range(200):
        train_loss = train_epoch(model, train_ex, device, optimizer)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            metrics = evaluate(model, val_ex, device)
            val_loss = train_loss  # approximate
            scheduler.step(val_loss)

            print(f"  epoch {epoch+1:3d}  loss={train_loss:.4f}  "
                  f"ct={metrics['ct_acc']:.0%}  "
                  f"pos={metrics['pos_acc']:.0%}  "
                  f"et={metrics['et_acc']:.0%}  "
                  f"gr={metrics['gr_acc']:.0%}  "
                  f"qt={metrics['qt_acc']:.0%}  "
                  f"sal_rmse={metrics['sal_rmse']:.3f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "epoch": epoch + 1,
                }, model_path)
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"  early stopping at epoch {epoch+1}")
                    break

    # Final eval
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = evaluate(model, val_ex, device)
    print(f"\nFinal val metrics:")
    print(f"  content_type accuracy : {final['ct_acc']:.1%}")
    print(f"  POS accuracy          : {final['pos_acc']:.1%}")
    print(f"  entity_type accuracy  : {final['et_acc']:.1%}")
    print(f"  gram_role accuracy    : {final['gr_acc']:.1%}")
    print(f"  query_type accuracy   : {final['qt_acc']:.1%}")
    print(f"  salience RMSE         : {final['sal_rmse']:.3f}")

    # Success criteria
    ok_ct = final["ct_acc"] > 0.85
    ok_et = final["et_acc"] > 0.80
    ok_gr = final["gr_acc"] > 0.75
    ok_qt = final["qt_acc"] > 0.90
    all_ok = ok_ct and ok_et and ok_gr and ok_qt
    print(f"\nSuccess criteria:")
    print(f"  content_type > 85%    : {'PASS' if ok_ct else 'FAIL'}")
    print(f"  entity_type > 80%     : {'PASS' if ok_et else 'FAIL'}")
    print(f"  gram_role > 75%       : {'PASS' if ok_gr else 'FAIL'}")
    print(f"  query_type > 90%      : {'PASS' if ok_qt else 'FAIL'}")
    print(f"  overall               : {'PASS' if all_ok else 'FAIL'}")

    print(f"\nTagger saved to {model_path}")

    # Save results JSON
    results = {"final_metrics": final, "epoch": ckpt["epoch"],
               "n_train": len(train_ex), "n_val": len(val_ex),
               "n_params": n_params, "pass": all_ok}
    with open(model_path.parent / "tagger_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
