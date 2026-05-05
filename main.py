"""
3008ICT Deep Learning – Assignment 2
Recurrent Neural Networks for IMDb Sentiment Analysis
-------------------------------------------------------
Implements: Vanilla RNN, LSTM, Bidirectional LSTM (BRNN)
Dataset   : IMDb (25 000 train / 25 000 test reviews)
Framework : PyTorch
"""

import re
import time
import random
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# PART 0 – Setup & Data Preprocessing
# ==============================================================================

# ── 0.1 Load the IMDb Dataset ──────────────────────────────────────────────────
dataset   = load_dataset("imdb")
train_data = dataset["train"]
test_data  = dataset["test"]

print(train_data[0]["text"][:200])
print("Label:", train_data[0]["label"])

# ── 0.2 Tokenisation & Vocabulary ──────────────────────────────────────────────
VOCAB_SIZE = 20_000   # keep only the 20 000 most common words
MAX_LEN    = 500      # truncate reviews to 500 tokens
PAD_IDX    = 0        # index used for padding shorter sequences
UNK_IDX    = 1        # index used for words outside the vocabulary


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens (letters + apostrophes only)."""
    text = text.lower()
    return re.findall(r"\b[a-z']+\b", text)


# Count word frequencies across the entire training set
counter = Counter()
for ex in train_data:
    counter.update(tokenize(ex["text"]))

# Build vocab: reserve index 0 for <pad> and index 1 for <unk>
vocab = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
vocab.update({w: i + 2 for i, (w, _) in
              enumerate(counter.most_common(VOCAB_SIZE))})
print(f"Vocabulary size (including special tokens): {len(vocab)}")


def encode(text: str) -> list[int]:
    """Convert raw text to a list of token indices, capped at MAX_LEN."""
    tokens = tokenize(text)[:MAX_LEN]
    return [vocab.get(t, UNK_IDX) for t in tokens]


# ── Dataset & DataLoader ────────────────────────────────────────────────────────
class IMDbDataset(Dataset):
    """Converts a HuggingFace IMDb split into (encoded_tensor, label_tensor) pairs."""

    def __init__(self, split: str):
        self.samples = [
            (
                torch.tensor(encode(ex["text"]), dtype=torch.long),
                torch.tensor(ex["label"],        dtype=torch.long),
            )
            for ex in dataset[split]
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch):
    """Pad sequences in a batch to the same length and record true lengths."""
    texts, labels = zip(*batch)
    # Store true lengths before padding (needed for pack_padded_sequence if used)
    lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
    # pad_sequence fills shorter sequences with PAD_IDX (0)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    return texts, torch.stack(labels), lengths


train_loader = DataLoader(
    IMDbDataset("train"), batch_size=64, shuffle=True,  collate_fn=collate
)
test_loader = DataLoader(
    IMDbDataset("test"),  batch_size=64, shuffle=False, collate_fn=collate
)


# ==============================================================================
# PART 1 – Vanilla RNN Sentiment Classifier
# ==============================================================================

class RNNClassifier(nn.Module):
    """
    Embed → Vanilla RNN → Classify

    Architecture:
        Embedding layer    : vocab_size × embed_dim
        nn.RNN             : processes the embedded sequence step-by-step
        Dropout            : applied to the final hidden state
        Linear classifier  : hidden_dim → n_classes
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, n_classes: int, dropout: float = 0.5):
        super().__init__()
        # Embedding maps token indices to dense vectors; PAD_IDX produces zero vector
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Single-layer RNN; batch_first=True expects input as (batch, seq, features)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout   = nn.Dropout(dropout)
        # Map the final hidden state to class logits
        self.fc        = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))          # (batch, seq_len, embed_dim)
        _, hidden = self.rnn(embedded)                       # hidden: (1, batch, hidden_dim)
        hidden = hidden.squeeze(0)                           # (batch, hidden_dim)
        return self.fc(self.dropout(hidden))                 # (batch, n_classes)


# ── Shared training utilities ──────────────────────────────────────────────────

def train_model(model, loader, optimizer, criterion, device):
    """
    One full training epoch.
    Returns: (average_loss, accuracy) over the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for texts, labels, _ in loader:
        # Move data to the same device as the model
        texts, labels = texts.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(texts)                                # forward pass
        loss   = criterion(logits, labels)
        loss.backward()

        # Gradient clipping stabilises RNN training (avoids exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * labels.size(0)          # accumulate weighted loss
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate_model(model, loader, criterion, device):
    """
    Evaluation pass (no gradient computation).
    Returns: (average_loss, accuracy).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():                                    # disable gradient tracking
        for texts, labels, _ in loader:
            texts, labels = texts.to(device), labels.to(device)
            logits  = model(texts)
            loss    = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


def run_training(model, train_loader, test_loader,
                 n_epochs: int = 5, lr: float = 1e-3) -> dict:
    """
    Full training run for n_epochs.
    Returns a history dict with lists of train/test loss and accuracy per epoch.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [],
               "test_loss":  [], "test_acc":  [],
               "epoch_time": []}

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_model(model, train_loader, optimizer, criterion, device)
        te_loss, te_acc = evaluate_model(model, test_loader, criterion, device)

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)
        history["epoch_time"].append(elapsed)

        print(
            f"Epoch {epoch}/{n_epochs} | "
            f"Train Loss: {tr_loss:.4f}, Train Acc: {tr_acc:.4f} | "
            f"Test  Loss: {te_loss:.4f}, Test  Acc: {te_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

    return history


# ── Initialise and train Vanilla RNN ───────────────────────────────────────────
model_rnn = RNNClassifier(
    vocab_size = len(vocab),
    embed_dim  = 128,
    hidden_dim = 256,
    n_classes  = 2,
)
print("\n── Vanilla RNN ──")
print(model_rnn)

history_rnn = run_training(model_rnn, train_loader, test_loader, n_epochs=5)
print(f"\nVanilla RNN – Final Test Accuracy: {history_rnn['test_acc'][-1]:.4f}")


# ==============================================================================
# PART 2 – LSTM Sentiment Classifier
# ==============================================================================

class LSTMClassifier(nn.Module):
    """
    Embed → Multi-layer LSTM → Classify

    Key difference from RNN:
        nn.LSTM returns (output, (hidden, cell)).
        We use only the final hidden state of the last layer for classification.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # dropout is applied between LSTM layers (only active when n_layers > 1)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=n_layers,
                                 batch_first=True,
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))           # (batch, seq, embed_dim)
        # hidden: (n_layers, batch, hidden_dim)  cell: same shape
        _, (hidden, _) = self.lstm(embedded)
        # Take the hidden state from the last layer only
        hidden = hidden[-1]                                  # (batch, hidden_dim)
        return self.fc(self.dropout(hidden))                 # (batch, n_classes)


# ── Initialise and train LSTM ──────────────────────────────────────────────────
model_lstm = LSTMClassifier(
    vocab_size = len(vocab),
    embed_dim  = 128,
    hidden_dim = 256,
    n_classes  = 2,
    n_layers   = 2,
)
print("\n── LSTM ──")
print(model_lstm)

history_lstm = run_training(model_lstm, train_loader, test_loader, n_epochs=5)
print(f"\nLSTM – Final Test Accuracy: {history_lstm['test_acc'][-1]:.4f}")


# ==============================================================================
# PART 3 – Bidirectional LSTM (BRNN)
# ==============================================================================

class BRNNClassifier(nn.Module):
    """
    Embed → Bidirectional LSTM → Classify

    Two changes from LSTMClassifier:
        1. bidirectional=True in nn.LSTM
        2. Linear head takes 2 × hidden_dim inputs (forward + backward concatenated)
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim,
                                 num_layers=n_layers,
                                 batch_first=True,
                                 bidirectional=True,               # key change #1
                                 dropout=dropout if n_layers > 1 else 0.0)
        self.dropout   = nn.Dropout(dropout)
        # hidden_dim * 2 because forward & backward states are concatenated
        self.fc        = nn.Linear(hidden_dim * 2, n_classes)      # key change #2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(x))                  # (batch, seq, embed)
        # For a 2-layer bidirectional LSTM, hidden has shape (4, batch, hidden_dim):
        #   indices 0,1 → layer 1 forward & backward
        #   indices 2,3 → layer 2 forward & backward  ← we use these
        _, (hidden, _) = self.lstm(embedded)
        # Concatenate the last-layer forward (index -2) and backward (index -1) states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)         # (batch, hidden*2)
        return self.fc(self.dropout(hidden))                        # (batch, n_classes)


# ── Initialise and train BRNN ──────────────────────────────────────────────────
model_brnn = BRNNClassifier(
    vocab_size = len(vocab),
    embed_dim  = 128,
    hidden_dim = 256,
    n_classes  = 2,
    n_layers   = 2,
)
print("\n── Bidirectional LSTM ──")
print(model_brnn)

history_brnn = run_training(model_brnn, train_loader, test_loader, n_epochs=5)
print(f"\nBRNN – Final Test Accuracy: {history_brnn['test_acc'][-1]:.4f}")


# ==============================================================================
# PART 4 – Experiments, Analysis and Report
# ==============================================================================

# ── Task 4.1 – Combined Loss Curves ────────────────────────────────────────────
epochs = range(1, 6)
plt.figure(figsize=(8, 5))
plt.plot(epochs, history_rnn["train_loss"],  label="Vanilla RNN",  marker="o")
plt.plot(epochs, history_lstm["train_loss"], label="LSTM",         marker="s")
plt.plot(epochs, history_brnn["train_loss"], label="BRNN (BiLSTM)",marker="^")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss vs Epoch — RNN vs LSTM vs BRNN")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()
print("Loss curve saved to loss_curves.png")


def count_parameters(model) -> int:
    """Return the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Head-to-head summary table ─────────────────────────────────────────────────
print("\n── Head-to-Head Summary ──")
print(f"{'Model':<20} {'Test Acc':>10} {'Parameters':>12} {'Avg Epoch Time':>16}")
print("-" * 62)
for name, model, history in [
    ("Vanilla RNN",   model_rnn,  history_rnn),
    ("LSTM",          model_lstm, history_lstm),
    ("BRNN (BiLSTM)", model_brnn, history_brnn),
]:
    acc        = history["test_acc"][-1]
    params     = count_parameters(model)
    avg_time   = sum(history["epoch_time"]) / len(history["epoch_time"])
    print(f"{name:<20} {acc:>10.4f} {params:>12,} {avg_time:>14.1f}s")


# ── Task 4.2 – Live Prediction ─────────────────────────────────────────────────

def predict_sentiment(text: str, model, vocab: dict, device) -> tuple[str, float]:
    """
    Predict sentiment for a raw English review string.

    Returns:
        label      : "Positive" or "Negative"
        confidence : probability of the predicted class (0–1)
    """
    model.eval()
    with torch.no_grad():
        # Encode the input text into a tensor and add a batch dimension
        indices = encode(text)
        tensor  = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
        logits  = model(tensor)                              # (1, 2)
        probs   = torch.softmax(logits, dim=1)              # convert logits to probabilities
        pred    = probs.argmax(dim=1).item()                # 0 = negative, 1 = positive
        conf    = probs[0, pred].item()

    label = "Positive" if pred == 1 else "Negative"
    return label, conf


# Test on four sample reviews (two positive, two ambiguous)
test_reviews = [
    "This film was absolutely wonderful. The acting was superb and I loved every minute.",
    "One of the best movies I have ever seen. Highly recommend to everyone!",
    "I'm not sure how I feel about this. Some parts were good, others not so much.",
    "It wasn't terrible but I wouldn't watch it again. The plot made no sense at times.",
]

print("\n── Live Predictions (best model: BRNN) ──")
for review in test_reviews:
    label, conf = predict_sentiment(review, model_brnn, vocab, device)
    print(f"  [{label} ({conf:.2%})] — {review[:80]}...")


# ── Task 4.3 – Error Analysis ──────────────────────────────────────────────────

def find_misclassified(model, loader, device, n_examples: int = 5) -> list[dict]:
    """
    Collect up to n_examples misclassified reviews from the test set.
    Returns a list of dicts with keys: tokens, true_label, pred_label.
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for texts, labels, _ in loader:
            texts, labels = texts.to(device), labels.to(device)
            logits = model(texts)
            preds  = logits.argmax(dim=1)

            # Find indices where prediction differs from true label
            wrong  = torch.where(preds != labels)[0]
            for idx in wrong:
                if len(errors) >= n_examples:
                    return errors
                errors.append({
                    "tokens":     texts[idx].cpu().tolist(),
                    "true_label": labels[idx].item(),
                    "pred_label": preds[idx].item(),
                })

    return errors


# Build reverse vocab to decode token indices back to words
idx_to_word = {idx: word for word, idx in vocab.items()}

misclassified = find_misclassified(model_brnn, test_loader, device, n_examples=5)

print("\n── Error Analysis (BRNN, 5 misclassified examples) ──")
label_map = {0: "Negative", 1: "Positive"}
for i, err in enumerate(misclassified, 1):
    # Decode the first 100 non-padding tokens back to readable text
    words = [idx_to_word.get(t, "<unk>") for t in err["tokens"]
             if t != PAD_IDX][:100]
    print(f"\nExample {i}:")
    print(f"  Text (first 100 tokens): {' '.join(words)}")
    print(f"  True label : {label_map[err['true_label']]}")
    print(f"  Predicted  : {label_map[err['pred_label']]}")

print("\nDone. See loss_curves.png for the training loss comparison plot.")
