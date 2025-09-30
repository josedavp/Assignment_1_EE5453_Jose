import os, math, csv, argparse, time, json, urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, RandomSampler

from model import TinyTransformer, ModelConfig

DEFAULT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi[ch] for ch in text]

def decode(indices, itos):
    return ''.join(itos[i] for i in indices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_url", type=str, default=DEFAULT_URL)
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embed", type=int, default=256)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download data
    raw_path = out_dir / "tinyshakespeare.txt"
    if not raw_path.exists():
        print(f"Downloading dataset to {raw_path} ...")
        urllib.request.urlretrieve(args.data_url, raw_path)

    text = raw_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    # Split train/val 90/10 as per assignment
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    # Datasets and loaders
    train_ds = CharDataset(train_data, args.block_size)
    val_ds = CharDataset(val_data, args.block_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=True)

    config = ModelConfig(vocab_size=len(stoi),
                         n_layer=args.n_layer,
                         n_head=args.n_head,
                         n_embed=args.n_embed,
                         block_size=args.block_size,
                         dropout=args.dropout)

    model = TinyTransformer(config).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    def evaluate(loader):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                _, loss = model(xb, yb)
                total += loss.item()
                count += 1
        model.train()
        return total / max(1, count)

    log_path = out_dir / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        train_loss = evaluate(train_loader)
        val_loss = evaluate(val_loader)

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.4f}", f"{val_loss:.4f}"])

        # save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "stoi": stoi,
                "itos": itos,
            }
            torch.save(ckpt, out_dir / "best_ckpt.pt")

        dt = time.time() - t0
        print(f"Epoch {epoch} | train {train_loss:.4f} | val {val_loss:.4f} | time {dt:.1f}s")

    print(f"Done. Best val loss: {best_val:.4f}")
    print(f"Logs saved to {log_path}, checkpoint to {out_dir/'best_ckpt.pt'}")

if __name__ == "__main__":
    main()