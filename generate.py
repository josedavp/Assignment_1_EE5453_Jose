


## review code

# generate.py
# Load a trained checkpoint and generate text with a prompt.
# Usage:
#   python generate.py --ckpt out/best_ckpt.pt --prompt "ROMEO:" --tokens 200 --device cuda
import argparse, torch
from pathlib import Path
from model import TinyTransformer, ModelConfig
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="out/best_ckpt.pt")
    parser.add_argument("--prompt", type=str, default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None, help="Optional output filename")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = ModelConfig(**ckpt["config"])
    model = TinyTransformer(config).to(args.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]

    def encode(s): return torch.tensor([[stoi[ch] for ch in s]], dtype=torch.long, device=args.device)
    def decode(idx): return ''.join(itos[i] for i in idx)

    idx = encode(args.prompt)
    out = model.generate(idx, max_new_tokens=args.tokens, temperature=args.temperature, top_k=args.top_k)
    text = decode(out[0].tolist())
    print(text)

    # Save to file next to checkpoint
    fname = args.out or f"sample_{int(datetime.now().timestamp())}.txt"
    out_path = Path(args.ckpt).parent / fname
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved sample to {out_path}")

if __name__ == "__main__":
    main()
