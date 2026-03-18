import argparse
import torch

from .gating_mlp import GatingMLP
from .gnn_similarity import SiameseGNN


def save_gating(path: str, d_in: int = 1536):
    model = GatingMLP(d_in=d_in)
    torch.save(model.state_dict(), path)


def save_gnn(path: str):
    model = SiameseGNN()
    torch.save(model.state_dict(), path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gating-out", type=str, default=None)
    p.add_argument("--gnn-out", type=str, default=None)
    p.add_argument("--d-in", type=int, default=1536)
    args = p.parse_args()
    if args.gating_out:
        save_gating(args.gating_out, d_in=args.d_in)
    if args.gnn_out:
        save_gnn(args.gnn_out)


if __name__ == "__main__":
    main()