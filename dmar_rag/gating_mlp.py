import torch
import torch.nn as nn


class GatingMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def load(self, path: str):
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)

    def weights(self, q: torch.Tensor, d: torch.Tensor) -> tuple:
        x = torch.cat([q, d], dim=-1).unsqueeze(0)
        w = self.forward(x)[0]
        return float(w[0].item()), float(w[1].item())