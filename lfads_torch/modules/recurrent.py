import torch
from torch import nn


def init_variance_scaling_(cell: nn.GRUCell, scale_dim: int = None):
    if scale_dim is None:
        ih_scale = cell.input_size
        hh_scale = cell.hidden_size
    else:
        ih_scale = scale_dim
        hh_scale = scale_dim
    nn.init.normal_(cell.weight_ih_l0, std=1 / torch.sqrt(ih_scale))
    nn.init.normal_(cell.weight_hh_l0, std=1 / torch.sqrt(hh_scale))
    # TODO: confirm that the gate weights come before candidate
    nn.init.ones_(cell.bias_ih_l0)
    cell.bias_ih_l0.data[-cell.hidden_size :] = 0.0
    # NOTE: these weights are not present in TF
    nn.init.zeros_(cell.bias_hh_l0)


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.clip_value = clip_value
        scale_dim = input_size + hidden_size if is_encoder else None
        init_variance_scaling_(self, scale_dim=scale_dim)

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        hidden = super().forward(input, hidden)
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class ClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.cell = ClippedGRUCell(
            input_size, hidden_size, clip_value=clip_value, is_encoder=True
        )

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class BidirectionalClippedGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
    ):
        super().__init__()
        self.fwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)
        self.bwd_gru = ClippedGRU(input_size, hidden_size, clip_value=clip_value)

    def forward(self, input: torch.Tensor, h_0: torch.Tensor):
        h0_fwd, h0_bwd = torch.split(h_0, 2)
        input_fwd = torch.transpose(input, 0, 1)
        input_bwd = torch.transpose(torch.flip(input, 1), 0, 1)
        output_fwd, hn_fwd = self.fwd_gru(input_fwd, h0_fwd)
        output_bwd, hn_bwd = self.bwd_gru(input_bwd, h0_bwd)
        output = torch.cat([output_fwd, output_bwd], dim=2)
        h_n = torch.stack([hn_fwd, hn_bwd], dim=0)
        return output, h_n
