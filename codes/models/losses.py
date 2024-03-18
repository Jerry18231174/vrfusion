import torch
from typing import List

from dataset.rays import RaySamples


def outer(
    t0_starts: torch.Tensor,  # "*batch num_samples_0",
    t0_ends: torch.Tensor,    # "*batch num_samples_0",
    t1_starts: torch.Tensor,  # "*batch num_samples_1",
    t1_ends: torch.Tensor,    # "*batch num_samples_1",
    y1: torch.Tensor,         # "*batch num_samples_1",
) -> torch.Tensor :           # "*batch num_samples_0"
    """
    From NeRFStudio:
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/model_components/losses.py
    Faster version of
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb8744a5cb4820a4d968fb3bfbed777550f4a/src/model/mipnerf360/helper.py#L117
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/stepfun.py#L64

    Args:
        t0_starts: start of the interval edges
        t0_ends: end of the interval edges
        t1_starts: start of the interval edges
        t1_ends: end of the interval edges
        y1: weights
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)

    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), side="right") - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), side="right")
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    y0_outer = cy1_hi - cy1_lo

    return y0_outer

def histogram_loss(samples : RaySamples, proposal_weights : torch.Tensor) -> torch.Tensor:
    """
    DEPRECATED Proposal sampling loss.
    Histogram loss between proposal weights and model weights.
    Args:
        samples -> torch.Tensor() : sampled points along the rays
        proposal_weights -> torch.Tensor() : proposal weights
    Returns:
        -> torch.Tensor() : histogram loss
    """
    with torch.no_grad():
        deltas = samples.deltas
        model_weights = samples.get_weights()
    return torch.mean(torch.abs(proposal_weights - model_weights) * deltas)

def proposal_loss(clips : torch.Tensor, samples : RaySamples,
                  proposal_weights : List[torch.Tensor],
                  proposal_samples : List[RaySamples]) -> torch.Tensor:
    """
    Proposal sampling loss.
    Outer measure loss between proposal weights and model weights.
    Args:
        clips -> torch.Tensor() : near/far clipping planes
        samples -> torch.Tensor() : sampled points along the rays
        proposal_weights -> List[torch.Tensor()] : proposal weights
        proposal_samples -> List[torch.Tensor()] : proposal samples
    Returns:
        -> torch.Tensor() : histogram loss
    """
    loss = 0.0
    with torch.no_grad():
        model_weights = samples.get_weights()
        # Padding far plane bins
        model_offsets = samples.offsets
        # from near, 0, to last-1
        model_lowers = torch.cat([clips[:, 0:1], model_offsets[:, :-1]], dim=-1)
    for i in range(len(proposal_weights)):
        prop_weights = proposal_weights[i]
        prop_offsets = proposal_samples[i].offsets
        # from near, 0, to last-1
        prop_lowers = torch.cat([clips[:, 0:1], prop_offsets[:, :-1]], dim=-1)
        with torch.no_grad():
            w_outer = outer(prop_lowers, prop_offsets, model_lowers, model_offsets, model_weights)
        prop_loss = torch.clip(prop_weights - w_outer, min=0.0) ** 2 / (prop_weights + 1e-7)
        loss += torch.mean(prop_loss)
    return loss

def distance_loss(samples : RaySamples) -> torch.Tensor:
    """
    Distance loss.
    Minimize the distance between histograms and size of intervals
    Args:
        samples -> torch.Tensor() : sampled points along the rays
    Returns:
        -> torch.Tensor() : distance loss
    """
    weights = samples.get_weights()
    offsets = samples.s_offsets
    # Intersection terms
    d_offsets = torch.abs(offsets[..., :, None] - offsets[..., None, :])
    intersection = torch.sum(weights * torch.sum(weights[..., None, :] * d_offsets, dim=-1), dim=-1)
    # Intrasection terms
    intrasection = torch.sum(weights * weights * samples.s_deltas, dim=-1) / 3
    return torch.mean(intersection + intrasection) / (weights.shape[-1] ** 2)