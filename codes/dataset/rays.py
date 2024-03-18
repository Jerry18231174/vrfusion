import torch

from dataclasses import dataclass
from typing import Optional, Callable, Dict


def t_offset_to_delta(t_offsets : torch.Tensor, clips : torch.Tensor) -> torch.Tensor:
    """
    Convert t-space offsets to deltas.
    Args:
        t_offsets -> torch.Tensor : sample offsets in t-space [ray_num, num_samples]
        clips -> torch.Tensor : near/far clipping plane [ray_num, 2]
    Returns:
        -> torch.Tensor : deltas [ray_num, num_samples]
    """
    # Calculate deltas (\delta = (t_{i+1} - t_{i-1}) / 2) [ray_num, num_samples]
    # from 1 to last, far
    upper_bounds = torch.cat([t_offsets[:, 1:], clips[:, 1:2]], dim=-1)
    # from near, 0, to last-1
    lower_bounds = torch.cat([clips[:, 0:1], t_offsets[:, :-1]], dim=-1)
    # from 0, 1, to last
    deltas = (upper_bounds - lower_bounds) * 0.5

    # # Calculate deltas (\delta = t_{i+1} - t_{i}) [ray_num, num_samples]
    # # from 0 to last, far
    # upper_bounds = torch.cat([t_offsets[:, 1:], clips[:, 1:2]], dim=-1)
    # # from 0, 1, to last
    # deltas = upper_bounds - t_offsets

    return deltas

def s_offset_to_delta(s_offsets : torch.Tensor) -> torch.Tensor:
    """
    Convert s-space offsets to deltas.
    Args:
        s_offsets -> torch.Tensor : sample offsets in s-space [ray_num, num_samples]
    Returns:
        -> torch.Tensor : deltas [ray_num, num_samples]
    """
    # Calculate deltas (\delta = s_{i+1} - s_{i}) [ray_num, num_samples]
    # from 0 to last, far
    upper_bounds = torch.cat([s_offsets[:, 1:], torch.ones_like(s_offsets[:, 1:2])], dim=-1)
    # from 0, 1, to last
    deltas = upper_bounds - s_offsets
    return deltas


@dataclass
class Points:
    """
    Base points class.
    """
    positions: torch.Tensor = None
    """Point positions in world space. [ray_num, samples_per_ray, 3]"""
    directions: torch.Tensor = None
    """Ray directions in world space. [ray_num, samples_per_ray, 3]"""
    normals: Optional[torch.Tensor] = None
    """Point normals in world space. [ray_num, samples_per_ray, 3]"""
    rgb: Optional[torch.Tensor] = None
    """Point colors."""


@dataclass
class RaySamples(Points):
    """
    Ray samples class.
    """
    samples_per_ray: int = 0
    """Samples per ray."""
    offsets: torch.Tensor = None
    """Sample offsets on the rays. (t values) [ray_num, num_samples]"""
    deltas: Optional[torch.Tensor] = None
    """Deltas intervals of the sampled points. (\delta values) [ray_num, num_samples]"""
    densities: Optional[torch.Tensor] = None
    """Densities of the sampled points. [ray_num, num_samples]"""
    s2t_fn: Optional[Callable] = None
    """s-space to t-space function. (s -> t)"""
    s_offsets: Optional[torch.Tensor] = None
    """Sample offsets on the rays. (s values) [ray_num, num_samples]"""
    s_deltas: Optional[torch.Tensor] = None
    """Deltas intervals of the sampled points. (\delta values) [ray_num, num_samples]"""

    def to(self, device) -> None:
        """
        Move the ray samples to the given device.
        Args:
            device -> torch.device : device
        """
        self.positions = self.positions.to(device)
        self.directions = self.directions.to(device)
        if self.normals is not None:
            self.normals = self.normals.to(device)
        if self.rgb is not None:
            self.rgb = self.rgb.to(device)
        if self.offsets is not None:
            self.offsets = self.offsets.to(device)
        if self.deltas is not None:
            self.deltas = self.deltas.to(device)
        if self.densities is not None:
            self.densities = self.densities.to(device)
        if self.s_offsets is not None:
            self.s_offsets = self.s_offsets.to(device)
        if self.s_deltas is not None:
            self.s_deltas = self.s_deltas.to(device)

    def get_weights(self) -> torch.Tensor:
        """
        Get weights for the sampled points.
        Returns:
            -> torch.Tensor : weights for the sampled points [ray_num, num_samples]
        """
        assert self.densities is not None, "Densities are not available for the samples."
        assert self.deltas is not None, "Deltas are not available for the samples."
        # Calculate weights
        delta_density = self.densities * self.deltas
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[:, :-1], dim=-1)
        transmittance = torch.cat(
            [torch.zeros(self.offsets.shape[0], 1).to(self.offsets.device), transmittance], dim=-1
        )
        transmittance = torch.exp(-transmittance)

        weights = alphas * transmittance
        weights = torch.nan_to_num(weights)

        return weights
    
    def merge_with(self, virtual : Dict[str, torch.Tensor], clips) -> None:
        """
        Merge the ray samples with another ray samples.
        Args:
            virtual -> dict : virtual samples (depths, rgbs, density)
            clips -> torch.Tensor : near/far clipping plane [ray_num, 2]
        """
        # Insert virtual samples according to the depth
        v_offset = virtual["depth"].to(self.offsets.device)
        v_rgb = virtual["rgb"].to(self.rgb.device)
        v_positions = self.positions[:, 0:1, :] + self.directions[:, 0:1, :] * v_offset[:, :, None]
        v_directions = -self.directions[:, 0:1, :].repeat(1, v_offset.shape[1], 1)
        # Append virtual samples to the ray samples then sort
        assert self.offsets.shape[0] == v_offset.shape[0], "Invalid virtual samples."
        # print(self.offsets[0:5], v_offset[0:5])
        # exit()
        merged_offsets = torch.cat([self.offsets, v_offset], dim=-1)
        merged_rgb = torch.cat([self.rgb, v_rgb], dim=-2)
        merged_densities = torch.cat([
            self.densities,
            torch.ones_like(v_offset) * virtual["density"]
        ], dim=-1)
        merged_positions = torch.cat([self.positions, v_positions], dim=-2)
        merged_directions = torch.cat([self.directions, v_directions], dim=-2)
        # Sort the merged samples
        sorted_indices = torch.argsort(merged_offsets, dim=-1)
        self.offsets = torch.gather(merged_offsets, 1, sorted_indices)
        invalid = (self.offsets > clips[:, 1:2]) | (self.offsets < clips[:, 0:1])
        self.rgb = torch.gather(merged_rgb, 1, sorted_indices[:, :, None].repeat(1, 1, 3))
        self.densities = torch.gather(merged_densities, 1, sorted_indices)
        self.densities[invalid] = 0.0
        self.positions = torch.gather(merged_positions, 1, sorted_indices[:, :, None].repeat(1, 1, 3))
        self.directions = torch.gather(merged_directions, 1, sorted_indices[:, :, None].repeat(1, 1, 3))
        self.offsets = torch.clip(self.offsets, clips[:, 0:1], clips[:, 1:2])
        self.deltas = t_offset_to_delta(self.offsets, clips)


@dataclass
class RayBundle:
    """
    Ray bundle class.
    """
    ray_num: int
    """Ray number."""
    positions: torch.Tensor = None
    """Ray positions in world space.  [ray_num, 3]"""
    directions: torch.Tensor = None
    """Ray directions in world space. [ray_num, 3]"""
    clips: Optional[torch.Tensor] = None
    """Near/far clipping plane.       [ray_num, 2]"""
    rgb: Optional[torch.Tensor] = None
    """Ground truth RGB values from images. [ray_num, 3]"""

    def to(self, device) -> None:
        """
        Move the ray bundle to the given device.
        Args:
            device -> torch.device : device
        """
        self.positions = self.positions.to(device)
        self.directions = self.directions.to(device)
        if self.clips is not None:
            self.clips = self.clips.to(device)
        if self.rgb is not None:
            self.rgb = self.rgb.to(device)
    
    def slice(self, start_idx, end_idx):
        """
        Slice the ray bundle.
        Args:
            start_idx -> int : start index
            end_idx -> int : end index
        Returns:
            -> RayBundle : sliced ray bundle
        """
        assert start_idx < end_idx, "Invalid indices."
        assert end_idx <= self.ray_num, "End index is out of range."
        sliced_bundle = RayBundle(end_idx - start_idx)
        sliced_bundle.positions = self.positions[start_idx:end_idx]
        sliced_bundle.directions = self.directions[start_idx:end_idx]
        if self.clips is not None:
            sliced_bundle.clips = self.clips[start_idx:end_idx]
        if self.rgb is not None:
            sliced_bundle.rgb = self.rgb[start_idx:end_idx]
        return sliced_bundle

    def get_samples_from_offsets(self, t_offsets : torch.Tensor, s_offsets : torch.Tensor,
                                 s2t_fn : Callable = None) -> RaySamples:
        """
        Get samples from offsets.
        Args:
            t_offsets -> torch.Tensor : sample offsets in t-space [ray_num, num_samples]
            s_offsets -> torch.Tensor : sample offsets in s-space [ray_num, num_samples]
        Returns:
            -> RaySamples : ray samples
        """
        # Sample positions [ray_num, num_samples, 3]
        sampled_positions = self.positions[:, None, :] + \
                            self.directions[:, None, :] * t_offsets[:, :, None]
        # Sample directions [ray_num, num_samples, 3] (Inverted directions for sampling points)
        sampled_directions = -self.directions[:, None, :].repeat(1, t_offsets.shape[1], 1)

        # Calculate deltas [ray_num, num_samples]
        deltas = t_offset_to_delta(t_offsets, self.clips)

        # Calculate s-space deltas
        s_deltas = s_offset_to_delta(s_offsets)

        # Create ray samples
        ray_samples = RaySamples(samples_per_ray=t_offsets.shape[1])
        ray_samples.positions = sampled_positions
        ray_samples.directions = sampled_directions
        ray_samples.offsets = t_offsets
        ray_samples.deltas = deltas
        ray_samples.s_offsets = s_offsets
        ray_samples.s_deltas = s_deltas
        ray_samples.s2t_fn = s2t_fn

        return ray_samples