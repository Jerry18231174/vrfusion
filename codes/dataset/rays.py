import torch

from dataclasses import dataclass
from typing import Optional


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

    def get_samples_from_offsets(self, offsets : torch.Tensor,
                             euclidean_offset=True) -> RaySamples:
        """
        Get samples from offsets.
        Args:
            offsets -> torch.Tensor : sample offsets [ray_num, num_samples]
            euclidean_offset -> bool : if the offset is in euclidean space
        Returns:
            -> RaySamples : ray samples
        """
        if not euclidean_offset:
            euclidean_offsets = offsets * self.clips[:, 1] + (1 - offsets) * self.clips[:, 0]
        else:
            euclidean_offsets = offsets
        
        # Sample positions [ray_num, num_samples, 3]
        sampled_positions = self.positions[:, None, :] + \
                            self.directions[:, None, :] * euclidean_offsets[:, :, None]
        # Sample directions [ray_num, num_samples, 3] (Inverted directions for sampling points)
        sampled_directions = -self.directions[:, None, :].repeat(1, offsets.shape[1], 1)

        # Calculate deltas (\delta = (t_{i+2} - t_{i}) / 2)
        # from 1 to last, far
        upper_bounds = torch.cat([euclidean_offsets[:, 1:], self.clips[:, 1:2]], dim=-1)
        # from near, 0, to last-1
        lower_bounds = torch.cat([self.clips[:, 0:1], euclidean_offsets[:, :-1]], dim=-1)
        # from 0, 1, to last
        deltas = (upper_bounds - lower_bounds) * 0.5

        # Create ray samples
        ray_samples = RaySamples(samples_per_ray=offsets.shape[1])
        ray_samples.positions = sampled_positions
        ray_samples.directions = sampled_directions
        ray_samples.offsets = euclidean_offsets
        ray_samples.deltas = deltas

        return ray_samples