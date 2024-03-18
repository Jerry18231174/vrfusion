import torch
from torch import nn
import numpy as np

from abc import abstractmethod
from typing import Optional, Callable, Tuple, List

from dataset.rays import RayBundle, RaySamples


class Sampler(nn.Module):
    """
    Base sampler class.
    Args:
        num_samples -> int : number of samples
    """
    def __init__(self, num_samples) -> None:
        super().__init__()
        self.num_samples = num_samples

    @abstractmethod
    def generate_ray_samples(self, ray_bundle : RayBundle) -> RaySamples:
        """
        Generate ray samples.
        """
    
    def forward(self, ray_bundle) -> RaySamples:
        return self.generate_ray_samples(ray_bundle)


class SpacedSampler(Sampler):
    """
    Sample points according to a function.
    Args:
        num_samples: Number of samples per ray
        spacing_fn: Function that dictates sample spacing (ie `lambda x : x` is uniform).
        spacing_fn_inv: The inverse of spacing_fn.
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        spacing_fn: Callable,
        spacing_fn_inv: Callable,
        num_samples: int = 32,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(num_samples=num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def generate_ray_samples(
        self,
        ray_bundle: RayBundle,
    ) -> RaySamples:
        """
        Generates position samples according to spacing function.
        Args:
            ray_bundle: Rays to generate samples for
            num_samples: Number of samples per ray

        Returns:
            Positions and deltas for samples along a ray
        """
        assert ray_bundle is not None
        assert ray_bundle.clips is not None

        num_samples = self.num_samples
        assert num_samples is not None
        num_rays = ray_bundle.ray_num

        offsets = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.positions.device)[None, ...]  # [1, num_samples+1]
        bin_lower = offsets[..., :-1]
        bin_upper = offsets[..., 1:]
        
        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=offsets.dtype, device=offsets.device)
            else:
                t_rand = torch.rand((num_rays, num_samples), dtype=offsets.dtype, device=offsets.device)
            offsets = bin_lower + (bin_upper - bin_lower) * t_rand
        else:
            offsets = (bin_upper + bin_lower) * 0.5
            offsets = offsets.repeat(num_rays, 1)

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.clips[:, 0:1], ray_bundle.clips[:, 1:2]))

        def spacing_to_euclidean_fn(x):
            return self.spacing_fn_inv(x * s_far + (1 - x) * s_near)

        s_offsets = offsets
        t_offsets = spacing_to_euclidean_fn(offsets)  # [num_rays, num_samples]

        ray_samples = ray_bundle.get_samples_from_offsets(t_offsets, s_offsets, s2t_fn=spacing_to_euclidean_fn)

        return ray_samples


class UniformSampler(SpacedSampler):
    """
    Sample uniformly along a ray
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispSampler(SpacedSampler):
    """
    Piecewise sampler along a ray that allocate the first half of the samples
    uniformly and the second half using linearly in disparity spacing.
    Args:
        num_samples: Number of samples per ray
        train_stratified: Use stratified sampling during training. Defaults to True
        single_jitter: Use a same random jitter for all samples along a ray. Defaults to False
    """
    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(Sampler):
    """
    Sampler that uses probability density function for sampling.
    """
    def __init__(self,
        num_samples : int = 32,
        train_stratified : bool = True,
        single_jitter : bool = False,
        include_orginal : bool = True) -> None:
        super().__init__(num_samples)
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.include_orginal = include_orginal
    
    def generate_ray_samples(self, ray_bundle : RayBundle, ray_samples : RaySamples,
                             weights : torch.Tensor, eps : float = 1e-5) -> RaySamples:
        """
        Inverse transform sampling using discrete distribution.
        Args:
            ray_bundle -> RayBundle : Rays to generate samples for
            ray_samples -> RaySamples : Positions of the distribution
            weights -> torch.Tensor : Weights of the distribution
            eps -> float : Small offset to prevent NaNs
        Returns:
            -> RaySamples : New ray samples of the discrete distribution
        """
        assert ray_bundle.ray_num == ray_samples.positions.shape[0], "Ray number mismatch."
        assert ray_samples.offsets.shape == weights.shape, "Samples and weights shape mismatch."
        num_rays = ray_bundle.ray_num
        num_samples_dist = ray_samples.samples_per_ray

        # Add small offset to rays with zero weight to prevent NaNs
        weights_sum = weights.sum(dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / num_samples_dist
        weights_sum += padding
        # Get CDF from density PDF
        cdf = weights.cumsum(dim=-1)
        cdf = cdf / cdf[:, -1:]
        # Adding zero to the beginning of the CDF
        cdf = torch.cat([torch.zeros(num_rays, 1).to(ray_bundle.positions.device), cdf], dim=-1)  # [ray_num, num_samples_dist+1]

        # Uniformly sample points from the CDF
        num_samples = self.num_samples
        u = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.positions.device)[None, ...]  # [1, num_samples+1]
        bin_lower = u[..., :-1]
        bin_upper = u[..., 1:]

        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=u.dtype, device=u.device)
            else:
                t_rand = torch.rand((num_rays, num_samples), dtype=u.dtype, device=u.device)
            u = bin_lower + (bin_upper - bin_lower) * t_rand
        else:
            u = (bin_upper + bin_lower) * 0.5
            u = u.repeat(num_rays, 1)
        u = u.contiguous()
        
        # Find the bin index for each sample
        indices = torch.searchsorted(cdf, u, side="right")
        idx_right = torch.clamp(indices, 0, num_samples_dist - 1)
        idx_left = torch.clamp(indices - 1, 0, num_samples_dist - 1)
        cdf_left = torch.gather(cdf, -1, idx_left)
        cdf_right = torch.gather(cdf, -1, idx_right)
        offsets_left = torch.gather(ray_samples.s_offsets, -1, idx_left)
        offsets_right = torch.gather(ray_samples.s_offsets, -1, idx_right)

        # Linearly interpolate the offsets
        t = torch.clip(torch.nan_to_num((u - cdf_left) / (cdf_right - cdf_left)), 0.0, 1.0)
        s_offsets = offsets_left + t * (offsets_right - offsets_left)

        if self.include_orginal:
            s_offsets, _ = torch.sort(torch.cat([s_offsets, ray_samples.s_offsets], dim=-1), dim=-1)
        
        # Stop gradients
        s_offsets = s_offsets.detach()

        t_offsets = ray_samples.s2t_fn(s_offsets)

        ray_samples = ray_bundle.get_samples_from_offsets(t_offsets, s_offsets, s2t_fn=ray_samples.s2t_fn)

        return ray_samples


class ProposalSampler(Sampler):
    """
    Sampler that uses a proposal network to generate samples.
    """
    def __init__(self, num_samples : int = 32,
        num_proposal_samples : int = 64,
        num_proposal_iterations : int = 2,
        update_every : int = 1,
        initial_sampler : Optional[Sampler] = None,
        pdf_sampler : Optional[PDFSampler] = None) -> None:
        super().__init__(num_samples)
        self.num_proposal_samples = num_proposal_samples
        self.num_proposal_iterations = num_proposal_iterations
        if initial_sampler is None:
            self.initial_sampler = UniformLinDispSampler(num_samples=num_proposal_samples)
        else:
            self.initial_sampler = initial_sampler
        if pdf_sampler is None:
            self.proposal_sampler = PDFSampler(num_samples=num_proposal_samples)
        else:
            self.proposal_sampler = pdf_sampler
        
        self.update_every = update_every
        self._step_count = 0
        self._step_since_update = 0
    
    def generate_ray_samples(self,
        ray_bundle : RayBundle,
        density_fn : Callable) -> Tuple[RaySamples, List, List]:
        """
        Generate samples according to PDF weight provided by the proposal network.
        Args:
            ray_bundle -> RayBundle : Rays to generate samples for
            density_fn -> Callable(positions -> torch.Tensor) : Density function
        Returns:
            -> RaySamples : New ray samples of the discrete distribution
            -> List : List of weights
            -> List : List of proposal ray samples
        """
        weights_list = []
        ray_samples_list = []
        ray_samples = None

        # Set proposal network update information
        update = False
        self._step_count += 1
        self._step_since_update += 1
        if self._step_since_update > self.update_every or self._step_count < 10:
            self._step_since_update = 0
            update = True

        # Iteratively refine the proposal samples
        for i in range(self.num_proposal_iterations + 1):
            use_prop_density = i < self.num_proposal_iterations
            # Sample points from the initial sampler or the proposal sampler
            if i == 0:  # Uniform sampling
                ray_samples = self.initial_sampler.generate_ray_samples(ray_bundle)
            elif use_prop_density:  # Proposal sampling
                self.proposal_sampler.num_samples = self.num_proposal_samples
                self.proposal_sampler.include_orginal = True
                ray_samples = self.proposal_sampler.generate_ray_samples(ray_bundle, ray_samples, weights)
            else:  # Final sampling
                self.proposal_sampler.num_samples = self.num_samples
                self.proposal_sampler.include_orginal = False
                ray_samples = self.proposal_sampler.generate_ray_samples(ray_bundle, ray_samples, weights)
            # Calculate weights from the density function
            # if use_prop_density:
            # Get weights from the density function
            flat_positions = ray_samples.positions.view(-1, 3)
            if update:
                flat_densities = density_fn(flat_positions)
            else:
                with torch.no_grad():
                    flat_densities = density_fn(flat_positions)
            ray_samples.densities = flat_densities.view(-1, ray_samples.samples_per_ray)
            weights = ray_samples.get_weights()
            weights_list.append(weights)
            ray_samples_list.append(ray_samples)
        
        assert ray_samples is not None
        return ray_samples, weights_list, ray_samples_list
            
