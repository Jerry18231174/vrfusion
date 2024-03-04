import torch
import torch.nn as nn
import tinycudann as tcnn

from abc import abstractmethod

from dataset.rays import RayBundle, RaySamples
from models.ray_samplers import ProposalSampler, UniformSampler


class NeRF(nn.Module):
    def __init__(self, config, scene_bbox) -> None:
        super().__init__()
        self.config = config
        self.scene_bbox = scene_bbox

    @abstractmethod
    def forward(self, ray_bundle) -> torch.Tensor:
        """Forward pass of the model.
        Args:
            ray_bundle -> torch.Tensor() : camera rays in the form of (positions, directions)
        Returns:
            -> torch.Tensor() : predicted RGB and density values for the given rays
        """
    
    def render_rgb(self, samples : RaySamples, ray_bundle : RayBundle) -> torch.Tensor:
        """Render RGB values for the sampled points.
        Args:
            samples -> torch.Tensor() : sampled points along the rays
            ray_bundle -> torch.Tensor() : camera rays in the form of (positions, directions)
        Returns:
            -> torch.Tensor() : rendered RGB values for the sampled points
        """
        weights = samples.get_weights()
        rgb = torch.sum(weights[:, :, None] * samples.rgb, dim=-2)
        assert rgb.shape == (ray_bundle.ray_num, 3), "Invalid RGB shape."
        return rgb
    
    def get_eval_rgb(self, ray_bundle, split_threshold=500000) -> dict:
        """Get the evaluation output.
        Args:
            ray_bundle -> torch.Tensor() : camera rays in the form of (positions, directions)
        Returns:
            -> dict : evaluation output dictionary
        """
        with torch.no_grad():
            if ray_bundle.ray_num <= split_threshold:
                return self(ray_bundle)
            split_num = (ray_bundle.ray_num + split_threshold - 1) // split_threshold
            outputs = []
            for i in range(split_num):
                if i == split_num - 1:
                    sub_ray_bundle = ray_bundle.slice(i * split_threshold, ray_bundle.ray_num)
                else:
                    sub_ray_bundle = ray_bundle.slice(i * split_threshold, (i + 1) * split_threshold)
                outputs.append(self(sub_ray_bundle)["rgb"])
            return torch.cat(outputs, dim=0)


class InstantNGP(NeRF):
    def __init__(self, config, scene_bbox) -> None:
        super().__init__(config, scene_bbox)
        # Main model
        default_grid_config = {
            "otype": "HashGrid",
            "n_dims_to_encode": 3,
            "n_levels": 12,
            "n_features_per_level": 2,
            "base_resolution": 16,
            "per_level_scale": 2,
            "interpolation": "Linear",
        }
        default_direction_config = {
            "otype": "SphericalHarmonics",
            "n_dim_to_encode": 3,
            "degree": 3,
        }
        default_network_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        }
        encoding_config = {"otype": "Composite", "nested": []}
        encoding_config["nested"].append(default_grid_config if "grid" not in config else config["grid"])
        encoding_config["nested"].append(default_direction_config if "direction" not in config else config["direction"])

        n_input_dims = 3 + 3  # position + direction
        n_output_dims = 3 + 1  # RGB + density
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims,
            n_output_dims,
            encoding_config,
            default_network_config if "network" not in config else config["network"]
        )

        # Proposal model
        default_proposal_config = {
            "grid": {
                "otype": "HashGrid",
                "n_levels": 5,
                "n_features_per_level": 1,
                "base_resolution": 8,
                "per_level_scale": 2,
                "interpolation": "Linear",
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 2,
            }
        }
        if "proposal" not in config:
            proposal_grid_config = default_proposal_config["grid"]
            proposal_network_config = default_proposal_config["network"]
        else:
            proposal_grid_config = config["proposal"]["grid"]
            proposal_network_config = config["proposal"]["network"]
        
        self.proposal_model = tcnn.NetworkWithInputEncoding(
            3, 1,
            proposal_grid_config,
            proposal_network_config
        )
        # Sampler
        self.sampler = ProposalSampler(
            num_samples=config["num_samples"] if "num_samples" in config else 32,
            num_proposal_samples=config["num_proposal_samples"] if "num_proposal_samples" in config else 64,
            num_proposal_iterations=config["num_proposal_iterations"] if "num_proposal_iterations" in config else 2
        )
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.exp = lambda x: torch.min(torch.exp(x), torch.relu(x) + 1e6)
        # Loss coefficients
        self.loss_config = {
            "proposal_loss": config["proposal_loss"] if "proposal_loss" in config else 1.0,
            "distance_loss": config["distance_loss"] if "distance_loss" in config else 1.0,
        }
    
    def forward(self, ray_bundle):
        output = {}
        # Sample points along the rays
        # samples, weights_list, proposal_samples_list = \
        #     self.sampler.generate_ray_samples(ray_bundle, self.proposal_model)
        uniform_sampler = UniformSampler(num_samples=128)
        samples = uniform_sampler.generate_ray_samples(ray_bundle)
        # Transform samples to the model's input space
        model_positions = (samples.positions - self.scene_bbox[0]) / (self.scene_bbox[1] - self.scene_bbox[0])
        # Predict density and RGB values
        x = torch.cat([model_positions, samples.directions], dim=-1).view(-1, 6)
        x = self.model(x).to(torch.float32)
        x = x.view(-1, samples.samples_per_ray, 4)  # [ray_num, num_samples, 4]
        density = self.exp(x[:, :, 0])
        point_rgb = self.sigmoid(x[:, :, 1:])
        # Accumulate density and RGB values
        samples.rgb = point_rgb
        samples.densities = density
        output["rgb"] = self.render_rgb(samples, ray_bundle)
        # Add proposal loss
        # output["prop_loss"] = self.proposal_loss(samples, weights_list[-1]) * self.loss_config["proposal_loss"]
        # Add distance loss
        output["dist_loss"] = self.distance_loss(samples) * self.loss_config["distance_loss"]
        return output
    
    def proposal_loss(self, samples, proposal_weights : torch.Tensor) -> torch.Tensor:
        """
        Proposal sampling loss.
        Histogram loss between proposal weights and model weights.
        Args:
            samples -> torch.Tensor() : sampled points along the rays
            proposal_weights -> torch.Tensor() : proposal weights
        Returns:
            -> torch.Tensor() : histogram loss
        """
        deltas = samples.deltas
        model_weights = samples.get_weights()
        return torch.mean(torch.abs(proposal_weights - model_weights) * deltas)
    
    def distance_loss(self, samples : RaySamples) -> torch.Tensor:
        """
        Distance loss.
        Minimize the distance between histograms and size of intervals
        Args:
            samples -> torch.Tensor() : sampled points along the rays
        Returns:
            -> torch.Tensor() : distance loss
        """
        weights = samples.get_weights()
        # Intersection terms
        i, j = torch.meshgrid(torch.arange(weights.shape[1]), torch.arange(weights.shape[1]))
        ray_idx = torch.arange(weights.shape[0])[:, None, None].repeat(1, weights.shape[1], weights.shape[1])
        sample_idx_row = i[None, :, :].repeat(weights.shape[0], 1, 1)
        sample_idx_col = j[None, :, :].repeat(weights.shape[0], 1, 1)
        idx_row = torch.cat((ray_idx[:, :, :, None], sample_idx_row[:, :, :, None]), dim=-1)
        idx_col = torch.cat((ray_idx[:, :, :, None], sample_idx_col[:, :, :, None]), dim=-1)
        w_i = weights[idx_row[:, :, :, 0], idx_row[:, :, :, 1]]
        w_j = weights[idx_col[:, :, :, 0], idx_col[:, :, :, 1]]
        t_i = samples.offsets[idx_row[:, :, :, 0], idx_row[:, :, :, 1]]
        t_j = samples.offsets[idx_col[:, :, :, 0], idx_col[:, :, :, 1]]
        intersection = torch.sum(w_i * w_j * torch.abs(t_j - t_i), dim=(1, 2))
        # Self-intersection terms
        self_intersection = torch.sum(weights * weights * samples.deltas, dim=-1) / 3
        return torch.mean((intersection + self_intersection) / (weights.shape[1] ** 2))


def load_model(config, scene_bbox) -> NeRF:
    """
    Load the model.
    Args:
        config -> dict : configuration dictionary
    Returns:
        -> NeRF : NeRF model
    """
    model_dict = {
        "instant-ngp": InstantNGP,
    }
    return model_dict[config["type"]](config, scene_bbox)