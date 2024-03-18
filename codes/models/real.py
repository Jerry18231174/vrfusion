import torch
import torch.nn as nn
import tinycudann as tcnn

from abc import abstractmethod
from typing import Dict
from tqdm import tqdm

from dataset.rays import RayBundle, RaySamples
from models.ray_samplers import ProposalSampler, UniformSampler
from models.losses import histogram_loss, proposal_loss, distance_loss


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
    
    def render_rgb(self, samples : RaySamples, ray_bundle : RayBundle,
                   virtual : Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Render RGB values for the sampled points.
        Args:
            samples -> RaySamples : sampled points along the rays
            ray_bundle -> RayBundle : camera rays in the form of (positions, directions)
            virtual -> dict : virtual objects' (depths, rgbs, density)
        Returns:
            -> torch.Tensor() : rendered RGB values for the sampled points
        """
        if not self.training and virtual is not None:
            # Add virtual objects
            samples.merge_with(virtual, ray_bundle.clips)
        weights = samples.get_weights()
        rgb = torch.sum(weights[:, :, None] * samples.rgb, dim=-2)
        assert rgb.shape == (ray_bundle.ray_num, 3), "Invalid RGB shape."
        return rgb
    
    def get_eval_rgb(self, ray_bundle : RayBundle,
                     virtual : Dict[str, torch.Tensor] = None,
                     split_threshold=150000) -> dict:
        """Get the evaluation output.
        Args:
            ray_bundle : camera rays in the form of (positions, directions)
            virtual : dict : virtual objects' (depths, rgbs, density)
            split_threshold : int : split threshold for large ray bundles
        Returns:
            -> dict : evaluation output dictionary
        """
        with torch.no_grad():
            if ray_bundle.ray_num <= split_threshold:
                return self(ray_bundle)
            split_num = (ray_bundle.ray_num + split_threshold - 1) // split_threshold
            outputs = []
            if virtual is not None:
                for i in tqdm(range(split_num)):
                    if i == split_num - 1:
                        sub_ray_bundle = ray_bundle.slice(i * split_threshold, ray_bundle.ray_num)
                        sub_virtual = {
                            "depth": virtual["depth"][i * split_threshold: ray_bundle.ray_num],
                            "rgb": virtual["rgb"][i * split_threshold: ray_bundle.ray_num],
                            "density": virtual["density"]
                        }
                    else:
                        sub_ray_bundle = ray_bundle.slice(i * split_threshold, (i + 1) * split_threshold)
                        sub_virtual = {
                            "depth": virtual["depth"][i * split_threshold: (i + 1) * split_threshold],
                            "rgb": virtual["rgb"][i * split_threshold: (i + 1) * split_threshold],
                            "density": virtual["density"]
                        }
                    outputs.append(self(sub_ray_bundle, sub_virtual)["rgb"])
            else:
                for i in tqdm(range(split_num)):
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
            "n_levels": 16,
            "n_features_per_level": 2,
            "base_resolution": 16,
            "per_level_scale": 1.611,
            "log2_hashmap_size": 19,
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
                "n_dims_to_encode": 3,
                "n_levels": 7,
                "n_features_per_level": 1,
                "base_resolution": 16,
                "per_level_scale": 2,
                "log2_hashmap_size": 17,
                "interpolation": "Linear",
            },
            "network": {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 16,
                "n_hidden_layers": 1,
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
            num_proposal_samples=config["num_proposal_samples"] if "num_proposal_samples" in config else 256,
            num_proposal_iterations=config["num_proposal_iterations"] if "num_proposal_iterations" in config else 2
        )
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.exp = lambda x: torch.min(torch.exp(x), torch.relu(x) + 1e6)
        # Loss coefficients
        self.loss_config = {
            "proposal_loss": config["proposal_loss"] if "proposal_loss" in config else 1.0,
            "distance_loss": config["distance_loss"] if "distance_loss" in config else 0.01,
        }
    
    def forward(self, ray_bundle : RayBundle,
                virtual : Dict[str, torch.Tensor] = None) -> dict:
        """
        Forward pass of the model.
        Args:
            ray_bundle -> RayBundle : camera rays
            virtual -> dict : virtual objects' (depths, rgbs, density)
        """
        output = {}
        # Sample points along the rays
        samples, weights_list, proposal_samples_list = \
            self.sampler.generate_ray_samples(ray_bundle, lambda x: self.proposal_model(self.global_to_model_cube(x)))
        # uniform_sampler = UniformSampler(num_samples=128)
        # samples = uniform_sampler.generate_ray_samples(ray_bundle)
        # Transform samples to the model's input space
        model_positions = self.global_to_model_cube(samples.positions.view(-1, 3))
        # Predict density and RGB values
        x = torch.cat([model_positions, samples.directions.view(-1, 3)], dim=-1)
        x = self.model(x).to(torch.float32)
        x = x.view(-1, samples.samples_per_ray, 4)  # [ray_num, num_samples, 4]
        density = self.exp(x[:, :, 0])
        point_rgb = self.sigmoid(x[:, :, 1:])
        # Accumulate density and RGB values
        samples.rgb = point_rgb
        samples.densities = density
        # Render RGB values
        output["rgb"] = self.render_rgb(samples, ray_bundle, virtual=virtual)
        # Add losses
        if self.training:
            # Add proposal loss
            # output["prop_loss"] = proposal_loss(ray_bundle.clips, samples, weights_list, proposal_samples_list) \
            #                     * self.loss_config["proposal_loss"]
            output["prop_loss"] = histogram_loss(samples, weights_list[-1]) * self.loss_config["proposal_loss"]
            # Add distance loss
            # output["dist_loss"] = distance_loss(samples) * self.loss_config["distance_loss"]
        return output
    
    def global_to_model_cube(self, positions : torch.Tensor, use_data_bbox=False) -> torch.Tensor:
        """
        Transform global positions to model positions.
        Implemented for the cube scene.
        Args:
            positions -> torch.Tensor() : global positions [sample_num, 3]
        Returns:
            -> torch.Tensor() : model positions
        """
        if use_data_bbox:
            raise NotImplementedError("Data bbox is not implemented.")
        # Cube radius (inner bbox)
        a = torch.max(torch.abs(self.scene_bbox[0])).item()
        inner_pos = torch.abs(positions).max(dim=1).values <= a
        x_major = (torch.abs(positions).max(dim=1).indices == 0) & ~inner_pos
        y_major = (torch.abs(positions).max(dim=1).indices == 1) & ~inner_pos
        z_major = (torch.abs(positions).max(dim=1).indices == 2) & ~inner_pos
        # First transform inner bbox to [-1, 1] cube, whole scene to [-2, 2] cube
        model_pos = positions.clone()
        model_pos[inner_pos] /= a
        model_pos[x_major, 0] = torch.where(positions[x_major, 0] > 0, 2 - a / positions[x_major, 0],
                                            -2 - a / positions[x_major, 0])
        model_pos[x_major, 1] = (model_pos[x_major, 0] / positions[x_major, 0]) * positions[x_major, 1]
        model_pos[x_major, 2] = (model_pos[x_major, 0] / positions[x_major, 0]) * positions[x_major, 2]
        model_pos[y_major, 1] = torch.where(positions[y_major, 1] > 0, 2 - a / positions[y_major, 1],
                                            -2 - a / positions[y_major, 1])
        model_pos[y_major, 0] = (model_pos[y_major, 1] / positions[y_major, 1]) * positions[y_major, 0]
        model_pos[y_major, 2] = (model_pos[y_major, 1] / positions[y_major, 1]) * positions[y_major, 2]
        model_pos[z_major, 2] = torch.where(positions[z_major, 2] > 0, 2 - a / positions[z_major, 2],
                                            -2 - a / positions[z_major, 2])
        model_pos[z_major, 0] = (model_pos[z_major, 2] / positions[z_major, 2]) * positions[z_major, 0]
        model_pos[z_major, 1] = (model_pos[z_major, 2] / positions[z_major, 2]) * positions[z_major, 1]
        # Then transform the positions to the [0, 1] cube
        model_pos = (model_pos + 2) / 4
        return model_pos
    
    def global_to_model_sphere(self, positions : torch.Tensor, use_data_bbox=False) -> torch.Tensor:
        """
        Transform global positions to model positions.
        Implemented for the sphere scene.
        Args:
            positions -> torch.Tensor() : global positions [sample_num, 3]
        Returns:
            -> torch.Tensor() : model positions
        """
        if use_data_bbox:
            raise NotImplementedError("Data bbox is not implemented.")
        # Sphere radius
        a = torch.max(torch.abs(self.scene_bbox[0])).item()
        pos_norm = torch.norm(positions, dim=-1)
        inner_pos = pos_norm <= a
        # First transform inner sphere to r=1 sphere, whole scene to r=2 sphere
        model_pos = positions.clone()
        model_pos[inner_pos] /= a
        model_pos[~inner_pos] *= (2 - a / pos_norm[~inner_pos].view(-1, 1))
        return model_pos


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