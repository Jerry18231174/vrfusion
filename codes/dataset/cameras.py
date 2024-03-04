import torch

from typing import Dict, Tuple

from dataset.rays import RayBundle


class Cameras:
    """
    Cameras class.
    """
    def __init__(self, camera_num) -> None:
        super().__init__()
        self.camera_num = camera_num
        # Intrinsics
        self.width = 0
        self.height = 0
        self.focal_length = 0  # Focal length value's unit is pixel
        self.clips = None  # Near/far plane values' unit is meter, not pixel
        # Extrinsics
        self.positions = None
        self.look_ats = None
        self.ups = None
        # Camera matrix (to world space, [cam_num, 3, 4])
        self._camera_matrix = None

    def setup(self, camera_matrix : torch.Tensor, cam_attrs : Dict = {}):
        """
        Setup the cameras.
        Args:
            camera_matrix -> torch.Tensor() : camera matrix [camera_num, 4, 4]
        """
        self._camera_matrix = camera_matrix
        self.positions = camera_matrix[:, :3, 3]
        # TODO: Implement look_at and up vectors
        self.look_ats = torch.Tensor([0, 0, -1]).to(camera_matrix.device)
        self.look_ats = self.look_ats @ camera_matrix[:, :3, :3].transpose(1, 2)

        if "width" in cam_attrs:
            self.width = int(cam_attrs["width"])
        if "height" in cam_attrs:
            self.height = int(cam_attrs["height"])
        if "focal_length" in cam_attrs:
            self.focal_length = float(cam_attrs["focal_length"])
        if "clips" in cam_attrs:
            self.clips = cam_attrs["clips"]

    def rearrange(self, indices):
        """
        Rearrange camera matrices and extrinsics.
        Args:
            indices -> torch.Tensor() : camera indices
        """
        self.camera_num = indices.shape[0]
        self._camera_matrix = self._camera_matrix[indices]
        self.positions = self.positions[indices]
        self.look_ats = self.look_ats[indices]
        self.ups = self.ups[indices]

    def get_scene_bbox(self) -> torch.Tensor:
        """
        Get the scene bounding box.
        Returns:
            -> torch.Tensor() : two corner points of the bounding box
        """
        # Check if the far plane overtook the bounding box
        far_pos = self.positions + self.look_ats * self.clips[:, 1:]
        all_pos = torch.cat((self.positions, far_pos), dim=0)
        min_pos, _ = torch.min(all_pos, dim=0)
        max_pos, _ = torch.max(all_pos, dim=0)
        return torch.stack((min_pos[None, :], max_pos[None, :]), dim=0)
    
    def get_rays(self, cam_idx, ray_num) -> RayBundle:
        """
        Get rays for a given camera index.
        Args:
            cam_idx -> int : camera index
            ray_num -> int : number of rays to sample
        Returns:
            -> Rays : ray bundle
            -> torch.Tensor() : pixel coordinates [ray_num, 2] (w, h)
        """
        # Sample pixel coordinates (Non-negative integer coordinates)
        pixel_coords = torch.rand((ray_num, 2))
        pixel_coords[:, 0] = torch.floor(pixel_coords[:, 0] * self.height)
        pixel_coords[:, 1] = torch.floor(pixel_coords[:, 1] * self.width)
        pixel_coords = pixel_coords.to(torch.int)
        # Get ray directions in world space
        ray_dirs, ray_clips = self._get_ray_directions_clips(pixel_coords, cam_idx)
        # Create ray bundle
        rays = RayBundle(ray_num)
        rays.positions = self.positions[cam_idx].repeat(ray_num, 1)
        rays.directions = ray_dirs
        rays.clips = ray_clips
        # Leave RGB values for Dataset class
        return rays, pixel_coords
    
    def get_all_rays(self, cam_idx) -> RayBundle:
        """
        Get all rays for a given camera index.
        Args:
            cam_idx -> int : camera index
        Returns:
            -> Rays : ray bundle
            -> torch.Tensor() : pixel coordinates [pixel_num, 2] (w, h)
        """
        # Sample pixel coordinates (Non-negative integer coordinates)
        pix_coord_y, pix_coord_x = torch.meshgrid(
            torch.arange(self.height),
            torch.arange(self.width),
        )
        pixel_coords = torch.stack((pix_coord_y, pix_coord_x), dim=-1).reshape(-1, 2)
        pixel_coords = pixel_coords.to(torch.int)
        # Get ray directions in world space
        ray_dirs, ray_clips = self._get_ray_directions_clips(pixel_coords, cam_idx)
        # Create ray bundle
        rays = RayBundle(self.width * self.height)
        rays.positions = self.positions[cam_idx].repeat(self.width * self.height, 1)
        rays.directions = ray_dirs
        rays.clips = ray_clips
        # Leave RGB values for Dataset class
        return rays, pixel_coords
    
    def _get_ray_directions_clips(self, pixel_coords, cam_idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ray directions in world space and near/far plane values.
        Args:
            pixel_coords -> torch.Tensor() : pixel coordinates [pixel_num, 2]
            cam_idx -> int : camera index
        Returns:
            -> torch.Tensor() : ray directions in world space [pixel_num, 3]
            -> torch.Tensor() : near/far plane values [pixel_num, 2]
        """
        # Get ray directions in camera space
        ray_dirs = torch.zeros((pixel_coords.shape[0], 3)).to(pixel_coords.device)
        ray_dirs[:, 0] = pixel_coords[:, 1] - ((self.width - 1) / 2)  # x (width)
        ray_dirs[:, 1] = -(pixel_coords[:, 0] - ((self.height - 1) / 2))  # y (height, flipped)
        ray_dirs[:, 2] = -self.focal_length  # z
        # Get near/far plane values
        ray_length = torch.norm(ray_dirs, dim=-1, keepdim=True)
        normed_ray_length = ray_length / self.focal_length
        ray_clips = self.clips[cam_idx].repeat(pixel_coords.shape[0], 1)
        ray_clips = ray_clips * normed_ray_length
        # Normalize ray directions
        ray_dirs = ray_dirs / ray_length
        # debug_ray = ray_dirs
        # Transform ray directions to world space
        ray_dirs = ray_dirs @ self._camera_matrix[cam_idx, :3, :3].T
        # DEBUG: plot rays
        if False and pixel_coords.shape[0] == self.width * self.height:
            print(-(pixel_coords[:, 0] - ((self.height - 1) / 2)))
            print(pixel_coords[:, 1] - ((self.width - 1) / 2))
            print(self.height, self.width, self.focal_length)
            up = torch.Tensor([0, 1, 0]).to(ray_dirs.device)
            up = up @ self._camera_matrix[:20, :3, :3].transpose(1, 2)
            right = torch.Tensor([1, 0, 0]).to(ray_dirs.device)
            right = right @ self._camera_matrix[:20, :3, :3].transpose(1, 2)
            ray_dirs = debug_ray @ self._camera_matrix[:20, :3, :3].transpose(1, 2)
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d", aspect='equal')
            # Plot cameras
            ax.quiver(
                self.positions[:20, 0].cpu().numpy(), self.positions[:20, 1].cpu().numpy(), self.positions[:20, 2].cpu().numpy(),
                self.look_ats[:20, 0].cpu().numpy(), self.look_ats[:20, 1].cpu().numpy(), self.look_ats[:20, 2].cpu().numpy(),
            )
            # Plot rays
            first_ray = ray_dirs[:, :1, :].view(-1, 3).cpu().numpy()
            corner_ray = ray_dirs[:, 1295:1296, :].view(-1, 3).cpu().numpy()
            last_ray = ray_dirs[:, 1088639:1088640, :].view(-1, 3).cpu().numpy()
            ax.quiver(
                self.positions[:20, 0].cpu().numpy(), self.positions[:20, 1].cpu().numpy(), self.positions[:20, 2].cpu().numpy(),
                first_ray[:20, 0], first_ray[:20, 1], first_ray[:20, 2],
                color="r"
            )
            ax.quiver(
                self.positions[:20, 0].cpu().numpy(), self.positions[:20, 1].cpu().numpy(), self.positions[:20, 2].cpu().numpy(),
                last_ray[:20, 0], last_ray[:20, 1], last_ray[:20, 2],
                color="orange"
            )
            ax.quiver(
                self.positions[:20, 0].cpu().numpy(), self.positions[:20, 1].cpu().numpy(), self.positions[:20, 2].cpu().numpy(),
                corner_ray[:20, 0], corner_ray[:20, 1], corner_ray[:20, 2],
                color="purple"
            )
            ax.quiver(
                self.positions[:20, 0].cpu().numpy(), self.positions[:20, 1].cpu().numpy(), self.positions[:20, 2].cpu().numpy(),
                up[:20, 0].cpu().numpy(), up[:20, 1].cpu().numpy(), up[:20, 2].cpu().numpy(),
                color="g"
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_box_aspect([1, 1, 1])
            plt.show()
            exit()
        return ray_dirs, ray_clips
