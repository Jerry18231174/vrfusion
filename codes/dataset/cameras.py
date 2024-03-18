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
        assert len(self._camera_matrix.shape) == 3, \
            "Only camera batches can be rearranged, FPSCamera is not supported."
        self.camera_num = indices.shape[0]
        self._camera_matrix = self._camera_matrix[indices]
        self.positions = self.positions[indices]
        self.look_ats = self.look_ats[indices]
        self.ups = self.ups[indices]

    def get_scene_bbox(self) -> torch.Tensor:
        """
        Get the scene bounding box.
        Returns:
            -> torch.Tensor() : two corner points of the inner/outer bounding box [2, 2, 3]
        """
        assert len(self._camera_matrix.shape) == 3, \
            "Only camera batches can get bounding box, FPSCamera is not supported."
        # Inner bbox contains all camera positions
        inner_min_pos, _ = torch.min(self.positions, dim=0)
        inner_max_pos, _ = torch.max(self.positions, dim=0)
        inner_bbox = torch.stack((inner_min_pos, inner_max_pos), dim=0)
        # Outer bbox contains all camera positions and far plane points
        far_pos = self.positions + self.look_ats * self.clips[:, 1:]
        all_pos = torch.cat((self.positions, far_pos), dim=0)
        min_pos, _ = torch.min(all_pos, dim=0)
        max_pos, _ = torch.max(all_pos, dim=0)
        outer_bbox = torch.stack((min_pos, max_pos), dim=0)
        scene_bbox = torch.stack((inner_bbox, outer_bbox), dim=0)
        return scene_bbox
    
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
        if len(self._camera_matrix.shape) == 3:
            # Batched camera matrices
            positions = self.positions[cam_idx]
        elif len(self._camera_matrix.shape) == 2:
            # Single camera matrix
            positions = self.positions
        else:
            raise ValueError("Invalid camera matrix shape.")
        # Sample pixel coordinates (Non-negative integer coordinates)
        pixel_coords = torch.rand((ray_num, 2))
        pixel_coords[:, 0] = torch.floor(pixel_coords[:, 0] * self.height)
        pixel_coords[:, 1] = torch.floor(pixel_coords[:, 1] * self.width)
        pixel_coords = pixel_coords.to(torch.int).to(self._camera_matrix.device)
        # Get ray directions in world space
        ray_dirs, ray_clips = self._get_ray_directions_clips(pixel_coords, cam_idx)
        # Create ray bundle
        rays = RayBundle(ray_num)
        rays.positions = positions.repeat(ray_num, 1)
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
        if len(self._camera_matrix.shape) == 3:
            # Batched camera matrices
            positions = self.positions[cam_idx]
        elif len(self._camera_matrix.shape) == 2:
            # Single camera matrix
            positions = self.positions
        else:
            raise ValueError("Invalid camera matrix shape.")
        # Sample pixel coordinates (Non-negative integer coordinates)
        pix_coord_y, pix_coord_x = torch.meshgrid(
            torch.arange(self.height),
            torch.arange(self.width),
        )
        pixel_coords = torch.stack((pix_coord_y, pix_coord_x), dim=-1).reshape(-1, 2)
        pixel_coords = pixel_coords.to(torch.int).to(self._camera_matrix.device)
        # Get ray directions in world space
        ray_dirs, ray_clips = self._get_ray_directions_clips(pixel_coords, cam_idx)
        # Create ray bundle
        rays = RayBundle(self.width * self.height)
        rays.positions = positions.repeat(self.width * self.height, 1)
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
        if len(self._camera_matrix.shape) == 3:
            # Batched camera matrices
            clips = self.clips[cam_idx]
            cam_matrix = self._camera_matrix[cam_idx]
        elif len(self._camera_matrix.shape) == 2:
            # Single camera matrix
            clips = self.clips if self.clips is not None else torch.Tensor([0.1, 50.0]).to(pixel_coords.device)
            cam_matrix = self._camera_matrix
        else:
            raise ValueError("Invalid camera matrix shape.")
        # Get ray directions in camera space
        ray_dirs = torch.zeros((pixel_coords.shape[0], 3)).to(pixel_coords.device)
        ray_dirs[:, 0] = pixel_coords[:, 1] - ((self.width - 1) / 2)  # x (width)
        ray_dirs[:, 1] = -(pixel_coords[:, 0] - ((self.height - 1) / 2))  # y (height, flipped)
        ray_dirs[:, 2] = -self.focal_length  # z
        # Get near/far plane values
        ray_length = torch.norm(ray_dirs, dim=-1, keepdim=True)
        normed_ray_length = ray_length / self.focal_length
        ray_clips = clips.repeat(pixel_coords.shape[0], 1)
        ray_clips = ray_clips * normed_ray_length
        # Normalize ray directions
        ray_dirs = ray_dirs / ray_length
        # debug_ray = ray_dirs
        # Transform ray directions to world space
        ray_dirs = ray_dirs @ cam_matrix[:3, :3].T
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
    
    def yield_fps_camera(self, cam_idx):
        """
        Yield a first-person shooter camera.
        Args:
            cam_idx -> int : camera index of the initial state as FPS camera
        Returns:
            -> FPSCamera : first-person shooter camera
        """
        outer_bbox = self.get_scene_bbox()[1]
        speed = torch.norm(outer_bbox[1] - outer_bbox[0]) * 0.01
        fps_camera = FPSCamera(speed)
        fps_camera.setup(self._camera_matrix[cam_idx].clone(), {
            "width": self.width,
            "height": self.height,
            "focal_length": self.focal_length,
            "clips": self.clips[cam_idx].clone(),
        })
        return fps_camera


class FPSCamera(Cameras):
    """
    First-person shooter camera class.
    """
    def __init__(self, speed=None) -> None:
        super().__init__(camera_num=1)
        self.rights = torch.Tensor([1, 0, 0])
        self.speed = speed if speed is not None else 0.1
        self.horizon_mode = False
        self.theta = None
        self.phi = None
    
    def setup(self, camera_matrix : torch.Tensor, cam_attrs : Dict = {}):
        """
        Setup the cameras.
        Args:
            camera_matrix -> torch.Tensor() : camera matrix [camera_num, 4, 4]
        """
        self._camera_matrix = camera_matrix
        self.positions = camera_matrix[:3, 3]
        # look_at, up and right vectors
        self.look_ats = -camera_matrix[:3, 2]
        self.ups = camera_matrix[:3, 1]
        self.rights = camera_matrix[:3, 0]
        # theta and phi
        self.theta = torch.rad2deg(torch.acos(-self.look_ats[1]))
        self.phi = torch.rad2deg(torch.atan2(-self.look_ats[0], -self.look_ats[2]))

        if "width" in cam_attrs:
            self.width = int(cam_attrs["width"])
        if "height" in cam_attrs:
            self.height = int(cam_attrs["height"])
        if "focal_length" in cam_attrs:
            self.focal_length = float(cam_attrs["focal_length"])
        if "clips" in cam_attrs:
            self.clips = cam_attrs["clips"]
    
    def update_cam_matrix(self):
        """
        Update the camera matrix according to pos, look_at, up vectors.
        """
        self._camera_matrix[:3, 3] = self.positions
        self._camera_matrix[:3, 2] = -self.look_ats
        self._camera_matrix[:3, 1] = self.ups
        self._camera_matrix[:3, 0] = self.rights

    def get_camera_matrix(self, camera_type="opengl") -> torch.Tensor:
        """
        Get the camera matrix.
        Args:
            camera_type -> str : camera type (opengl, mitsuba)
        Returns:
            -> torch.Tensor() : camera matrix [4, 4]
        """
        if camera_type == "opengl":
            # OpenGL / NeRF / Blender camera matrix
            # Look at -z, Up y, Right x
            mat = torch.cat([self._camera_matrix,
                             torch.Tensor([[0, 0, 0, 1]]).to(self._camera_matrix.device)], dim=0)
        elif camera_type == "mitsuba":
            # Mitsuba / Pytorch3D camera matrix
            # Look at z, Up y, Right -x
            mat = torch.cat([-self._camera_matrix[:, 0:1],
                              self._camera_matrix[:, 1:2],
                             -self._camera_matrix[:, 2:3],
                              self._camera_matrix[:, 3:4]], dim=1)
            mat = torch.cat([mat, torch.Tensor([[0, 0, 0, 1]]).to(self._camera_matrix.device)], dim=0)
        else:
            raise ValueError("Invalid camera type:", camera_type)
        return mat
                              
    def move(self, idx, delta):
        """
        Move the camera.
        Args:
            idx -> int : index (0: forward, 1: up, 2: right)
            delta -> float : delta
        """
        if not self.horizon_mode:
            if idx == 0:
                self.positions += delta * self.look_ats * self.speed
            elif idx == 1:
                self.positions += delta * self.ups * self.speed
            elif idx == 2:
                self.positions += delta * self.rights * self.speed
            else:
                raise ValueError("Invalid camera move index:", idx)
        else:
            if idx == 0:
                # Move along look_at vector's projection on xz plane
                hori_look_at = self.look_ats.clone()
                hori_look_at[1] = 0
                self.positions += delta * (hori_look_at / torch.norm(hori_look_at)) * self.speed
            elif idx == 1:
                # Move along up y-axis
                self.positions += delta * torch.Tensor([0, 1, 0]).to(self.positions.device) * self.speed
            elif idx == 2:
                # Move along right vector's projection on xz plane
                hori_right = self.rights.clone()
                hori_right[1] = 0
                self.positions += delta * (hori_right / torch.norm(hori_right)) * self.speed
            else:
                raise ValueError("Invalid camera move index:", idx)
        self.update_cam_matrix()

    def rotate(self, xoffset, yoffset):
        """
        Rotate the camera.
        Args:
            xoffset -> float : x offset
            yoffset -> float : y offset
        """
        if not self.horizon_mode:
            theta = torch.deg2rad(torch.Tensor([yoffset * self.speed * 10]))
            phi = torch.deg2rad(torch.Tensor([xoffset * self.speed * 10]))
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)

            c2w_rot = self._camera_matrix[:3, :3]

            self.look_ats = torch.Tensor([sin_phi, sin_theta * cos_phi, -cos_theta * cos_phi]).to(self.look_ats.device) \
                            @ c2w_rot.T
            self.ups = torch.Tensor([0, cos_theta, sin_theta]).to(self.ups.device) @ c2w_rot.T
            self.rights = torch.Tensor([cos_phi, -sin_theta * sin_phi, cos_theta * sin_phi]).to(self.rights.device) \
                          @ c2w_rot.T
        else:
            self.theta = torch.clip(self.theta + yoffset, 1, 179)
            self.phi += xoffset

            cos_theta = torch.cos(torch.deg2rad(self.theta))
            sin_theta = torch.sin(torch.deg2rad(self.theta))
            cos_phi = torch.cos(torch.deg2rad(self.phi))
            sin_phi = torch.sin(torch.deg2rad(self.phi))

            self.look_ats = -torch.Tensor(
                [sin_theta * cos_phi, cos_theta, sin_theta * sin_phi]).to(self.look_ats.device)
            self.rights = torch.Tensor([sin_phi, 0, -cos_phi]).to(self.rights.device)
            self.ups = torch.Tensor(
                [-cos_theta * cos_phi, sin_theta, -cos_theta * sin_phi]).to(self.ups.device)
        
        self.update_cam_matrix()