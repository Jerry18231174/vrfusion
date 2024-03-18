import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

from abc import abstractmethod
from typing import List
import numpy.typing as npt

from dataset.cameras import Cameras
from dataset.rays import RayBundle


class NeRFDataset(Dataset):
    """
    NeRF Dataset class.
    """
    def __init__(self, split, config) -> None:
        super().__init__()
        self.dataset_path = os.path.join(os.getcwd(), "data", config["dataset"]["name"])
        self.rays_per_batch = config["dataset"]["rays_per_batch"]
        self.dataset_size = 0
        self.cameras : Cameras = None
        self.image_filenames : List[str] = []
        self.images : List[torch.Tensor] = []
        self.load_images = False  # Load images on setup if True, else load on the fly
        # Train/eval splitting
        self.split = split
        assert self.split in ["train", "eval"], f"Split {self.split} is not supported."
        self.shuffle_seed = 0
        self.split_ratio = 0.875
        if "shuffle_seed" in config["dataset"]:
            self.shuffle_seed = config["dataset"]["shuffle_seed"]
        if "split_ratio" in config["dataset"]:
            self.split_ratio = config["dataset"]["split_ratio"]
        self.shuffled_indices : npt.NDArray = None
        # Setup the dataset
        self.setup()
        # Setup scene bounding box
        self.scene_bbox = self.cameras.get_scene_bbox()

    @abstractmethod
    def setup(self):
        """
        Setup the dataset.
        """
    
    def print_info(self):
        """
        Print dataset information.
        """
        print("{:s} dataset size: {:d}".format(self.split, self.dataset_size))

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, cam_idx):
        if self.split == "train":
            return self.collate_train(cam_idx)
        elif self.split == "eval":
            return self.collate_eval(cam_idx)

    def collate_train(self, cam_idx) -> RayBundle:
        """
        Collate training data (Sampling rays of batch size).
        """
        if isinstance(cam_idx, list):
            assert len(cam_idx) == 1, "Only one camera index is allowed for training."
            cam_idx = cam_idx[0] % self.dataset_size
        ray_bundle, pixel_coords = self.cameras.get_rays(cam_idx, self.rays_per_batch)
        if self.images[cam_idx] is None:
            self.images[cam_idx] = get_numpy_image(self.image_filenames[cam_idx])
        img_tensor = torch.from_numpy(self.images[cam_idx].astype(np.float32) / 255.0)
        ray_bundle.rgb = img_tensor[pixel_coords[:, 0], pixel_coords[:, 1]]
        return ray_bundle
    
    def collate_eval(self, cam_idx) -> RayBundle:
        """
        Collate evaluation data (Sample all rays for a given camera index.)
        Args:
            cam_idx -> int : camera index
        Returns:
            -> dict : evaluation data dictionary
        """
        if isinstance(cam_idx, list):
            assert len(cam_idx) == 1, "Only one camera index is allowed for training."
            cam_idx = cam_idx[0]
        ray_bundle, pixel_coords = self.cameras.get_all_rays(cam_idx)
        if self.images[cam_idx] is None:
            self.images[cam_idx] = get_numpy_image(self.image_filenames[cam_idx])
        img_tensor = torch.from_numpy(self.images[cam_idx].astype(np.float32) / 255.0)
        ray_bundle.rgb = img_tensor[pixel_coords[:, 0], pixel_coords[:, 1]]
        return ray_bundle


class ColmapDataset(NeRFDataset):
    """
    Colmap dataset class. (LLFF style dataset)
    """
    def __init__(self, split, config, scale_ratio=4) -> None:
        self.scale_ratio = scale_ratio
        super().__init__(split, config)

    def setup(self):
        # Check if poses_bounds.npy exist
        poses_bounds_path = os.path.join(self.dataset_path, "poses_bounds.npy")
        if not os.path.exists(poses_bounds_path):
            raise FileNotFoundError(f"Poses bounds file not found: {poses_bounds_path}")
        # Load poses and bounds
        poses_bounds = np.load(poses_bounds_path)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)
        bounds = poses_bounds[:, 15:]
        self.dataset_size = poses_bounds.shape[0]
        # Generate shuffled indices
        indices = np.arange(self.dataset_size)
        np.random.seed(self.shuffle_seed)
        np.random.shuffle(indices)
        if self.split == "train":
            self.shuffled_indices = indices[:int(self.split_ratio * self.dataset_size)]
        elif self.split == "eval":
            self.shuffled_indices = indices[int(self.split_ratio * self.dataset_size):]
        self.dataset_size = self.shuffled_indices.shape[0]
        poses = poses[self.shuffled_indices]
        bounds = bounds[self.shuffled_indices]
        # Scale the bounds
        bounds[:, 0] = bounds[:, 0] * 0.9
        # bounds[:, 1] = bounds[:, 1] * 0.9
        # Extract camera attributes
        cam_attrs = {
            "height": poses[0, 0, 4] / self.scale_ratio,
            "width": poses[0, 1, 4] / self.scale_ratio,
            "focal_length": poses[0, 2, 4] / self.scale_ratio,
            "clips": torch.from_numpy(bounds).to(torch.float32)
        }
        poses = poses[:, :, :4]  # Discard the last column (H, W, F)
        # Transform coordinate from Colmap to NeRF
        # poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], axis=1)
        # Note: The transform above is rotation along z-axis in world space, maybe not necessary
        #       We need rotation along z-axis in camera space.
        #       i.e. camera matrix right multiplies a rotation matrix
        poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:]], axis=2)
        # Setup cameras
        self.cameras = Cameras(self.dataset_size)
        self.cameras.setup(torch.from_numpy(poses).to(torch.float32), cam_attrs)
        # Load image filenames, and (optionally) images
        if self.scale_ratio != 1:
            images_path = os.path.join(self.dataset_path, "images_{:d}".format(self.scale_ratio))
        else:
            images_path = os.path.join(self.dataset_path, "images")
        images_path_list = sorted(os.listdir(images_path))
        for index in self.shuffled_indices.tolist():
            item = images_path_list[index]
            if item.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_filenames.append(os.path.join(images_path, item))
                if self.load_images:
                    self.images.append(get_numpy_image(self.image_filenames[-1]))
                else:
                    self.images.append(None)
        # Print dataset information
        self.print_info()


class BlenderDataset(NeRFDataset):
    """
    Blender dataset class.
    """
    def __init__(self, split, config) -> None:
        super().__init__(split, config)

    def setup(self):
        # TODO: Implement setup method for BlenderDataset
        pass


def load_dataset(split, config) -> NeRFDataset:
    """
    Load the dataset.
    Args:
        split -> str : "train" / "eval"
        config -> dict : configuration dictionary
    Returns:
        -> NeRFDataset : NeRF dataset
    """
    dataset_dict = {
        "blender": BlenderDataset,
        "colmap": ColmapDataset,
    }
    if config["dataset"]["type"] not in dataset_dict:
        # Judge dataset type
        if os.path.exists(os.path.join(config["dataset"]["path"], "poses_bounds.npy")):
            config["dataset"]["type"] = "colmap"
        elif os.path.exists(os.path.join(config["dataset"]["path"], "transforms_train.json")) and \
             os.path.exists(os.path.join(config["dataset"]["path"], "transforms_test.json")):
            config["dataset"]["type"] = "blender"
    return dataset_dict[config["dataset"]["type"]](split, config)

def get_numpy_image(image_filename, scale_factor=1.0) -> npt.NDArray[np.uint8]:
    """
    Returns the image of shape (H, W, 3 or 4).
    Args:
        image_filename -> str : image filename
        scale_factor -> float : The scale factor for the image.
    """
    pil_image = Image.open(image_filename)
    if scale_factor != 1.0:
        width, height = pil_image.size
        newsize = (int(width * scale_factor), int(height * scale_factor))
        pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
    image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
    if len(image.shape) == 2:
        image = image[:, :, None].repeat(3, axis=2)
    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
    return image

def save_numpy_image(image, image_filename):
    """
    Save the image to the given filename.
    Args:
        image -> np.ndarray : image
        image_filename -> str : image filename
    """
    pil_image = Image.fromarray(image)
    pil_image.save(image_filename)

def compare_psnr(pred_rgb, gt_rgb) -> float:
    """
    Compare PSNR between predicted and ground truth RGB values.
    Args:
        pred_rgb -> torch.Tensor() : predicted RGB values
        gt_rgb -> torch.Tensor() : ground truth RGB values
    Returns:
        -> float : PSNR values
    """
    mse = torch.mean((pred_rgb - gt_rgb) ** 2)
    psnr = 10 * torch.log10(1.0 / mse)
    return float(psnr)