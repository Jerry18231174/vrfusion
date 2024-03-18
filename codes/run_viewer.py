import torch
import numpy as np
import argparse
import json
import os

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from models.real import load_model
from dataset.real_data import load_dataset, compare_psnr, save_numpy_image
from dataset.cameras import FPSCamera
from dataset.rays import RaySamples, RayBundle
from viewer.ui import UI
from integrators.depth import *
from integrators.normal import *


def run_viewer(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the dataset
    train_dataset = load_dataset("train", config)

    # Prepare the model and optimizer
    model = load_model(config["model"], train_dataset.scene_bbox.to(device)).to(device)
    model_ckpt_dir = os.path.join("outputs", config["dataset"]["name"], config["model"]["type"])
    model.load_state_dict(torch.load(os.path.join(model_ckpt_dir, "model_30000.pt")))
    model.eval()
    fps_camera = train_dataset.cameras.yield_fps_camera(0)

    # Load the virtual scene
    scene : mi.Scene = mi.load_file(os.path.join("data", config["scene"]["name"], "scene.xml"))
    scene_params = mi.traverse(scene)
    # fps_camera = FPSCamera(speed=0.1)
    # cam_matrix = scene_params["camera.to_world"].matrix.torch().to(device)[0, :3]
    # cam_matrix = torch.cat([-cam_matrix[:, 0:1], cam_matrix[:, 1:2], -cam_matrix[:, 2:3], cam_matrix[:, 3:4]], dim=1)
    # width, height = scene_params["camera.film.size"]
    # fps_camera.setup(cam_matrix, {
    #     "width": width,
    #     "height": height,
    #     "focal_length": width,
    # })
    depth_integrator = mi.load_dict({"type": "depth"})
    normal_integrator = mi.load_dict({"type": "normal"})

    # Initialize the viewer
    ui = UI(fps_camera.width, fps_camera.height, fps_camera)
    scene_params["camera.film.size"] = mi.ScalarVector2u(fps_camera.width, fps_camera.height)
    scene_params["camera.x_fov"] = mi.Float(2 * np.arctan(fps_camera.width / (2 * fps_camera.focal_length)) * 180 / np.pi)

    # Run the viewer
    need_render = True
    last_state = fps_camera._camera_matrix.clone()
    with torch.no_grad():
        while not ui.should_close():
            ui.begin_frame()

            if True or need_render:
                # Render the virtual (mitsuba) scene
                scene_params["camera.to_world"] = fps_camera.get_camera_matrix(camera_type="mitsuba").cpu().numpy()
                scene_params.update()
                v_depth = mi.render(scene, integrator=depth_integrator)
                v_depth = v_depth.torch()[:, :, 0]
                img = mi.render(scene)
                img = img.torch()
                img = torch.log1p(img)
                img = img ** (1 / 2.2)

                # Render the real (NeRF) scene
                ray_bundle, _ = fps_camera.get_all_rays(0)
                # Generate virtual ray samples
                virtual = {
                    "depth": v_depth.view(-1, 1),
                    "rgb": img.view(-1, 1, 3),
                    "density": 100000.0
                }
                ray_bundle.to(device)
                pred_rgb = model.get_eval_rgb(ray_bundle, virtual=virtual)
                img = pred_rgb.view(fps_camera.height, fps_camera.width, -1)

                # img = np.random.random((720, 1280, 3))
                ui.write_texture_cpu(img.cpu().numpy())
            ui.end_frame()
            need_render = (last_state != fps_camera._camera_matrix).any()
            last_state = fps_camera._camera_matrix.clone()
        # Close the viewer
        ui.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRFusion Viewer Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    run_viewer(config)