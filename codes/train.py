import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import os
from tqdm import tqdm

from models.real import load_model
from dataset.real_data import load_dataset, compare_psnr, save_numpy_image


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_nerf(config):
    train_schedual = {
        "eval_every": config["model"]["eval_every"] if "eval_every" in config["model"] else 1000,
        "save_every": config["model"]["save_every"] if "save_every" in config["model"] else 10000,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    train_dataset = load_dataset("train", config)
    train_loader = torch.utils.data.DataLoader(
        np.arange(config["model"]["training_steps"]),
        batch_size=1,
        shuffle=True,
        collate_fn=train_dataset.collate_train,
    )
    eval_dataset = load_dataset("eval", config)
    eval_loader = torch.utils.data.DataLoader(
        np.arange(len(eval_dataset)),
        batch_size=1,
        shuffle=False,
        collate_fn=eval_dataset.collate_eval,
    )

    # Prepare the model and optimizer
    model = load_model(config["model"], train_dataset.scene_bbox.to(device)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

    # Load checkpoint if available
    output_dir = os.path.join("outputs", config["dataset"]["name"], config["model"]["type"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(os.path.join(output_dir, "model.pt")):
        model.load_state_dict(torch.load(os.path.join(output_dir, "model.pt")))
        print("Loaded the checkpoint.")

    # Train the model
    np.random.seed(config["model"]["seed"])
    torch.manual_seed(config["model"]["seed"])
    model.train()
    with tqdm(total=len(train_loader)) as tbar:
        tbar.set_description("Training")
        for i, ray_batch in enumerate(train_loader):
            optimizer.zero_grad()
            ray_batch.to(device)
            pred_ray = model(ray_batch)
            loss = torch.mean((pred_ray["rgb"] - ray_batch.rgb) ** 2)
            loss += pred_ray["prop_loss"]
            # loss += pred_ray["dist_loss"]
            loss.backward()
            optimizer.step()
            # Evaluate the model
            if (i + 1) % train_schedual["eval_every"] == 0:
                model.eval()
                print("Evaluating the model...")
                psnrs = torch.zeros(len(eval_loader))
                with torch.no_grad():
                    for j, eval_ray_batch in tqdm(enumerate(eval_loader)):
                        eval_ray_batch.to(device)
                        pred_rgb = model.get_eval_rgb(eval_ray_batch)
                        psnrs[j] = compare_psnr(pred_rgb, eval_ray_batch.rgb)
                        print(f"PSNR: {psnrs[j]}")
                        pred_rgb = pred_rgb.view(eval_dataset.cameras.height, eval_dataset.cameras.width, -1)
                        pred_rgb = (pred_rgb.cpu().numpy() * 255).astype(np.uint8)
                        save_numpy_image(pred_rgb, os.path.join(output_dir, f"eval_{i+1}_{j+1}.png"))
                        # gt_rgb = eval_ray_batch.rgb.view(eval_dataset.cameras.height, eval_dataset.cameras.width, -1)
                        # gt_rgb = (gt_rgb.cpu().numpy() * 255).astype(np.uint8)
                        # save_numpy_image(gt_rgb, os.path.join(output_dir, f"gt_{i+1}_{j+1}.png"))
                        break
                model.train()
            # Save the model
            if (i + 1) % train_schedual["save_every"] == 0:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_{i+1}.pt"))
            # Update progress bar
            tbar.set_postfix(loss=loss.item())
            tbar.update()

    model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRFusion Training Script")
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    config = json.load(open(args.config))

    train_nerf(config)