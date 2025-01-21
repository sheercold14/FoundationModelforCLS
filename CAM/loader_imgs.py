import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from defaults import *
from utils.helpfuns import check_dir
from torch.utils.data import DataLoader
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Save images from val dataloader')
    parser.add_argument('--params_path', type=str, default="./configs/cross/param_all_organ.json", help='Path to the JSON file containing the parameters')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for the validation dataloader')
    return parser.parse_args()

def save_images(images, save_dir, batch_idx):
    check_dir(save_dir)  # Ensure directory exists
    
    for i, img in enumerate(images):
        img_path = os.path.join(save_dir, f"val_batch_{batch_idx}_img_{i}.png")
        
        # Convert tensor to NumPy array and save as an image
        img_np = img.cpu().numpy().transpose(1, 2, 0)  # Assuming format (C, H, W)
        img_np = (img_np * 255).astype(np.uint8)  # Rescale to 0-255 range
        
        plt.imsave(img_path, img_np)

def process_validation(val_dataloader, save_dir):
    # Iterate over the validation dataloader and save images
    for batch_idx, (images, _) in enumerate(val_dataloader):
        save_images(images, save_dir, batch_idx)

def main(args):
    # Load parameters and setup system (assumed to be defined in your framework)
    parameters = edict(load_params(args.params_path))
    define_system_params(parameters.system_params)

    # Setup validation dataset and dataloader
    val_dataset = setup_val_dataset(parameters)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Process the validation dataloader and save images
    process_validation(val_dataloader, args.save_dir)

if __name__ == '__main__':
    args = parse_arguments()

    # Execute the main function
    main(args)