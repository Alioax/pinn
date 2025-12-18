"""
Create GIF animation from PINN training epoch plots.

This script collects all epoch plot images from results/gif_frames/ and creates
a GIF animation showing the training progress.
"""

import imageio.v2 as imageio
import glob
import re
from pathlib import Path
import numpy as np
from PIL import Image

# Script directory
script_dir = Path(__file__).parent
gif_frames_dir = script_dir / 'results' / 'gif_frames'
output_gif_path = script_dir / 'results' / 'pinn_training_animation.gif'

def extract_epoch_number(filename):
    """Extract epoch number from filename like 'pinn_analytical_comparison_with_pde_loss_epoch_0100.png'"""
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def create_training_gif():
    """Create GIF animation from epoch plot images."""
    
    # Find all epoch plot files
    pattern = str(gif_frames_dir / 'pinn_analytical_comparison_with_pde_loss_epoch_*.png')
    image_files = glob.glob(pattern)
    
    if not image_files:
        print(f"No epoch plot files found in {gif_frames_dir}")
        print(f"Pattern searched: {pattern}")
        return
    
    # Sort files by epoch number
    image_files.sort(key=extract_epoch_number)
    
    print(f"Found {len(image_files)} epoch plot files")
    print(f"Epochs: {extract_epoch_number(image_files[0])} to {extract_epoch_number(image_files[-1])}")
    
    # First pass: read first image to determine target size and channels
    print("Reading first image to determine target dimensions...")
    first_img = imageio.imread(image_files[0])
    first_shape = first_img.shape
    target_height = first_shape[0]
    target_width = first_shape[1]
    target_channels = first_shape[2] if len(first_shape) == 3 else 1
    
    # Cap at reasonable maximum for GIFs (1920x1080) to avoid memory issues
    MAX_GIF_WIDTH = 1920
    MAX_GIF_HEIGHT = 1080
    
    if target_width > MAX_GIF_WIDTH or target_height > MAX_GIF_HEIGHT:
        # Scale down proportionally to fit within max dimensions
        scale = min(MAX_GIF_WIDTH / target_width, MAX_GIF_HEIGHT / target_height)
        target_width = int(target_width * scale)
        target_height = int(target_height * scale)
        print(f"Images are very large ({first_shape[1]}x{first_shape[0]}). Resizing to {target_width}x{target_height} for GIF...")
    else:
        print(f"Standardizing image sizes to {target_width}x{target_height} (channels: {target_channels})...")
    
    # Process images one at a time to avoid memory issues
    print("Processing images...")
    standardized_images = []
    
    for i, image_file in enumerate(image_files):
        if i % 10 == 0:
            print(f"  Processing image {i+1}/{len(image_files)}...")
        
        # Read and process image immediately
        img = imageio.imread(image_file)
        current_shape = img.shape
        
        # Always resize to target dimensions (more memory efficient)
        if len(current_shape) == 3:
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.fromarray(img, mode='L')
        
        pil_img = pil_img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        img = np.array(pil_img)
        
        # Ensure channel count matches target
        if len(img.shape) == 2 and target_channels > 1:
            # Convert grayscale to RGB/RGBA
            if target_channels == 3:
                img = np.stack([img] * 3, axis=-1)
            elif target_channels == 4:
                img = np.stack([img] * 3 + [np.ones_like(img) * 255], axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 4 and target_channels == 3:
                # RGBA to RGB
                img = img[:, :, :3]
            elif img.shape[2] == 3 and target_channels == 4:
                # RGB to RGBA
                alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                img = np.concatenate([img, alpha], axis=2)
        
        standardized_images.append(img)
    
    # Create GIF
    # Duration: 0.1 seconds (100ms) per frame
    # Loop: 0 = infinite loop
    print(f"Creating GIF with {len(standardized_images)} frames at 0.1s per frame...")
    imageio.mimsave(
        str(output_gif_path),
        standardized_images,
        duration=0.1,  # 100ms per frame
        loop=0  # Infinite loop
    )
    
    print(f"GIF animation saved to: {output_gif_path}")
    print(f"Total duration: {len(standardized_images) * 0.1:.1f} seconds")

if __name__ == "__main__":
    create_training_gif()
