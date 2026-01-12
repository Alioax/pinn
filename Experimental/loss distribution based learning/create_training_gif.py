"""
Create GIF animation from epoch export images.

This script creates a looping GIF from all images in the epoch_exports directory.
Works with both pinn_adaptive_playground.py and pinn_simple.py exports.
- Each frame (except last) displays for 100ms
- Last frame displays for 5 seconds before looping
"""

import os
import re
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Get script directory
script_dir = Path(__file__).parent
epoch_exports_dir = script_dir / 'results' / 'epoch_exports'
output_gif_path = script_dir / 'results' / 'training_animation.gif'

def extract_epoch_number(filename):
    """Extract epoch number from filename.
    
    Supports both patterns:
    - 'pinn_adaptive_playground_profiles_epoch_00100.png'
    - 'pinn_simple_profiles_epoch_00100.png'
    """
    match = re.search(r'epoch_(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 0

def create_training_gif():
    """Create GIF from epoch export images."""
    
    # Check if epoch_exports directory exists
    if not epoch_exports_dir.exists():
        print(f"Error: Directory not found: {epoch_exports_dir}")
        print("Please run the training script first to generate epoch exports.")
        return
    
    # Get all PNG files from epoch_exports directory that match epoch pattern
    all_png_files = list(epoch_exports_dir.glob('*.png'))
    
    # Filter to only files with epoch numbers in filename
    image_files = [f for f in all_png_files if extract_epoch_number(f.name) > 0]
    
    if not image_files:
        print(f"Error: No epoch export PNG files found in {epoch_exports_dir}")
        print(f"  (Found {len(all_png_files)} PNG files total, but none match epoch pattern)")
        print("  Expected filenames like: 'pinn_simple_profiles_epoch_00100.png' or")
        print("                            'pinn_adaptive_playground_profiles_epoch_00100.png'")
        return
    
    # Sort files by epoch number
    image_files.sort(key=lambda x: extract_epoch_number(x.name))
    # image_files = image_files[::2]
    
    # Detect which script was used based on filename pattern
    first_filename = image_files[0].name
    if 'pinn_simple' in first_filename:
        script_name = 'pinn_simple.py'
    elif 'pinn_adaptive_playground' in first_filename:
        script_name = 'pinn_adaptive_playground.py'
    else:
        script_name = 'unknown'
    
    print(f"Found {len(image_files)} epoch export images in {epoch_exports_dir}")
    print(f"  Source script: {script_name}")
    print(f"  Epoch range: {extract_epoch_number(image_files[0].name):,} to {extract_epoch_number(image_files[-1].name):,}")
    
    # Read all images and ensure they have the same size
    print("\nReading images...")
    images = []
    target_size = None
    
    for img_file in image_files:
        img = imageio.imread(img_file)
        
        # Get target size from first image
        if target_size is None:
            target_size = img.shape[:2]  # (height, width)
            print(f"  Target size: {target_size[1]}x{target_size[0]} pixels")
        
        # Resize if necessary
        if img.shape[:2] != target_size:
            print(f"  Resizing {img_file.name} from {img.shape[1]}x{img.shape[0]} to {target_size[1]}x{target_size[0]}")
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_size[1], target_size[0]), Image.Resampling.LANCZOS)
            img = np.array(pil_img)
        
        images.append(img)
        print(f"  Loaded: {img_file.name}")
    
    # Create frame durations: 100ms (0.1s) for all frames except last (5s)
    durations = [0.1] * (len(images) - 1) + [5.0]
    
    print(f"\nCreating GIF with {len(images)} frames...")
    print(f"  Frame durations: {len(images) - 1} frames at 100ms, 1 frame at 5s")
    print(f"  Output: {output_gif_path}")
    
    # Create GIF with loop
    imageio.mimsave(
        output_gif_path,
        images,
        duration=durations,
        loop=0  # 0 means infinite loop
    )
    
    print(f"\n[SUCCESS] GIF created successfully: {output_gif_path}")
    print(f"  Total frames: {len(images)}")
    print(f"  Total duration per loop: {(len(images) - 1) * 0.1 + 5.0:.1f} seconds")

if __name__ == "__main__":
    print("="*60)
    print("Creating Training Animation GIF")
    print("="*60)
    create_training_gif()
