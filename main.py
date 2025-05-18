# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:17:26 2025

@author: resul
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main script for background removal and image processing.
"""
import os
import time
import argparse
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from dis_bg_remover import remove_background

from src.image_processor import center_and_resize_object
from src.utils import convert_to_png

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Background Remover and Image Processor')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--output', '-o', required=True, help='Output image path')
    parser.add_argument('--model', '-m', default='./models/isnet_dis.onnx', help='Path to model file')
    parser.add_argument('--size', nargs=2, type=int, default=[1600, 1600], help='Target size (width height)')
    parser.add_argument('--factor', type=float, default=0.94, help='Size factor')
    parser.add_argument('--brightness', type=float, default=1.05, help='Brightness factor')
    parser.add_argument('--show-plots', action='store_true', help='Show plots of processing stages')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    print(f"Processing image: {args.input}")
    start_time = time.time()
    
    # Step 1: Convert image to PNG if needed
    png_path = convert_to_png(args.input)
    
    # Step 2: Remove background
    print("Removing background...")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    img, mask = remove_background(args.model, png_path)
    
    # Save the mask for debugging
    mask_path = os.path.join(os.path.dirname(args.output), 'mask.jpg')
    cv2.imwrite(mask_path, mask)
    
    # Save the transparent background image
    temp_transparent = os.path.join(os.path.dirname(args.output), 'temp_transparent.png')
    cv2.imwrite(temp_transparent, img)
    
    # Display original and mask if requested
    if args.show_plots:
        fig = plt.figure(figsize=(12, 6))
        
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
        ax2.set_title("Mask")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Step 3: Center and resize the object with proper lighting
    print("Centering and resizing object...")
    transparent_img = Image.open(temp_transparent)
    if transparent_img.mode != 'RGBA':
        transparent_img = transparent_img.convert('RGBA')
    
    success, output_file = center_and_resize_object(
        transparent_img,
        args.output,
        target_size=tuple(args.size),
        size_factor=args.factor,
        brightness_factor=args.brightness,
        background_color=(255, 255, 255, 255)  # White background (RGBA)
    )
    
    # Clean up temporary files
    if os.path.exists(temp_transparent) and temp_transparent != args.input:
        os.remove(temp_transparent)
    if png_path != args.input:
        os.remove(png_path)
    
    # Display final result if requested
    if success and args.show_plots:
        final_img = Image.open(args.output)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.array(final_img))
        plt.title("Final Processed Image (Centered and Resized)")
        plt.axis('off')
        plt.show()
    
    elapsed_time = time.time() - start_time
    print(f"Process completed in {elapsed_time:.2f} seconds.")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()