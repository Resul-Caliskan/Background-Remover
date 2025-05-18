# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:24:07 2025

@author: resul
"""

"""
Utility functions for image processing.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detect_object_bounds(image, debug=False):
    """
    Detect the boundaries of an object in an image with transparency.
    
    Args:
        image (PIL.Image): Image object (in RGBA mode)
        debug (bool): If True, display image and alpha mask
    
    Returns:
        tuple: (left, top, right, bottom) coordinates or None if no object detected
    """
    # Convert image to NumPy array
    img_array = np.array(image)
    
    # Use alpha channel for masking
    alpha_channel = img_array[:, :, 3]
    mask = alpha_channel > 128
    
    # Check if mask is empty
    if not np.any(mask):
        return None
    
    # Find boundaries
    rows_with_object = np.any(mask, axis=1)
    cols_with_object = np.any(mask, axis=0)
    
    top = np.argmax(rows_with_object)
    bottom = len(rows_with_object) - np.argmax(rows_with_object[::-1]) - 1
    left = np.argmax(cols_with_object)
    right = len(cols_with_object) - np.argmax(cols_with_object[::-1]) - 1
    
    # Display debug visualization if requested
    if debug:
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title("RGBA Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Alpha Mask")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return (left, top, right, bottom)

def convert_to_png(input_path):
    """
    Convert image to PNG format if needed.
    
    Args:
        input_path (str): Path to input image
    
    Returns:
        str: Path to PNG image (original or converted)
    """
    try:
        # Check if already PNG
        if input_path.lower().endswith('.png'):
            return input_path
            
        # Open the image
        input_img = Image.open(input_path)
        
        # Check and convert image format
        if input_img.mode != 'RGB' and input_img.mode != 'RGBA':
            input_img = input_img.convert('RGB')
            print(f"Image converted from {input_img.mode} format to RGB")
        
        # Check file type and convert if needed
        file_ext = Path(input_path).suffix.lower()
        if file_ext in ['.dng', '.cr2', '.nef', '.arw']:
            print(f"RAW format detected: {file_ext}, converting...")
            input_img = input_img.convert('RGB')
        
        # Create output PNG path
        output_path = os.path.splitext(input_path)[0] + '.png'
        
        # Save as PNG
        input_img.save(output_path)
        print(f"Image converted and saved as {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"Error during image conversion: {e}")
        # Return original path if conversion fails
        return input_path