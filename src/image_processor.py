# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:23:34 2025

@author: resul
"""

"""
Core image processing functions for background removal and object manipulation.
"""

import os
import time
import numpy as np
from PIL import Image

from src.effects import apply_daylight_effect
from src.utils import detect_object_bounds

def center_and_resize_object(input_path, output_path, target_size=(1600, 1600), 
                            size_factor=1.25, brightness_factor=1.2, background_color=(255, 255, 255, 255)):
    """
    Center and resize an object in an image with transparency.
    
    Args:
        input_path (str or PIL.Image): Input image path or PIL Image object
        output_path (str): Output image path
        target_size (tuple): Target dimensions (width, height)
        size_factor (float): Size multiplier for the object
        brightness_factor (float): Brightness adjustment factor
        background_color (tuple): Background color as (R,G,B,A)
    
    Returns:
        tuple: (success, message)
    """
    try:
        start_time = time.time()
        
        # Open image if path is provided, otherwise use the PIL Image object
        if isinstance(input_path, str):
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"File not found: {input_path}")
            input_img = Image.open(input_path)
        else:
            input_img = input_path
        
        # Convert to RGBA mode for transparency
        if input_img.mode != 'RGBA':
            input_img = input_img.convert('RGBA')
        
        # Detect object boundaries
        bounds = detect_object_bounds(input_img)
        if not bounds:
            raise ValueError("No object detected in image or no alpha channel")
        
        left, top, right, bottom = bounds
        
        # Calculate object dimensions
        object_width = right - left
        object_height = bottom - top
        
        # Find object center
        object_center_x = (left + right) // 2
        object_center_y = (top + bottom) // 2
        
        print(f"Object bounds: Left={left}, Top={top}, Right={right}, Bottom={bottom}")
        print(f"Object dimensions: {object_width}x{object_height} pixels")
        print(f"Object center: ({object_center_x}, {object_center_y})")
        
        # Create new image with target dimensions
        new_width, new_height = target_size
        new_img = Image.new('RGBA', (new_width, new_height), background_color)
        
        # Find the largest dimension of the object
        max_object_dimension = max(object_width, object_height)
        
        # Scale object according to target size and size factor
        scale_factor = min(new_width, new_height) / max_object_dimension * size_factor
        
        print(f"Size factor: {size_factor} (actual scale: {scale_factor:.2f})")
        
        # Resize object
        new_object_width = int(object_width * scale_factor)
        new_object_height = int(object_height * scale_factor)
        
        print(f"New object dimensions: {new_object_width}x{new_object_height} pixels")
        
        # Crop original object
        cropped_img = input_img.crop((left, top, right, bottom))
        
        # Resize cropped object
        resized_img = cropped_img.resize((new_object_width, new_object_height), Image.LANCZOS)
        
        # Apply daylight effect (Photoshop-like)
        if brightness_factor != 1.0:
            resized_img = apply_daylight_effect(
                resized_img,
                brightness_factor=brightness_factor,
                contrast_factor=1.05,
                saturation_factor=1.15,
                warmth_shift=15  # Slight warmth
            )
            print(f"Daylight effect applied, brightness factor: {brightness_factor:.2f}")
        
        # Calculate target image center
        target_center_x = new_width // 2
        target_center_y = new_height // 2
        
        # Position object in center of target image
        paste_x = target_center_x - (new_object_width // 2)
        paste_y = target_center_y - (new_object_height // 2)
        
        # Paste object
        new_img.paste(resized_img, (paste_x, paste_y), resized_img)
        
        # Check and create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save result
        new_img.save(output_path)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Object centered and resized: {output_path} ({elapsed_time:.2f} seconds)")
        
        return True, output_path
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False, str(e)