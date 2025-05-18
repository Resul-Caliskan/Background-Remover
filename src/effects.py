# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:23:52 2025

@author: resul
"""

"""
Visual enhancement effects for images.
"""

import numpy as np
import cv2
from PIL import Image

def apply_daylight_effect(image, brightness_factor=1.1, contrast_factor=1.05, saturation_factor=1.1, warmth_shift=15):
    """
    Apply a natural daylight effect similar to Photoshop:
    - Brightness and contrast enhancement
    - Slight warm temperature (white balance)
    - Saturation increase
    - Preservation of metallic areas
    
    Args:
        image (PIL.Image): Input image
        brightness_factor (float): Brightness multiplier
        contrast_factor (float): Contrast multiplier
        saturation_factor (float): Saturation multiplier
        warmth_shift (int): White balance warmth shift (higher = warmer)
    
    Returns:
        PIL.Image: Enhanced image
    """
    img_array = np.array(image).astype(float)

    has_alpha = img_array.shape[2] == 4
    if has_alpha:
        alpha_channel = img_array[:, :, 3]
        img_array = img_array[:, :, :3]

    # Detect silver/metallic areas: low saturation + high brightness
    hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    silver_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 150)

    # 1. Brightness & Contrast
    img_array = np.clip(img_array * brightness_factor, 0, 255)
    mean_intensity = np.mean(img_array)
    img_array = (img_array - mean_intensity) * contrast_factor + mean_intensity
    img_array = np.clip(img_array, 0, 255)

    # 2. White balance - warm tone (add to R and G channels)
    warmth_mask = np.ones_like(img_array[:, :, 0], dtype=float)
    warmth_mask[silver_mask] = 0.2  # Less warmth on silver areas

    img_array[:, :, 0] = np.clip(img_array[:, :, 0] + warmth_shift * 0.4 * warmth_mask, 0, 255)  # R
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] + warmth_shift * 0.2 * warmth_mask, 0, 255)  # G

    # 3. Saturation
    hsv = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(float)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
    img_array = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(float)

    # 4. Highlight boost (for silver areas)
    highlight_mask = (img_array > 180)
    img_array = np.where(highlight_mask, np.clip(img_array * 1.05, 0, 255), img_array)

    # Reattach alpha channel if present
    img_array = np.clip(img_array, 0, 255)
    if has_alpha:
        img_array = np.dstack((img_array, alpha_channel))

    return Image.fromarray(img_array.astype(np.uint8))