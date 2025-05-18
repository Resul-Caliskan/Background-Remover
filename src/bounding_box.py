# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:14:01 2025

@author: resul
"""

import numpy as np

def get_bounding_box_from_alpha(image):
    np_img = np.array(image)
    alpha = np_img[:, :, 3]
    mask = alpha > 128
    if not np.any(mask):
        raise ValueError("Alpha maskesi nesne içermiyor.")
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
    left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1
    return (left, top, right, bottom)
