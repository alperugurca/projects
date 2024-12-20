#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global AI Image Processing Bootcamp
Dec 2024
Diclehan and Oguzhan Ulucan
"""

import cv2
from utils import *

# Input image is from Animals with Attributes 2 (AwA2)

# Apply light sources to the input images
input_img = cv2.cvtColor(cv2.imread("input.jpg"), cv2.COLOR_BGR2RGB) / 255.0
get_manipulated_images(input_img)


# Apply color constancy to manipulated images
input_img = cv2.cvtColor(cv2.imread("manipulated_purplish.jpg"), cv2.COLOR_BGR2RGB) / 255.0
get_wb_images(input_img)

