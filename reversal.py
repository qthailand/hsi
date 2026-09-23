# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 09:34:18 2026

@author: TSC
"""
import spectral as sp
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = sp.open_image(
    r"C:\Datacubes\New folder (2)\curcumin001-FromSelection-ConvertFormat.bil.hdr")
arr = image.load()
arr = arr.transpose((1, 0, 2))

line0 = arr[0, :, :]
plt.figure(figsize=(10, 5))
plt.imshow(
    line0,
    aspect='auto',
)
plt.xlabel('Spatial Pixel')
plt.ylabel('Band')
plt.title('Raw Pushbroom Line')
plt.colorbar(label='Intensity')
plt.show()
