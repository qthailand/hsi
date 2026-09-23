# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:15:04 2026

@author: TSC
"""
import numpy as np
data = np.load(
    f"C:\ClassificationPipeLine\CameraSystem-V3\hsi\dark_frames\dark_b2_e100.0_g180.0.npz")
print(data['mean'].shape)
