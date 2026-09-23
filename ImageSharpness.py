# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 09:03:49 2023

@author: nanophotonics
"""
import numpy as np
import cv2
class ImageSharpness:
    img = None
    def __init__(self):
        super().__init__()
    def LoadImage(self, data:np.ndarray)->'ImageSharpness':
        self.img = data
        return self
    def SharpnessScore(self, axis=None):
        """
        axis=None : Laplacian variance (both directions)
        axis=0    : Sobel vertical only — use for pushbroom (spatial axis)
        axis=1    : Sobel horizontal only
        """
        img = self.img.astype(np.float32)
        if axis == 0:
            g = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)
            return float(np.mean(np.abs(g)))
        elif axis == 1:
            g = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3)
            return float(np.mean(np.abs(g)))
        else:
            return float(cv2.Laplacian(img, cv2.CV_32F).var())