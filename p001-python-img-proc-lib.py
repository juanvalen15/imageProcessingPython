#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 08:05:36 2022

@author: juan
"""

from skimage import io
img1 = io.imread("images/Osteosarcoma_01.tif")

import cv2
img2 = cv2.imread("images/Osteosarcoma_01.tif")

import numpy as 