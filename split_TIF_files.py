#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 22:16:42 2022

@author: juan
"""

from PIL import Image

img = Image.open('images/training.tif')
msk = Image.open('images/training_groundtruth.tif')


#for i in range(4):
i = 0
while True:
    try:
        img.seek(i)
        img.save('images/largeDS_generatePatches/images/img%s.tif'%(i,))
        i = i + 1
    except EOFError:
        break
    
i = 0
while True:
    try:
        msk.seek(i)
        msk.save('images/largeDS_generatePatches/masks/mask%s.tif'%(i,))
        i = i + 1
    except EOFError:
        break    