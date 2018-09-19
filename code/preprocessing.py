# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:27:16 2018

@author: sguly
"""
import os
import numpy as np
from PIL import Image, ImageOps
import skimage
from multiprocessing import Pool
import itertools

def crop_retina(retina_df, root_image):
    
    if not os.path.exists(root_image):
        os.makedirs(root_image)

    with Pool() as p:
        p.starmap(crop_one_retina, itertools.product(range(len(retina_df)), [root_image]))
    return;

def crop_one_retina(image_name, root_image):
    image = Image.open(image_name)

    image_np = np.array(image.convert('L'))
    mask = image_np > 10
    
    lbls = skimage.measure.label(mask, background=0)
    largestCC = lbls == np.argmax(np.bincount(lbls.flat)[1:])+1

    ymin, ymax = np.where(largestCC)[0].min(), np.where(largestCC)[0].max()
    xmin, xmax = np.where(largestCC)[1].min(), np.where(largestCC)[1].max()
    image_crop = image.crop((xmin, ymin, xmax, ymax))
    
    w,h = image_crop.size
    s = max(w,h)
    
    padding = (int((s-w)/2), int((s-h)/2), int((s-w)/2), int((s-h)/2))
    image_sq = ImageOps.expand(image_crop, padding)
    image_sq.save(os.path.join(root_image, os.path.basename(image_name)))