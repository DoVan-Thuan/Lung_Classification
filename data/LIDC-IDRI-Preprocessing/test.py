import numpy as np
import cv2
import os
import glob

folder = "nodules"
if not os.path.exists(folder): os.mkdir(folder)

for name in os.listdir('data/Image'):
    if "LIDC" not in name: continue
    images = glob.glob(f'data/Image/{name}/*.npy')
    masks = glob.glob(f'data/Mask/{name}/*.npy')
    for i in range(len(images)):
        mask = np.load(masks[i])
        image = np.load(images[i])
