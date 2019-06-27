

import cfar
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

import glob
CACFAR = cfar.cfar().execute_cfar
import sys

paths = glob.glob("/media/nasir/Drive1/code/SAR/AutomatedSARShipDetection/python_cfar/SAR-Ship-Dataset/JPEGImages/*.jpg");

for index in range(0, len(paths)):
# for index in range(0, 12):
    path = paths[index]
    output_file = path.replace('JPEGImages', 'results')
    box_file = path.replace('JPEGImages', 'detection-results').replace('.jpg', '.txt')
    gt_file = box_file.replace('detection-results', 'ground-truth')
    image = CACFAR(path, output_file, box_file, gt_file, 50, 40, 0.8)
    sys.stdout.write(f"\r {index} / {len(paths)}")
    sys.stdout.flush()