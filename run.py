

import cfar
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

import glob
import sys

# source = "JPEGImages"
# dest = "results"

source = "subset"
dest = "subset_results"
paths = glob.glob(f"/media/nasir/Drive1/code/SAR/AutomatedSARShipDetection/python_cfar/SAR-Ship-Dataset/{source}/*.jpg")
CACFAR = cfar.ca_cfar

sys.stdout.write(f"\n")
sys.stdout.flush()

for index in range(0, len(paths)):
# for index in range(0, 12):
    path = paths[index]
    output_file = path.replace(f"{source}", f'{dest}')
    box_file = path.replace(f"{source}", 'detection-results').replace('.jpg', '.txt')
    gt_file = box_file.replace('detection-results', 'ground-truth')
    image = CACFAR(path, output_file, box_file, gt_file, 100, 40, 20, 1.35)
    sys.stdout.write(f"\r {index + 1} / {len(paths)}")
    sys.stdout.flush()

sys.stdout.write(f"\n\n")
sys.stdout.flush()
    