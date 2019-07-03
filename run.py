
from multiprocessing import Process

import cfar
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import ellipse
from skimage.measure import label, regionprops
from skimage.transform import rotate

import glob
import sys
import shutil
import os

def union_area(a,b):

    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0]+a[2], b[0]+b[2]) - x
    h = max(a[1]+a[3], b[1]+b[3]) - y
    return w*h

def intersection_area(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return 0
    else:
        return w * h

def str2int(a):
    return [int(x) for x in a]

def extract_boxes(fname):
    with open(fname) as f:
        content = f.readlines()
        f.close()
        content = [x.strip() for x in content]
        content = [str2int(x.split(' ')[-4:]) for x in content]
        return content

def get_precision_recall(threshold):
    paths = glob.glob("/media/nasir/Drive1/code/SAR/AutomatedSARShipDetection/python_cfar/SAR-Ship-Dataset/detection-results/*.txt")
    files_stats = {}
    falseNegative = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0

    for index, path in enumerate(paths):
        pred_bboxes = extract_boxes(path)
        gt_bboxes = extract_boxes(path.replace('detection-results', 'ground-truth'))
        fp = 0; tp = 0; fn = 0
        box_index_of_tp = []
        for index_g, gt_box in enumerate(gt_bboxes):
            ious = []
            for index_p, pred_box in enumerate(pred_bboxes):
                iou = intersection_area(gt_box, pred_box) / union_area(gt_box, pred_box)
                if iou > threshold:
                    box_index_of_tp.append(index_p)
                    ious.append(iou)

            if len(ious) == 0:
                fn+=1
            elif len(ious) > 0:
                tp+=1

        diff = len(pred_bboxes) - (len(list(set(box_index_of_tp))))
        if diff > 0:
            fp+=diff

        falseNegative+=fn
        truePositive+=tp
        falsePositive+=fp

        files_stats[path.split('/')[-1].split('.')[0]] = {
            "falseNegative": fn,
            "truePositive": tp,
            "falsePositive": fp
        }

        sys.stdout.write(f"\r {index + 1} / {len(paths)}")
        sys.stdout.flush()
    print(f"\n\nfalsePositives: {falsePositive} , truePositives: {truePositive} , falseNegatives: {falseNegative}")
    recall = truePositive / (truePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    return precision, recall

def predict(paths, root, source, dest, i): 
    
    CACFAR = cfar.ca_cfar

    for index in range(0, len(paths)):
    # for index in range(0, 12):
        path = paths[index]
        output_file = path.replace(f"{source}", f'{dest}')
        box_file = path.replace(f"{source}", 'detection-results').replace('.jpg', '.txt')
        gt_file = box_file.replace('detection-results', 'ground-truth')
        CACFAR(path, output_file, box_file, gt_file, 100, 40, 30, 1.55)
        
        sys.stdout.write(f'\r {i}: {index + 1} / {len(paths)}')
        sys.stdout.flush()
    
    sys.stdout.write(f"\r {i}: Done\n")
    sys.stdout.flush()

    # sys.stdout.write(f"\n\n")
    # sys.stdout.flush()

if __name__ == "__main__":
    source = "subset"
    # source = "JPEGImages"

    dest = "results"
    
    root = "/media/nasir/Drive1/code/SAR/AutomatedSARShipDetection/python_cfar/SAR-Ship-Dataset"
    num_of_process = 3

    paths = glob.glob(f"{root}/{source}/*.jpg")
    os.path.exists(f'{root}/detection-results') and shutil.rmtree(f'{root}/detection-results')
    os.path.exists(f'{root}/{dest}') and shutil.rmtree(f'{root}/{dest}')

    if not os.path.exists(f'{root}/detection-results'):
        os.mkdir(f'{root}/detection-results')
        print(f"Directory  detection-results Created ")
    
    if not os.path.exists(f'{root}/{dest}'):
        os.mkdir(f'{root}/{dest}')
        print(f"Directory  {dest} Created ")

    proceses = []
    paths_per_process = len(paths) // num_of_process

    for i in range(0, num_of_process):
        start = paths_per_process*i
        end = (i+1)*paths_per_process
        p = Process(target=predict, args=(paths[start : end], root, source, dest, i))
        proceses.append(p)
        p.start()
        
    if end + 1 < len(paths):
        p = Process(target=predict, args=(paths[end:], root, source, dest, num_of_process))
        proceses.append(p)
        p.start()
    
    for p in proceses:
        p.join()


    # predict(paths, root, source, dest)

    thresholds = [0.4]
    precisions = []
    recalls = []

    for threshold in thresholds:
        precision, recall = get_precision_recall(threshold)
        precisions.append(precision)
        recalls.append(recall)
        print(f"\nthreshold: {threshold} recall: {round(recall * 100, 2)}% precision: {round(precision*100, 2)}% \n")

