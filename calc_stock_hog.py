import argparse
from skimage.feature import hog
import numpy as np
import os
import cv2
import json

parser = argparse.ArgumentParser()
parser.add_argument('in_dir', type=str)
parser.add_argument('out_file', type=str)

args = parser.parse_args()

data = {}
for cat_name in sorted(os.listdir(args.in_dir)):
    cat_path = os.path.join(args.in_dir, cat_name)
    if not os.path.isdir(cat_path):
        continue

    data_chara = []
    for in_file in sorted(os.listdir(cat_path)):
        in_path = os.path.join(cat_path, in_file)
        img = cv2.imread(in_path, 0) # grayscale
        img = cv2.resize(img, (30, 30))
        ft = hog(img)
        data_chara.append(ft.tolist())

    data[cat_name] = data_chara

os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
with open(args.out_file, 'w') as fp:
    json.dump(data, fp)
