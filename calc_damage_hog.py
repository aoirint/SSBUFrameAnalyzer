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

    digit = int(cat_name)
    print(digit)

    data_digit = []
    for in_file in sorted(os.listdir(cat_path)):
        in_path = os.path.join(cat_path, in_file)
        img = cv2.imread(in_path, 0) # grayscale
        img = cv2.resize(img, (35, 55))
        ft = hog(img)
        data_digit.append(ft.tolist())

    data[digit] = data_digit

os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
with open(args.out_file, 'w') as fp:
    json.dump(data, fp)
