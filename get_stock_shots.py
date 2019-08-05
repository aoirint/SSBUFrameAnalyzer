import cv2
import argparse
from PIL import Image
import os
import numpy as np
import SSBUBoundingBoxUtil

parser = argparse.ArgumentParser()
parser.add_argument('in_file', type=str)
parser.add_argument('out_dir', type=str)
parser.add_argument('fighter_num', type=int)
args = parser.parse_args()


os.makedirs(args.out_dir, exist_ok=True)


frame_files = None
cap = None
if os.path.isdir(args.in_file):
    frame_files = [ os.path.join(args.in_file, file) for file in os.listdir(args.in_file) ]
else:
    cap = cv2.VideoCapture(args.in_file)

frame_idx = 0
def next_frame():
    global frame_idx

    if frame_files is not None:
        if len(frame_files) <= frame_idx:
            return None
        frame = cv2.imread(frame_files[frame_idx])
    else:
        ret, frame = cap.read()

    frame_idx += 1
    return frame

in_file = os.path.basename(args.in_file)

fighters_stock_bboxes = SSBUBoundingBoxUtil.fighters_stock_bboxes(args.fighter_num, stock_num=3)

while True:
    img = next_frame()
    if img is None:
        break

    for fighter_idx, bboxes in enumerate(fighters_stock_bboxes):
        for idx in range(len(bboxes)):
            # if idx != 3:
            #     continue
            bbox = bboxes[idx]
            chara = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]

            file = '%s-f%d-s%d-%d.png' % (in_file, fighter_idx, idx, frame_idx, )
            path = os.path.join(args.out_dir, file)
            cv2.imwrite(path, chara)

        print(frame_idx, fighter_idx)
