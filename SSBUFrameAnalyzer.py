from skimage.feature import hog
import cv2
import json
# import numpy as np
from SSBUDigitClassifier import SSBUDigitClassifier
from SSBUNameRecognizer import SSBUNameRecognizer

class SSBUFrameAnalyzer:
    def __init__(self, digit_classifier, name_recognizer):
        self.digit_classifier = digit_classifier
        self.name_recognizer = name_recognizer

    def __call__(self, frame):
        fighter_num = 2

        dmgs = self.analyze_damage(frame, fighter_num=fighter_num)
        names = self.analyze_name(frame, fighter_num=fighter_num)

        fighters = {}
        for fighter_idx in range(fighter_num):
            fighters[fighter_idx] = {
                'name': names[fighter_idx],
                'damage': dmgs[fighter_idx],
            }

        result = {
            'fighters': fighters,
        }

        return result

    def analyze_damage(self, frame, fighter_num):
        assert fighter_num == 2, 'Not implemented'

        dc = self.digit_classifier
        def predict_digit(img):
            digits, dists = dc(img, k=3)
            min_dist = dists[0]

            thresh_dist = 2.

            digit = digits[0] if min_dist < thresh_dist else None
            return digit

        # 2P Battle Only
        fighters_dmg_bboxes = [ # FIXME: magic number
            # Fighter 1
            [
                [ 335   ,610   , 35,55 ],
                [ 335+30,610   , 35,55 ],
                [ 335+65,610   , 35,55 ],
                [ 335+97,610+28, 18,25 ],
            ],
            # Fighter 2
            [
                [ 830   ,610   , 35,55 ],
                [ 830+30,610   , 35,55 ],
                [ 830+65,610   , 35,55 ],
                [ 830+97,610+28, 18,25 ],
            ],
        ]

        assert fighter_num == len(fighters_dmg_bboxes)

        result = {}
        for fighter_idx in range(fighter_num):
            bboxes = fighters_dmg_bboxes[fighter_idx]

            dmg_str = ''
            for bbox_idx, bbox in enumerate(bboxes):
                dimg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
                dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY) # GRAY
                dimg = cv2.resize(dimg, (35, 55)) # FIXME: magic number

                # cv2.imwrite('fighter-%d-%d.png' % (fighter_idx, bbox_idx, ), dimg)

                digit = predict_digit(dimg)
                if digit is None:
                    digit = 0 # placeholder

                end = bbox_idx == len(bboxes)-1

                digit_str = str(digit)
                if end:
                    digit_str = '.' + digit_str
                dmg_str += digit_str

            dmg = float(dmg_str)
            result[fighter_idx] = dmg

        return result

    def analyze_name(self, frame, fighter_num):
        assert fighter_num == 2, 'Not implemented'

        nr = self.name_recognizer

        # 2P Battle Only
        fighters_name_bbox = [ # FIXME: magic number
            # Fighter 1
            [ 350,670, 120,16 ],
            # Fighter 2
            [ 845,670, 120,16 ],
        ]

        assert fighter_num == len(fighters_name_bbox)

        result = {}
        for fighter_idx in range(fighter_num):
            bbox = fighters_name_bbox[fighter_idx]

            nimg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY) # GRAY

            name = nr(nimg)
            result[fighter_idx] = name

        return result



if __name__ == '__main__':
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('digit_dictionary', type=str)
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    print('loading digit classifier...')
    digit_classifier = SSBUDigitClassifier(feature_json=args.digit_dictionary)
    print('loaded digit classifier')

    print('loading name recognizer...')
    name_recognizer = SSBUNameRecognizer()
    print('loaded name recognizer')

    analyzer = SSBUFrameAnalyzer(digit_classifier=digit_classifier, name_recognizer=name_recognizer)

    frame = cv2.imread(args.input, 1) # RGB
    frame = cv2.resize(frame, (1280, 720))

    # print(frame.shape)
    assert frame.shape[1] == 1280 and frame.shape[0] == 720

    t = time.time()

    ret = analyzer(frame)
    elapsed = time.time() - t

    print(ret)
    print('FPS: %f (%f s)' % (1/elapsed, elapsed, ))
