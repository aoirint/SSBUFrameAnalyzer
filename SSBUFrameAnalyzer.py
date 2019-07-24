from skimage.feature import hog
import cv2
import json
# import numpy as np
from SSBUDigitClassifier import SSBUDigitClassifier
from SSBUNameRecognizer import SSBUNameRecognizer
from SSBUCharaClassifier import SSBUCharaClassifier
from SSBUStockClassifier import SSBUStockClassifier
import SSBUBoundingBoxUtil


class SSBUFrameAnalyzer:
    def __init__(self, digit_classifier, name_recognizer, chara_classifier, stock_classifier):
        self.digit_classifier = digit_classifier
        self.name_recognizer = name_recognizer
        self.chara_classifier = chara_classifier
        self.stock_classifier = stock_classifier

    def __call__(self, frame, fighter_num=2):
        dmgs = self.analyze_damage(frame, fighter_num=fighter_num)
        names = self.analyze_name(frame, fighter_num=fighter_num)
        charas = self.analyze_chara(frame, fighter_num=fighter_num)
        stocks = self.analyze_stock(frame, fighter_num=fighter_num)

        fighters = {}
        for fighter_idx in range(fighter_num):
            fighters[fighter_idx] = {
                'chara_name': charas[fighter_idx],
                'name': names[fighter_idx],
                'damage': dmgs[fighter_idx],
                'stocks': stocks[fighter_idx],
            }

        result = {
            'fighters': fighters,
        }

        return result

    def analyze_damage(self, frame, fighter_num):
        fighters_dmg_bboxes = SSBUBoundingBoxUtil.fighters_damage_bboxes(fighter_num=fighter_num)
        assert fighter_num == len(fighters_dmg_bboxes)

        dc = self.digit_classifier
        def predict_digit(img):
            digits, dists = dc(img, k=3)
            min_dist = dists[0]

            thresh_dist = 2.

            digit = digits[0] if min_dist < thresh_dist else None
            return digit

        result = {}
        for fighter_idx in range(fighter_num):
            bboxes = fighters_dmg_bboxes[fighter_idx]

            dmg_str = ''
            for bbox_idx, bbox in enumerate(bboxes):
                dimg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
                dimg = cv2.cvtColor(dimg, cv2.COLOR_BGR2GRAY) # GRAY
                dimg = cv2.resize(dimg, (35, 55)) # FIXME: magic number

                # cv2.imwrite('fighter-dmg-%d-%d.png' % (fighter_idx, bbox_idx, ), dimg)

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
        fighters_name_bbox = SSBUBoundingBoxUtil.fighters_name_bbox(fighter_num=fighter_num)
        assert fighter_num == len(fighters_name_bbox)

        nr = self.name_recognizer

        result = {}
        for fighter_idx in range(fighter_num):
            bbox = fighters_name_bbox[fighter_idx]

            nimg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
            nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2GRAY) # GRAY

            # cv2.imwrite('fighter-name-%d.png' % (fighter_idx, ), nimg)

            name = nr(nimg)
            result[fighter_idx] = name

        return result

    def analyze_chara(self, frame, fighter_num):
        fighters_chara_bbox = SSBUBoundingBoxUtil.fighters_chara_bbox(fighter_num=fighter_num)
        assert fighter_num == len(fighters_chara_bbox)

        cc = self.chara_classifier
        def predict_chara(img):
            names, dists = cc(img, k=3)
            min_dist = dists[0]

            # print(names, dists)
            thresh_dist = 10.

            name = names[0] if min_dist < thresh_dist else None
            return name

        result = {}
        for fighter_idx in range(fighter_num):
            bbox = fighters_chara_bbox[fighter_idx]

            cimg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
            cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY) # GRAY

            # cv2.imwrite('fighter-face-%d.png' % (fighter_idx, ), cimg)

            name = predict_chara(cimg)
            result[fighter_idx] = name

        return result

    def analyze_stock(self, frame, fighter_num):
        fighters_stock_bboxes = SSBUBoundingBoxUtil.fighters_stock_bboxes(fighter_num=fighter_num, stock_num=5)
        assert fighter_num == len(fighters_stock_bboxes)

        sc = self.stock_classifier
        def predict_stock(img):
            stocks, dists = sc(img, k=3)
            min_dist = dists[0]

            # print(stocks, dists)
            thresh_dist = 0.6

            stock = stocks[0] if min_dist < thresh_dist else None
            return stock

        result = {}
        for fighter_idx in range(fighter_num):
            bboxes = fighters_stock_bboxes[fighter_idx]

            stocks = []
            for bbox_idx, bbox in enumerate(bboxes):
                simg = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] # RGB
                simg = cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY) # GRAY

                # cv2.imwrite('fighter-stock-%d-%d.png' % (fighter_idx, bbox_idx, ), simg)

                stock = predict_stock(simg)
                if stock is None:
                    break
                stocks.append(stock)

            result[fighter_idx] = stocks

        return result



if __name__ == '__main__':
    import argparse
    import time
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('digit_dictionary', type=str)
    parser.add_argument('chara_dictionary', type=str)
    parser.add_argument('stock_dictionary', type=str)
    parser.add_argument('input', type=str)
    parser.add_argument('fighter_num', type=int) # TODO: predict
    args = parser.parse_args()

    print('loading digit classifier...')
    digit_classifier = SSBUDigitClassifier(feature_json=args.digit_dictionary)
    print('loaded digit classifier')

    print('loading name recognizer...')
    name_recognizer = SSBUNameRecognizer()
    print('loaded name recognizer')

    print('loading chara classifier...')
    chara_classifier = SSBUCharaClassifier(feature_json=args.chara_dictionary)
    print('loaded chara classifier')

    print('loading stock classifier...')
    stock_classifier = SSBUStockClassifier(feature_json=args.stock_dictionary)
    print('loaded stock classifier')

    analyzer = SSBUFrameAnalyzer(digit_classifier=digit_classifier, name_recognizer=name_recognizer, chara_classifier=chara_classifier, stock_classifier=stock_classifier)

    frame = cv2.imread(args.input, 1) # RGB
    frame = cv2.resize(frame, (1280, 720))

    # print(frame.shape)
    assert frame.shape[1] == 1280 and frame.shape[0] == 720

    t = time.time()

    ret = analyzer(frame, fighter_num=args.fighter_num)
    elapsed = time.time() - t

    print(ret)
    print('FPS: %f (%f s)' % (1/elapsed, elapsed, ))
