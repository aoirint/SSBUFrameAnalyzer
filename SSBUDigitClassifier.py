from skimage.feature import hog
import numpy as np
import json

class SSBUDigitClassifier:
    def __init__(self, feature_json):
        self.feature_json = feature_json

        with open(feature_json, 'r') as fp:
            data = json.load(fp)

        digits = []
        features = []
        for digit, fts in data.items():
            # n = 0
            for feature in fts:
                digits.append(int(digit))
                features.append(np.asarray(feature, dtype=np.float32))
                # n += 1
                # if n == 4:
                #     break

        self.categories = set(digits)
        self.datacount = len(digits)

        self.digits = np.asarray(digits, dtype=np.int32)
        self.features = np.asarray(features, dtype=np.float32)


    def __call__(self, img, k=3):
        # img isinstance of np.ndarray

        h0 = hog(img)

        dists = np.linalg.norm(self.features - h0, axis=1)

        sarg = np.argsort(dists) # sorted-arg
        topKarg = sarg[:k]

        return self.digits[topKarg].tolist(), dists[topKarg].tolist()

if __name__ == '__main__':
    import argparse
    import time
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('dictionary', type=str)
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    print('loading...')
    classifier = SSBUDigitClassifier(feature_json=args.dictionary)
    print('loaded')

    img = cv2.imread(args.input, 0)

    t = time.time()
    img = cv2.resize(img, (35, 55))

    ret = classifier(img)
    elapsed = time.time() - t

    print(ret)
    print('FPS: %f (%f s)' % (1/elapsed, elapsed, ))
