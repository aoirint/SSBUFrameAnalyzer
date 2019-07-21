from skimage.feature import hog
import numpy as np
import json

class SSBUCharaClassifier:
    def __init__(self, feature_json):
        self.feature_json = feature_json

        with open(feature_json, 'r') as fp:
            data = json.load(fp)

        chara_id2name = sorted(list(data.keys()))

        charas = []
        features = []
        for chara_id, chara_name in enumerate(chara_id2name):
            fts = data[chara_name]

            # n = 0
            for feature in fts:
                charas.append(int(chara_id))
                features.append(np.asarray(feature, dtype=np.float32))
                # n += 1
                # if n == 4:
                #     break

        self.categories = set(charas)
        self.datacount = len(charas)

        self.chara_id2name = chara_id2name
        self.charas = np.asarray(charas, dtype=np.int32)
        self.features = np.asarray(features, dtype=np.float32)


    def __call__(self, img, k=3):
        # img isinstance of np.ndarray
        # print(img.shape)
        assert len(img.shape) == 2 # gray
        assert img.shape[1] == 110 and img.shape[0] == 110
        h0 = hog(img)

        dists = np.linalg.norm(self.features - h0, axis=1)

        sarg = np.argsort(dists) # sorted-arg
        topKarg = sarg[:k]

        charas = self.charas[topKarg].tolist()
        names = []
        for i in range(len(charas)):
            chara_id = int(charas[i])
            name = self.chara_id2name[chara_id]
            names.append(name)

        return names, dists[topKarg].tolist()

if __name__ == '__main__':
    import argparse
    import time
    import cv2

    parser = argparse.ArgumentParser()
    parser.add_argument('dictionary', type=str)
    parser.add_argument('input', type=str)
    args = parser.parse_args()

    print('loading...')
    classifier = SSBUCharaClassifier(feature_json=args.dictionary)
    print('loaded')

    img = cv2.imread(args.input, 0)

    t = time.time()
    # img = cv2.resize(img, (110, 110))

    ret = classifier(img)
    elapsed = time.time() - t

    print(ret)
    print('FPS: %f (%f s)' % (1/elapsed, elapsed, ))
