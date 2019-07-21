import argparse
import os
import cv2
import time
from SSBUDigitClassifier import SSBUDigitClassifier
from SSBUNameRecognizer import SSBUNameRecognizer
from SSBUCharaClassifier import SSBUCharaClassifier
from SSBUFrameAnalyzer import SSBUFrameAnalyzer
import threading

parser = argparse.ArgumentParser()
parser.add_argument('outdir', type=str)
parser.add_argument('-d', '--digit_dictionary', type=str, default='../digit_dictionary.json')
parser.add_argument('-c', '--chara_dictionary', type=str, default='../chara_dictionary.json')
parser.add_argument('-f', '--fighter_num', type=int, default=2) # TODO: predict
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


print('loading digit classifier...')
digit_classifier = SSBUDigitClassifier(feature_json=args.digit_dictionary)
print('loaded digit classifier')

print('loading name recognizer...')
name_recognizer = SSBUNameRecognizer()
print('loaded name recognizer')

print('loading chara classifier...')
chara_classifier = SSBUCharaClassifier(feature_json=args.chara_dictionary)
print('loaded chara classifier')

analyzer = SSBUFrameAnalyzer(digit_classifier=digit_classifier, name_recognizer=name_recognizer, chara_classifier=chara_classifier)


fps = 30

cap = cv2.VideoCapture(2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, fps)

outfile = '%f.avi' % time.time()
outpath = os.path.join(args.outdir, outfile)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(outpath, fourcc, fps, (1280, 720))

task = None
def do_task():
    global task

    t = time.time()

    ret = analyzer(frame, fighter_num=args.fighter_num)
    elapsed = time.time() - t

    print('-' * 40)
    print(ret)
    print('FPS: %f (%f s)' % (1/elapsed, elapsed, ))

    task = None

while True:
    ret, frame = cap.read()

    out.write(frame)

    # print(frame.shape)
    cv2.imshow('img', frame)

    if task is None:
        task = threading.Thread(target=do_task)
        task.start()

    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
