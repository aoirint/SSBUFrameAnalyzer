import argparse
import os
import cv2
import time
from SSBUDigitClassifier import SSBUDigitClassifier
from SSBUNameRecognizer import SSBUNameRecognizer
from SSBUCharaClassifier import SSBUCharaClassifier
from SSBUStockClassifier import SSBUStockClassifier
from SSBUFrameAnalyzer import SSBUFrameAnalyzer
import threading

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--digit_dictionary', type=str, default='../digit_dictionary_v1.json')
parser.add_argument('-c', '--chara_dictionary', type=str, default='../chara_dictionary_v3.json')
parser.add_argument('-s', '--stock_dictionary', type=str, default='../stock_dictionary_v3.json')
parser.add_argument('-f', '--fighter_num', type=int, default=2) # TODO: predict
parser.add_argument('-p', '--capture_device', type=int, default=None)
parser.add_argument('-i', '--video', type=str, default=None)
parser.add_argument('--dump', action='store_true')
parser.add_argument('-o', '--outdir', type=str, default='./')
parser.add_argument('--sync', action='store_true')
args = parser.parse_args()

assert (args.capture_device is not None) ^ (args.video is not None)

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

print('loading stock classifier...')
stock_classifier = SSBUStockClassifier(feature_json=args.stock_dictionary)
print('loaded stock classifier')

analyzer = SSBUFrameAnalyzer(digit_classifier=digit_classifier, name_recognizer=name_recognizer, chara_classifier=chara_classifier, stock_classifier=stock_classifier)


if args.capture_device:
    fps = 30

    cap = cv2.VideoCapture(args.capture_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, fps)
else:
    cap = cv2.VideoCapture(args.video)

if args.dump:
    outfile = '%f.avi' % time.time()
    outpath = os.path.join(args.outdir, outfile)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(outpath, fourcc, fps, (1280, 720))

    print('Dump mode: %s' % outpath)
else:
    print('No Dump mode')

task = None
def do_task():
    global task

    t = time.time()

    ret = analyzer(frame, fighter_num=args.fighter_num)
    elapsed = time.time() - t

    print('-' * 40)
    print(ret)
    print('Task FPS: %f (%f s)' % (1/elapsed, elapsed, ))

    task = None

while True:
    ts = time.time()
    ret, frame = cap.read()

    if task is None:
        task = threading.Thread(target=do_task)
        task.start()
    if args.sync:
        task.join()

    if args.dump:
        out.write(frame)

    # print(frame.shape)
    cv2.imshow('img', frame)
    elapsed = time.time() - ts
    print('Camera FPS: %f (%f s)' % (1/elapsed, elapsed, ))

    c = cv2.waitKey(1)
    if c & 0xFF == ord('q'):
        break

cap.release()
if args.dump:
    out.release()
cv2.destroyAllWindows()
