from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import json
import random
File = '/home/yjh/yolos/vocdatasets/tococo/voc2007_val.json'
resFile = '/home/yjh/yolos/vocdatasets/tococo/voc2007_val_re.json'
js = json.load(open(File, 'r'))

for i in js['annotations']:
    del i["area"]
    del i["iscrowd"]
    del i["id"]
    i['score'] = random.random()
js = js['annotations']
json.dump(js, open(resFile, 'w'),indent=4)  # indent=4 更加美观显示