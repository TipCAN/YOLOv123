from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import cv2
import json
annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print('Running demo for *%s* results.'%(annType))
#initialize COCO ground truth api
annFile = '/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_coco.json'
cocoGt=COCO(annFile)
#initialize COCO detections api
# resFile='%s/results/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)
resFile = '/home/yjh/yolos/yolo_pytorch_v1/result/eval.json'
cocoDt = cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
# imgIds=imgIds[0:100]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
