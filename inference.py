import os
import sys
import random
import torch
import torchvision.models.mobilenet
import time
import tqdm
import glob
import cv2
import numpy as np
import coco_transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from coco_dataset import cocoDataset, ToBbox
from loss_yolov1 import loss_yolov1
from models.resnet_YOLO import resnet50
from models.mobilenetv2_YOLO import mobilenet_v2
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
def inference_once(SS, img):
    Result = ToBbox(SS.cpu())
    Bboxresult = Result[0]
    Clsresult = Result[1]
    Confresult = Result[2]
    img_resize = img.copy()
    for box, cls, conf in zip(Bboxresult, Clsresult, Confresult):
        box[:2] = box[:2] * img_resize.shape[:2][::-1]
        box[2:] = box[2:] * img_resize.shape[:2][::-1]
        img_resize = cv2.rectangle(img_resize, tuple(box[:2]), tuple(box[2:]), (255, 0, 0), 3)
        cls_conf_str = '%s:%.2f' % (CLASSES[int(cls)], conf)
        img_resize = cv2.putText(img_resize, cls_conf_str, tuple(box[:2]),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 255, 255),2)
    cv2.imshow('imgorg', img_resize)
    cv2.waitKey(0)

def main():
    # model_path = './save_person1/epoch_500.pth'
    model_path = './save_person4_m/epoch_250.pth'

    # model = resnet50()
    model = mobilenet_v2()
    model.load_state_dict(torch.load(model_path).state_dict())
    model = model.cuda().eval().half()
    img_root = '/home/yjh/catkin_ws/src/Firmware/yjhworlds/images'
    # img_root = '/home/yjh/yolos/pytorch-YOLO-v1'
    img_paths = glob.glob(os.path.join(img_root,'*.jpg'))
    img_paths.sort()
    for img_path in img_paths:
        # img_org = cv2.imread('/home/yjh/catkin_ws/src/Firmware/yjhworlds/images/50.jpg')
        img_org = cv2.imread(img_path)
        img_resize = cv2.resize(img_org, (448, 448))
        t_totensor = torchvision.transforms.ToTensor()
        img = t_totensor(img_resize)
        t_Normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = t_Normalize(img)  # RGB
        img = img.unsqueeze(0)
        # img = coco_transforms.to(img_resize)
        # img = coco_transforms.subMean(img)
        # img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
        SS = model(img.cuda().half())
        inference_once(SS.float(), img_resize)
if __name__ == '__main__':
    main()
