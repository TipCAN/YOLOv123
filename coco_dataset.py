#encoding:utf-8
#
#created by xiongzihua
#
'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
import cv2
#import matplotlib.pyplot as plt
import glob
import json
import time
import torchvision
from collections import defaultdict
import coco_transforms
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.utils import decoder, draw_debug_rect, cv_resize
# CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
#              'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

class cocoDataset(data.Dataset):
    def __init__(self, imgdirs_list, annfiles_list, trainorval, transforms_pipeline, resizescale=(448, 448)):
        print('data init')
        self.imgdirs_list = imgdirs_list
        self.anns_list = annfiles_list
        self.trainorval = trainorval
        self.resizescale = resizescale
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB
        self.transforms_pipeline = transforms_pipeline
        self.dataset_list = []
        for imgdir, annfile in zip(self.imgdirs_list, self.anns_list):
            print('handle dataset:\n\t'+imgdir+'\n\t'+annfile)
            annfile_json = json.load(open(annfile, 'r'))
            images = annfile_json['images']
            annotations = annfile_json['annotations']
            ann_dicts = {}
            for ann in annotations:
                if ann['image_id'] not in ann_dicts.keys():
                    ann_dicts[ann['image_id']] = []
                ann_dicts[ann['image_id']].append(ann)
            for img in images:
                img['file_name'] = os.path.join(imgdir, img['file_name'])
                if img['id'] in ann_dicts.keys():
                    anns = ann_dicts[img['id']]
                else:
                    continue
                image_ann = {'image_info':img, 'ann':anns}
                self.dataset_list.append(image_ann)
        self.num_samples = len(self.dataset_list)
        print('There are %d pics in datasets.'%(self.num_samples))

    def __getitem__(self, idx):
        dataget = self.dataset_list[idx]
        anns = dataget['ann']
        bboxes = torch.zeros(len(anns), 4)
        labels = torch.zeros(len(anns))
        img_org = cv2.imread(dataget['image_info']['file_name'])
        for i, ann in enumerate(anns):
            bboxes[i] = torch.tensor(ann['bbox'])
            labels[i] = torch.tensor(ann['category_id'])
        itemdata = {'img_name': dataget['image_info']['file_name'],
                    'img_id': dataget['image_info']['id'],
                    'img_org': img_org,
                    'img': img_org,
                    'bboxes': bboxes,
                    'labels': labels}
        if self.trainorval == 'train':
            itemdata = self.pipelines(itemdata, self.transforms_pipeline)
            # img_result = torch.tensor(itemdata['img']).permute(2, 0, 1)
            img_result = itemdata['img']
            Target_tensor = ToTarget(itemdata)#.permute(2, 0, 1)

            # boxes, clss, confs = decoder(Target_tensor, grid_num=14, gt=True)
            # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            # un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            # img = un_normal_trans(img_result)
            # draw_debug_rect(img.squeeze(0).permute(1, 2 ,0), boxes, clss, confs)

            return [img_result, Target_tensor]
        else:
            itemdata = self.pipelines(itemdata, self.transforms_pipeline)
            # img_result = torch.tensor(itemdata['img']).permute(2, 0, 1)
            img_result = itemdata['img']
            img_id = itemdata['img_id']
            org_size_yx = torch.tensor(itemdata['img_org'].shape[:2])
            return [img_result, img_id, org_size_yx]

    def pipelines(self, data, transforms_pipeline):
        for t in transforms_pipeline:
            if t == 'BGR2RGB':
                data['img'] = coco_transforms.BGR2RGB(data['img'])
            if t == 'BGR2HSV':
                data['img'] = coco_transforms.BGR2HSV(data['img'])
            if t == 'HSV2BGR':
                data['img'] = coco_transforms.HSV2BGR(data['img'])
            if t == 'Resize':
                data['img'], data['bboxes'] = coco_transforms.Resize(img=data['img'], boxes=data['bboxes'], scale=(448, 448))
            if t == 'RandomBrightness':
                data['img'] = coco_transforms.RandomBrightness(img_bgr=data['img'], limits=(0.5, 1.5), p=0.5)
            if t == 'RandomSaturation':
                data['img'] = coco_transforms.RandomSaturation(img_bgr=data['img'], limits=(0.5, 1.5), p=0.5)
            if t == 'RandomHue':
                data['img'] = coco_transforms.RandomHue(img_bgr=data['img'], limits=(0.5, 1.5), p=0.5)
            if t == 'randomBlur':
                data['img'] = coco_transforms.randomBlur(img_bgr=data['img'], cernel=(5, 5), p=0.5)
            # if t == 'randomShift':
            #     data['img'] = transforms.randomShift(img_bgr=data['img'], limits=[0.5, 1.5], p=0.5)
            # if t == 'randomScale':
            #     data['img'], data['bboxes'] = transforms.randomScale(img_bgr=data['img'], boxes=data['bboxes'], limits=[0.5, 1.5], p=0.5)
            # if t == 'randomCrop':
            #     data['img'] = transforms.randomCrop(img_bgr=data['img'], limits=[0.5, 1.5], p=0.5)
            if t == 'random_flip':
                data['img'], data['bboxes'] = coco_transforms.random_flip(img=data['img'], boxes=data['bboxes'], p=0.5)
            # if t == 'random_bright':
            #     data['img'] = transforms.random_bright(img=data['img'], delta=16, alpha_H=0.3)
            if t == 'subMean':
                data['img'] = coco_transforms.subMean(img_bgr=data['img'], mean=(123, 117, 104))  # RGB
            if t == 'normalization':
                # data['img'] = coco_transforms.normalization(img_bgr=data['img'])  # RGB
                t_Normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                data['img'] = t_Normalize(data['img'])  # RGB
            if t == 'to_tensor':
                t_totensor = torchvision.transforms.ToTensor()
                data['img'] = t_totensor(data['img'])
        return data

    def __len__(self):
        return self.num_samples


def ToTarget(data, cell_num=14, class_num=20, B=2):
    Target_tensor = torch.zeros(cell_num, cell_num, B*5+class_num)
    labels = data['labels']
    bboxes_ = data['bboxes']
    bboxes = bboxes_.clone()
    bboxes[:, 0::2] /= data['img'].shape[2]
    bboxes[:, 1::2] /= data['img'].shape[1]
    bboxes = torch.cat((bboxes, torch.ones(len(bboxes),1)),1)
    cell_scale = 1/cell_num
    # print(bboxes)
    for box_, label in zip(bboxes, labels):
        box = box_.clone()
        cx, cy = box[0]+box[2]/2, box[1]+box[3]/2
        cellx, celly = int((cx/cell_scale).ceil()-1), int((cy/cell_scale).ceil()-1)
        box[0], box[1] = cx*cell_num-cellx, cy*cell_num-celly
        for b in range(B):
            Target_tensor[celly, cellx, b*5:b*5+5] = box
        # print(box)
        Target_tensor[celly, cellx, B*5+int(label)-1] = 1
    # boxes, clss, confs = decoder(Target_tensor, grid_num=14, gt=True)

    return Target_tensor


def ToBbox(pred):
    grid_num = 14
    S = grid_num
    B = 2
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    mask_1 = contain1>0.1
    mask_2 = contain2>0.1
    mask_ = (mask_1 | mask_2).squeeze(-1)
    pred_obj = pred[mask_]
    row = torch.arange(14, dtype=torch.float).unsqueeze(-1).expand_as(mask_)[mask_].unsqueeze(-1)
    col = torch.arange(14, dtype=torch.float).unsqueeze(0).expand_as(mask_)[mask_].unsqueeze(-1)
    colrow = torch.cat((col, row), dim=1)
    pred_obj_xyxy = pred_obj.clone()
    for b in range(B):
        pred_obj_xyxy[:, b * 5:b * 5 + 2] = pred_obj[:, b * 5:b * 5 + 2] / S - 0.5 * pred_obj[:, b * 5 + 2:b * 5 + 4] + colrow / S
        pred_obj_xyxy[:, b * 5 + 2:b * 5 + 4] = pred_obj[:, b * 5:b * 5 + 2] / S + 0.5 * pred_obj[:, b * 5 + 2:b * 5 + 4] + colrow / S

    boxes = torch.zeros(pred_obj_xyxy.size(0),4)
    cls_indexs = torch.zeros(pred_obj_xyxy.size(0))
    probs = torch.zeros(pred_obj_xyxy.size(0))
    for i in range(pred_obj_xyxy.size(0)):
        pred_grid = pred_obj_xyxy[i]
        max_conf, max_conf_index = torch.max(pred_grid[[b*5+4 for b in range(B)]], 0)
        max_prob, cls_index = torch.max(pred_grid[B*5:], 0)
        # print(max_conf_index)
        box = pred_grid[max_conf_index*5:max_conf_index*5+4]
        conf = max_conf*max_prob
        cls = cls_index
        boxes[i] = box
        probs[i] = conf
        cls_indexs[i] = cls

    keep = nms(boxes,probs, 0.2)
    return boxes[keep].numpy(),cls_indexs[keep].numpy(),probs[keep].numpy()


def nms(bboxes,scores,threshold=0.5):
    '''
    bboxes(tensor) [N,4]
    scores(tensor) [N,]
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]
    areas = (x2-x1) * (y2-y1)

    _,order = scores.sort(0,descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (ovr<=threshold).nonzero().squeeze(-1)
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)

class eval_results():
    def __init__(self, json_path, val_annfiles_list):
        self.json_path = json_path
        self.val_annfiles_list = val_annfiles_list
        self.j_result = []


    def add_item(self, result_item, data_batch):
        boxes = result_item[0]
        cls_indexs = result_item[1]
        probs = result_item[2]

        img_result, img_id, org_size_yx = data_batch
        img_id = img_id.numpy()[0]
        org_size_yx = org_size_yx.numpy()
        for box, category_id, score in zip(boxes, cls_indexs, probs):
            item = {}
            box[2:] -= box[:2]
            box[:2] *= org_size_yx[0][::-1]
            box[2:] *= org_size_yx[0][::-1]
            item['image_id'] = img_id
            item['bbox'] = box.tolist()
            item['category_id'] = int(category_id+1)
            item['score'] = float(score)
            self.j_result.append(item)

    def eval(self):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(NpEncoder, self).default(obj)
        if not os.path.exists(self.json_path.replace(self.json_path.split('/')[-1], '')):
            os.makedirs(self.json_path.replace(self.json_path.split('/')[-1], ''))
        with open(self.json_path, 'w') as f:
            json.dump(self.j_result, f, cls=NpEncoder, indent=4, separators=(',', ': '))

        annType = ['segm', 'bbox', 'keypoints']
        annType = annType[1]  # specify type here
        annFile = self.val_annfiles_list[0]
        cocoGt=COCO(annFile)
        resFile = self.json_path
        cocoDt = cocoGt.loadRes(resFile)
        imgIds=sorted(cocoGt.getImgIds())
        # imgIds=imgIds[0:100]

        # running evaluation
        cocoEval = COCOeval(cocoGt,cocoDt,annType)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()


def make_val_map(annFile = '/home/yjh/yolos/data/vocdata/tococo/voc2007_val.json', idval_num=None):
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    nms = set([cat['supercategory'] for cat in cats])
    catIds = coco.getCatIds(catNms=nms)
    imgIds = coco.getImgIds(imgIds=[i for i in range(len(coco.imgs))])
    img = coco.loadImgs(imgIds)
    val_map =defaultdict(list)
    img_size = (448., 448.)
    for idx, i in enumerate(img):
        if idval_num:
            if idx >= idval_num:
                break
        w, h = float(i['width']), float(i['height'])
        annIds = coco.getAnnIds(imgIds=i['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            box = np.array(ann['bbox'])
            box[2:] = box[:2]+box[2:]
            box[::2] = box[::2]/w*img_size[0]-1
            box[1::2] = box[1::2]/h*img_size[1]-1
            val_map[i['file_name'].split('.')[0], CLASSES[ann['category_id'] - 1]].append(list(box))
    return val_map

def main():
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from utils.utils import decoder, draw_debug_rect, cv_resize
    train_pipelines = ['Resize',
                       'RandomBrightness',
                       'RandomSaturation',
                       'RandomHue',
                       'randomBlur',
                       'random_flip',
                       'to_tensor',
                       'normalization',
                       ]
    imgdirs_list = ['../data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/']
    annfiles_list = ['../data/vocdata/tococo/voc2007_val.json']
    train_dataset = cocoDataset(imgdirs_list=imgdirs_list,
                                annfiles_list=annfiles_list,
                                trainorval='train',
                                transforms_pipeline=train_pipelines)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    train_iter = iter(train_loader)
    for i in range(100):
        data_batch = next(train_iter)
        img_result, Target_tensor = data_batch
        boxes, clss, confs = decoder(Target_tensor, grid_num=14, gt=True)
        print(boxes, clss, confs)
        # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        # un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        # img = un_normal_trans(img_result.squeeze(0))
        # draw_debug_rect(img.squeeze(0).permute(1, 2 ,0), boxes, clss, confs)
        time.sleep(0.5)

if __name__ == '__main__':

    # seed = 999
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
    # main()
    make_val_map()

