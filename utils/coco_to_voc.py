import os
from tqdm import tqdm
import argparse
import json
import random

def convert_json(json_path, image_path, out_path = './labels/'):
    label_j = json.load(open(json_path, 'r'))
    print(label_j.keys())
    label_list = []
    image_list = []
    for image in label_j['images']:
        id = image['id']
        w = image['width']
        h = image['height']
        name = image['file_name']
        image_list.append(os.path.join(image_path,name))
        if os.path.exists(out_path) is False:
            os.makedirs(out_path)
        out_file_path = os.path.abspath(os.path.join(out_path, name.split('.')[0]+'.txt'))
        out_file = open(out_file_path, 'w')
        label_list.append(out_file_path)
        for ann in label_j['annotations']:
            if ann['image_id']==id:
                cls_id = ann['category_id']
                bb = (max(1, ann['bbox'][0]), max(1, ann['bbox'][1]),
                      min(w - 1, ann['bbox'][0]+ann['bbox'][2]), min(h - 1, ann['bbox'][1]+ann['bbox'][3]))
                out_file.write(str(cls_id)+ ','+",".join([str(a) for a in bb])+'\n')
    print('finish convert_json.')
    return image_list, label_list

def make_lists(image_list,save_dir, train_rate=0.9):
    train_file = open(os.path.join(save_dir, 'train.txt'), 'w')
    val_file = open(os.path.join(save_dir, 'val.txt'), 'w')
    for image_path in image_list:
        if random.random()>train_rate:
            val_file.write(image_path.split('/')[-1].split('.')[0]+'\n')
        else:
            train_file.write(image_path.split('/')[-1].split('.')[0]+'\n')
    print('finish make_lists.')
if __name__ == "__main__":
    image_list, label_list = convert_json(
        '/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_add1_coco.json',
        '/home/yjh/catkin_ws/src/Firmware/yjhworlds/images2/',
        '/home/yjh/catkin_ws/src/Firmware/yjhworlds/VOC/labels/')
    make_lists(image_list, '/home/yjh/catkin_ws/src/Firmware/yjhworlds/VOC/')