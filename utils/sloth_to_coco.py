import os
import cv2
import json
import numpy as np

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


def sloth_to_coco(sloth_labeled_path, coco_labeled_path):
    j_sloth = json.load(open(sloth_labeled_path, 'r'))

    j_coco_out = {'images': [], 'annotations': [], 'categories': []}
    img_id = -1
    ann_id = -1
    for i in j_sloth:
        if len(i['annotations']) <= 0:
            continue
        file_name = i['filename'].split('/')[-1]
        img_id += 1
        j_coco_out['images'].append({"file_name": file_name,
                                     "id": int(img_id),
                                     "width": 320,
                                     "height": 240
                                     })
        for ann in i['annotations']:
            ann_id += 1
            j_coco_out['annotations'].append({"image_id": int(img_id),
                                              "bbox": [ann["x"], ann["y"], ann["width"], ann["height"]],
                                              "category_id": class_dict[ann["class"]],
                                              "iscrowd": 0,
                                              "id": int(ann_id),
                                              "area": 10000.0, })
    j_coco_out['categories'] = [{"id": class_dict[i], "supercategory": i, "name": i} for i in class_dict]

    with open(coco_labeled_path, 'w') as f:
        json.dump(j_coco_out, f, cls=NpEncoder, indent=4, separators=(',', ': '))
        print("Finish : " + coco_labeled_path)

sloth_labeled_path = "/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_add1.json"
coco_labeled_path = "/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_add1_coco.json"
class_dict = {'person':1}


sloth_to_coco(sloth_labeled_path, coco_labeled_path)
