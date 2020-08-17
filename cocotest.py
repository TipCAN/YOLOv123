from pycocotools.coco import COCO
import numpy as np
import cv2

def draw_rectangle(anns, image):
    coordinates = []
    for j in range(len(anns)):
        coordinate = []
        coordinate.append(anns[j]['bbox'][0])
        coordinate.append(anns[j]['bbox'][1] + anns[j]['bbox'][3])
        coordinate.append(anns[j]['bbox'][0] + anns[j]['bbox'][2])
        coordinate.append(anns[j]['bbox'][1])
        # print(coordinate)
        coordinates.append(coordinate)
    for coordinate in coordinates:
        left = np.rint(coordinate[0])
        right = np.rint(coordinate[1])
        top = np.rint(coordinate[2])
        bottom = np.rint(coordinate[3])
        # 左上角坐标, 右下角坐标
        image = cv2.rectangle(image,
                              (int(left), int(right)),
                              (int(top), int(bottom)),
                              (0, 255, 0),
                              2)
    return image

annFile='/home/yjh/yolos/data/vocdata/tococo/voc2007_train.json'
imgRoot = '/home/yjh/yolos/data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/'
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=nms)
imgIds = coco.getImgIds(catIds=catIds)
imgIds = coco.getImgIds(imgIds=[100])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = cv2.imread(imgRoot+img['file_name'])[:,:,::-1]
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
I = draw_rectangle(anns, I)
cv2.imshow('I', I)
cv2.waitKey()
