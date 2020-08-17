import cv2
import numpy as np
import random
import torch
import torchvision

def BGR2RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def BGR2HSV(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def HSV2BGR(img):
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def Resize(img, boxes, scale):
    w_ratio = scale[0] / img.shape[1]
    h_ratio = scale[1] / img.shape[0]
    boxes[:, ::2] *= w_ratio
    boxes[:, 1::2] *= h_ratio
    return cv2.resize(img, scale), boxes


def RandomBrightness(img_bgr, limits=(0.5, 1.5), p=0.5):
    if random.random() < p:
        hsv = BGR2HSV(img_bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(limits[0], limits[1])
        v = v * adjust
        v = np.clip(v, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_bgr = HSV2BGR(hsv)
    return img_bgr


def RandomSaturation(img_bgr, limits=(0.5, 1.5), p=0.5):
    if random.random() < p:
        hsv = BGR2HSV(img_bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(limits[0], limits[1])
        s = s * adjust
        s = np.clip(s, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_bgr = HSV2BGR(hsv)
    return img_bgr


def RandomHue(img_bgr, limits=(0.5, 1.5), p=0.5):
    if random.random() < p:
        hsv = BGR2HSV(img_bgr)
        h, s, v = cv2.split(hsv)
        adjust = random.uniform(limits[0], limits[1])
        h = h * adjust
        h = np.clip(h, 0, 255).astype(hsv.dtype)
        hsv = cv2.merge((h, s, v))
        img_bgr = HSV2BGR(hsv)
    return img_bgr


def randomBlur(img_bgr, cernel=(5, 5), p=0.5):
    if random.random() <p:
        img_bgr = cv2.blur(img_bgr, cernel)
    return img_bgr


def randomShift(bgr, boxes, labels):
    # 平移变换
    center = (boxes[:, 2:] + boxes[:, :2]) / 2
    if random.random() < 0.5:
        height, width, c = bgr.shape
        after_shfit_image = np.zeros((height, width, c), dtype=bgr.dtype)
        after_shfit_image[:, :, :] = (104, 117, 123)  # bgr
        shift_x = random.uniform(-width * 0.2, width * 0.2)
        shift_y = random.uniform(-height * 0.2, height * 0.2)
        # print(bgr.shape,shift_x,shift_y)
        # 原图像的平移
        if shift_x >= 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, int(shift_x):, :] = bgr[:height - int(shift_y), :width - int(shift_x), :]
        elif shift_x >= 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), int(shift_x):, :] = bgr[-int(shift_y):, :width - int(shift_x), :]
        elif shift_x < 0 and shift_y >= 0:
            after_shfit_image[int(shift_y):, :width + int(shift_x), :] = bgr[:height - int(shift_y), -int(shift_x):, :]
        elif shift_x < 0 and shift_y < 0:
            after_shfit_image[:height + int(shift_y), :width + int(shift_x), :] = bgr[-int(shift_y):, -int(shift_x):, :]

        shift_xy = torch.FloatTensor([[int(shift_x), int(shift_y)]]).expand_as(center)
        center = center + shift_xy
        mask1 = (center[:, 0] > 0) & (center[:, 0] < width)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < height)
        mask = (mask1 & mask2).view(-1, 1)
        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if len(boxes_in) == 0:
            return bgr, boxes, labels
        box_shift = torch.FloatTensor([[int(shift_x), int(shift_y), int(shift_x), int(shift_y)]]).expand_as(boxes_in)
        boxes_in = boxes_in + box_shift
        labels_in = labels[mask.view(-1)]
        return after_shfit_image, boxes_in, labels_in
    return bgr, boxes, labels


def randomScale(img_bgr, boxes, limits=(0.5, 1.5), p=0.5):
    # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
    if random.random() < p:
        scale = random.uniform(limits[0], limits[1])
        height, width, c = img_bgr.shape
        bgr = cv2.resize(img_bgr, (int(width * scale), height))
        scale_tensor = torch.FloatTensor([[scale, 1, scale, 1]]).expand_as(boxes)
        boxes = boxes * scale_tensor
        return img_bgr, boxes
    return img_bgr, boxes


def randomCrop(img_bgr, boxes, labels, p=0.5):
    if random.random() < p:
        center = (boxes[:, 2:] + boxes[:, :2]) / 2
        height, width, c = img_bgr.shape
        h = random.uniform(0.6 * height, height)
        w = random.uniform(0.6 * width, width)
        x = random.uniform(0, width - w)
        y = random.uniform(0, height - h)
        x, y, h, w = int(x), int(y), int(h), int(w)

        center = center - torch.FloatTensor([[x, y]]).expand_as(center)
        mask1 = (center[:, 0] > 0) & (center[:, 0] < w)
        mask2 = (center[:, 1] > 0) & (center[:, 1] < h)
        mask = (mask1 & mask2).view(-1, 1)

        boxes_in = boxes[mask.expand_as(boxes)].view(-1, 4)
        if (len(boxes_in) == 0):
            return img_bgr, boxes, labels
        box_shift = torch.FloatTensor([[x, y, x, y]]).expand_as(boxes_in)

        boxes_in = boxes_in - box_shift
        boxes_in[:, 0] = boxes_in[:, 0].clamp_(min=0, max=w)
        boxes_in[:, 2] = boxes_in[:, 2].clamp_(min=0, max=w)
        boxes_in[:, 1] = boxes_in[:, 1].clamp_(min=0, max=h)
        boxes_in[:, 3] = boxes_in[:, 3].clamp_(min=0, max=h)

        labels_in = labels[mask.view(-1)]
        img_croped = img_bgr[y:y + h, x:x + w, :]
        return img_croped, boxes_in, labels_in
    return img_bgr, boxes, labels

def subMean(img_bgr, mean=(123, 117, 104)):
    mean = np.array(mean, dtype=np.float32)
    img_bgr = img_bgr - mean
    return img_bgr

def normalization(img_bgr, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return (img_bgr-mean)/std

def random_flip(img, boxes, p=0.5):
    if random.random() < p:
        im_lr = np.fliplr(img).copy()
        h, w, _ = img.shape
        # xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]-boxes[:, 2]-1
        boxes[:, 0] = xmax
        # boxes[:, 2] = xmax
        return im_lr, boxes
    return img, boxes


def random_bright(img, delta=16, alpha_H=0.3):
    alpha = random.random()
    if alpha > alpha_H:
        im = img * alpha + random.randrange(-delta, delta)
        im = im.clip(min=0, max=255).astype(np.uint8)
    return img
