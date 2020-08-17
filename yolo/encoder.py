import numpy as np

def yolo_encoder(box_list,ceil_size,box_num,cls_num):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''
    w,h = ceil_size
    box_list.resize(ceil_size)
    labels = box_list.get_field('labels')

    bb_class = np.zeros((cls_num,h, w))
    bb_response = np.zeros((box_num,h, w))
    bb_boxes = np.zeros((box_num*4,h, w))

    #TODO avoid loop
    for gt,l in zip(box_list.box,labels):
        local_x = min(int(round((gt[2] + gt[0]) / 2)),w-1)
        local_y = min(int(round((gt[3] + gt[1]) / 2)),h-1)

        for j in range(box_num):
            bb_response[j, local_y, local_x] = 1
            bb_boxes[j * 4 + 0, local_y, local_x] = (gt[2] + gt[0])/2
            bb_boxes[j * 4 + 1, local_y, local_x] = (gt[3] + gt[1])/2
            bb_boxes[j * 4 + 2, local_y, local_x] = np.sqrt(max((gt[2] - gt[0]),0.01))
            bb_boxes[j * 4 + 3, local_y, local_x] = np.sqrt(max((gt[3] - gt[1]),0.01))

        bb_class[l, local_y, local_x] = 1
    boxes = (bb_class, bb_response, bb_boxes)
    return boxes


def yolov2_encoder(box_list, ceil_size, ratio, cls_num, iou_conf = 0.5,force_encoder=False):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    '''
    w, h = ceil_size
    box_list.resize(ceil_size)
    labels = box_list.get_field('labels')
    box_num = len(ratio)
    bb_class = np.zeros((cls_num * box_num, h, w))
    bb_response = np.zeros((box_num, h, w))
    bb_boxes = np.zeros((box_num * 4, h, w))

    # TODO avoid loop
    for gt, l in zip(box_list.box, labels):
        local_x = min(int((gt[2] + gt[0]) / 2), w - 1)
        local_y = min(int((gt[3] + gt[1]) / 2), h - 1)

        gt_w = gt[2] - gt[0]
        gt_h = gt[3] - gt[1]

        max_iou = 0
        obj_idx = 0
        for j in range(box_num):
            anchor_w, anchor_h = ratio[j]
            area_anchor = anchor_w * anchor_h
            area_ground_truth = gt_w * gt_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            iou = inter / (area_anchor + area_ground_truth - inter)
            if iou > max_iou:
                max_iou = iou
                obj_idx = j
            if 0:#iou > iou_conf:
                bb_response[j, local_y, local_x] = 1
                bb_boxes[j * 4 + 0, local_y, local_x] = ((gt[2] + gt[0]) / 2)
                bb_boxes[j * 4 + 1, local_y, local_x] = ((gt[3] + gt[1]) / 2)
                bb_boxes[j * 4 + 2, local_y, local_x] = max((gt[2] - gt[0]), 0.01)
                bb_boxes[j * 4 + 3, local_y, local_x] = max((gt[3] - gt[1]), 0.01)
                bb_class[j * cls_num + l, local_y, local_x] = 1

        if force_encoder and gt_w > 0.5 and gt_h > 0.5:# and max_iou < iou_conf:
            bb_response[obj_idx, local_y, local_x] = 1
            bb_boxes[obj_idx * 4 + 0, local_y, local_x] = ((gt[2] + gt[0]) / 2)
            bb_boxes[obj_idx * 4 + 1, local_y, local_x] = ((gt[3] + gt[1]) / 2)
            bb_boxes[obj_idx * 4 + 2, local_y, local_x] = max((gt[2] - gt[0]), 0.01)
            bb_boxes[obj_idx * 4 + 3, local_y, local_x] = max((gt[3] - gt[1]), 0.01)
            bb_class[obj_idx * cls_num + l, local_y, local_x] = 1

    boxes = (bb_class, bb_response, bb_boxes)

    return boxes



def yolov2_encoder_index(box_list, ceil_size, ratio, iou_conf = 0.5,force_encoder=False):
    '''
    pred_cls = [C,S,S]
    pred_response = [2,S,S]
    pred_bboxes = [4*2,S,S]
    ceil_x , ceil_y , box_idx , xyxy, cls
    '''
    w, h = ceil_size
    box_list.resize(ceil_size)
    labels = box_list.get_field('labels')
    box_num = len(ratio)
    result = []

    # TODO avoid loop
    for gt, l in zip(box_list.box, labels):

        local_x = min(int((gt[2] + gt[0]) / 2), int(w - 1))
        local_y = min(int((gt[3] + gt[1]) / 2), int(h - 1))
        gt_w = gt[2] - gt[0]
        gt_h = gt[3] - gt[1]

        max_iou = 0
        obj_idx = 0
        for j in range(box_num):
            anchor_w, anchor_h = ratio[j]
            area_anchor = anchor_w * anchor_h
            area_ground_truth = gt_w * gt_h
            inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
            iou = inter / (area_anchor + area_ground_truth - inter)
            if iou > max_iou:
                max_iou = iou
                obj_idx = j

            if iou > iou_conf:
                ofx = ((gt[2] + gt[0]) / 2) - local_x
                ofy = ((gt[3] + gt[1]) / 2) - local_y
                bw = max((gt[2] - gt[0]), 0.01)
                bh = max((gt[3] - gt[1]), 0.01)
                data = [ofx,ofy,bw,bh,l]
                result.append([local_x,local_y, j, data])

        if force_encoder and max_iou < iou_conf:
            ofx = ((gt[2] + gt[0]) / 2) - local_x
            ofy = ((gt[3] + gt[1]) / 2) - local_y
            bw = max((gt[2] - gt[0]), 0.01)
            bh = max((gt[3] - gt[1]), 0.01)
            data = [ofx, ofy, bw, bh, l]
            result.append([local_x, local_y, obj_idx, data])

    return result