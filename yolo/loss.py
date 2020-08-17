import torch
import torch.nn.functional as F
import torch.nn as nn
from yolo.encoder import yolo_encoder,yolov2_encoder,yolov2_encoder_index
from yolo._utils import *


class yolov1_loss(nn.Module):
    def __init__(self, l_coord, l_obj, l_noobj):
        super(yolov1_loss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj

    def _prepare_target(self,meta,ceil_size,bbox_num,cls_num,device):
        target_cls = []
        target_obj = []
        target_box = []
        for target in meta:
            t = target['boxlist']
            t.resize(ceil_size)
            cls,obj,box = yolo_encoder(t,ceil_size,bbox_num,cls_num)
            target_cls.append(torch.from_numpy(cls).unsqueeze(dim=0).float())
            target_obj.append(torch.from_numpy(obj).unsqueeze(dim=0).float())
            target_box.append(torch.from_numpy(box).unsqueeze(dim=0).float())
        target_cls = torch.cat(target_cls).to(device)
        target_obj = torch.cat(target_obj).to(device)
        target_box = torch.cat(target_box).to(device)
        return target_cls,target_obj,target_box

    def offset2box(self,box):
        box[:, 0] = box[:, 0]
        box[:, 1] = box[:, 1]
        box[:, 2] = (box[:, 2] * box[:, 2])
        box[:, 3] = (box[:, 3] * box[:, 3])
        """
        cxcywh -> xywh -> xyxy
        """
        box[:, 0] = box[:, 0] - box[:, 2] / 2
        box[:, 1] = box[:, 1] - box[:, 3] / 2
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]

        return box

    def forward(self,pred,meta):

        # pred_cls, pred_response, pred_bboxes = pred
        pred = pred.permute(0,3,1,2)
        meta = meta.permute(0,3,1,2)

        pred_cls = pred[:,10:,:,:]
        pred_response = torch.cat([pred[:,4,:,:].unsqueeze(1),pred[:,9,:,:].unsqueeze(1)],1)
        pred_bboxes = torch.cat([pred[:,0:4,:,:],pred[:,5:9,:,:]],1)
        label_cls = meta[:,10:,:,:]
        label_response = torch.cat([meta[:,4,:,:].unsqueeze(1),meta[:,9,:,:].unsqueeze(1)],1)
        label_bboxes = torch.cat([meta[:,0:4,:,:],meta[:,5:9,:,:]],1)

        B_size,cls_num,h,w = pred_cls.shape
        bbox_num = pred_response.shape[1]


        ceil_size = (w,h)
        # label_cls, label_response, label_bboxes = self._prepare_target(meta,ceil_size,bbox_num,cls_num,device)

        device = pred_cls.get_device()
        label_cls = label_cls.to(device)
        label_response = label_response.to(device)
        label_bboxes = label_bboxes.to(device)

        with torch.no_grad():
            tmp_response = label_response.sum(dim=1).unsqueeze(dim=1)
            k = (tmp_response>0.9).sum()
            x_list,y_list,c_list,b_list = get_kp_torch_batch(tmp_response,conf=0.5,topk=int(k))

        t_responses = label_response[b_list, :, y_list, x_list]
        p_responses = pred_response[b_list, :, y_list, x_list]

        t_boxes = label_bboxes[b_list, :, y_list, x_list]
        p_boxes = pred_bboxes[b_list, :, y_list, x_list]

        t_classes = label_cls[b_list, :, y_list, x_list]
        p_classes = pred_cls[b_list, :, y_list, x_list]

        loss_pos_cls = F.mse_loss(p_classes,t_classes, reduction='sum')


        t_offset = t_boxes.view(-1, 4)
        p_offset = p_boxes.view(-1, 4)
        with torch.no_grad():
            t_box = self.offset2box(t_offset.clone().float()).to(device)
            p_box = self.offset2box(p_offset.clone().float()).to(device)
            iou = compute_iou(t_box, p_box).view(-1,bbox_num)

        idx = iou.argmax(dim=1)
        idx = idx.unsqueeze(dim=1)

        loss_pos_response = F.mse_loss(p_responses.gather(1,idx),iou.gather(1,idx), reduction='sum')

        idx = idx.unsqueeze(dim=1)
        p_boxes = p_boxes.view(-1, bbox_num, 4)
        t_boxes = t_boxes.view(-1, bbox_num, 4)
        off_idx = idx.repeat(1,1,4)
        loss_pos_offset = F.mse_loss(p_boxes.gather(1,off_idx), t_boxes.gather(1,off_idx), reduction='sum')



        neg_mask = label_response < 1
        neg_pred = pred_response[neg_mask]
        neg_target = label_response[neg_mask]

        loss_neg_response = F.mse_loss(neg_pred, neg_target, reduction='sum') / B_size * self.l_noobj
        loss_pos_response = loss_pos_response / B_size * self.l_obj
        loss_pos_offset = loss_pos_offset / B_size * self.l_coord
        loss_pos_cls = loss_pos_cls / B_size
        loss_sum = loss_neg_response+loss_pos_response+loss_pos_offset+loss_pos_cls

        # return {'pObj': loss_pos_response,
        #         'nObj':loss_neg_response,
        #         'cls': loss_pos_cls,
        #         'offset': loss_pos_offset}
        return loss_sum,loss_pos_response,loss_neg_response,loss_pos_cls,loss_pos_offset



class yolov2_loss(nn.Module):
    def __init__(self, ratio,l_coord, l_obj, l_noobj , force_encoder, encoder_iou):
        super(yolov2_loss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj
        self.ratio = ratio
        self.force_encoder = force_encoder
        self.encoder_iou = encoder_iou


    def gen_anchor(self,ceil):
        anchor_xy = []
        w,h = ceil
        for r in self.ratio:
            x = torch.linspace(0, w-1, w).unsqueeze(dim=0).repeat(h, 1).unsqueeze(dim=0)
            y = torch.linspace(0, h-1, h).unsqueeze(dim=0).repeat(w, 1).unsqueeze(dim=0).permute(0, 2, 1)
            anchor_xy.append(torch.cat((x, y), dim=0).unsqueeze(dim=0))
        anchor_xy = torch.cat(anchor_xy, dim=0).view(-1, h, w)
        return anchor_xy

    def _prepare_target_loop(self,meta,ceil_size,device):
        data = []
        for target in meta:
            t = target['boxlist'].copy()
            t.resize(ceil_size)
            #print(t.box)
            print(t.get_field('labels'))
            #print()
            result = yolov2_encoder_index(t, ceil_size, self.ratio,self.encoder_iou, self.force_encoder)
            data.append(result)
        anchor_xy = self.gen_anchor(ceil_size).unsqueeze(dim=0).repeat(len(meta),1,1,1).to(device)
        return data,anchor_xy


    def bbox_ious(self, boxes1, boxes2):
        """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.
        Args:
            boxes1 (torch.Tensor): List of bounding boxes
            boxes2 (torch.Tensor): List of bounding boxes
        Note:
            List format: [[xc, yc, w, h],...]
        """
        b1_len = boxes1.size(0)
        b2_len = boxes2.size(0)

        b1x1, b1y1 = (boxes1[:, :2]).split(1, 1)
        b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 1)).split(1, 1)
        b2x1, b2y1 = (boxes2[:, :2]).split(1, 1)
        b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 1)).split(1, 1)

        dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
        dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        unions = (areas1 + areas2.t()) - intersections
        return intersections / unions

    def forward(self,pred,meta):

        pred_cls, pred_response, pred_bboxes = pred
        device = pred_cls.get_device()

        B_size,cls_num,h,w = pred_cls.shape
        bbox_num = pred_response.shape[1]
        cls_num = int(cls_num/bbox_num)

        ceil_size = (w,h)

        target,anchor_xy = self._prepare_target_loop(meta,ceil_size,device)

        gt_response = torch.zeros_like(pred_response).to(device)
        pos_response_mask = torch.zeros_like(pred_response).bool()
        neg_response_mask = torch.zeros_like(pred_response).bool()
        loss_pos_cls = 0
        loss_pos_xy = 0
        loss_pos_wh = 0
        pos_num = 0
        for bs in range(len(target)):
            target_one = target[bs]

            for idx_x,idx_y,idx_box,target_box in target_one:
                cx = target_box[0]+idx_x
                cy = target_box[1]+idx_y
                l = target_box[4]

                target_one_offset = torch.tensor([target_box[0],target_box[1],target_box[2] ,target_box[3]]).to(device)
                pred_ceil_box = pred_bboxes[bs,:,:,:].clone().permute(1,2,0).contiguous().view(-1,4)
                anchor = anchor_xy[bs,:,:,:].clone().permute(1,2,0).contiguous().view(-1, 2)

                with torch.no_grad():
                    gt_box = torch.tensor([cx - target_box[2] /2,
                                           cy - target_box[3] /2,
                                           target_box[2],
                                           target_box[3]]).unsqueeze(dim=0)

                    gt_box = gt_box.to(device)
                    pred_box = pred_ceil_box.clone()
                    pred_box[:, :2] = pred_box[:, :2].sigmoid() + anchor - pred_box[:, 2:] / 2
                    iou = self.bbox_ious(pred_box,gt_box).view(h,w,-1).permute(2,0,1).contiguous()

                    #print(iou[idx_box,idx_y,idx_x],iou.max())
                    neg_response_mask[bs,:,:,:] = iou < 0.6
                    gt_response[bs, idx_box, idx_y, idx_x] = iou[idx_box,idx_y,idx_x]
                    pos_response_mask[bs,idx_box,idx_y,idx_x] = True

                pred_one_offset = pred_bboxes[bs, idx_box * 4:(idx_box + 1) * 4, idx_y, idx_x]
                pred_xy = pred_one_offset[:2]
                target_xy = target_one_offset[:2]
                loss_pos_xy = loss_pos_xy + F.binary_cross_entropy_with_logits(pred_xy,
                                                                               target_xy,
                                                                               reduction='sum')

                pred_wh = pred_one_offset[2:]
                target_wh = target_one_offset[2:]
                #print(pred_wh.shape,target_wh.shape,pred_xy.shape)
                loss_pos_wh = loss_pos_wh + F.mse_loss(pred_wh,target_wh, reduction='sum')

                pred_one_cls = pred_cls[bs, idx_box * cls_num:(idx_box + 1) * cls_num, idx_y, idx_x]

                target_one_cls = torch.tensor([l]).to(device)
                pred_one_cls = pred_one_cls.unsqueeze(dim=0)
                #print(target_one_cls)
                loss_pos_cls = loss_pos_cls + F.cross_entropy(pred_one_cls,target_one_cls,reduction='sum')
                #print(target_one_cls)
                pos_num += 1


        loss_pos_response = F.binary_cross_entropy_with_logits(pred_response[pos_response_mask],
                                   gt_response[pos_response_mask],reduction='sum') * self.l_obj

        neg_response_mask = neg_response_mask*(pos_response_mask==False)
        loss_neg_response = F.binary_cross_entropy_with_logits(pred_response[neg_response_mask],
                                   gt_response[neg_response_mask],reduction='sum') * self.l_noobj


        return {'pObj': loss_pos_response / B_size,
                'nObj': loss_neg_response / B_size,
                 'cls': loss_pos_cls / B_size,
                  'xy': loss_pos_xy / B_size ,
                  'wh': loss_pos_wh / B_size * self.l_coord}





class yolov2_loss_fast(nn.Module):
    def __init__(self, ratio,l_coord, l_obj, l_noobj , force_encoder, encoder_iou):
        super(yolov2_loss_fast, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj
        self.ratio = ratio
        self.force_encoder = force_encoder
        self.encoder_iou = encoder_iou

    def _prepare_target(self,meta,ceil_size,cls_num,device):
        target_cls = []
        target_obj = []
        target_box = []
        for target in meta:
            t = target['boxlist']
            t.resize(ceil_size)
            #print(t.box)
            #print(t.get_field("labels"))
            cls,obj,box = yolov2_encoder(t, ceil_size, self.ratio, cls_num,self.encoder_iou, self.force_encoder)
            target_cls.append(torch.from_numpy(cls).unsqueeze(dim=0).float())
            target_obj.append(torch.from_numpy(obj).unsqueeze(dim=0).float())
            target_box.append(torch.from_numpy(box).unsqueeze(dim=0).float())
        target_cls = torch.cat(target_cls).to(device)
        target_obj = torch.cat(target_obj).to(device)
        target_box = torch.cat(target_box).to(device)
        return target_cls,target_obj,target_box

    def offset2box(self,box):

        box[:, 0] = box[:, 0] - box[:, 2] / 2
        box[:, 1] = box[:, 1] - box[:, 3] / 2
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]

        return box

    def forward(self,pred,meta):

        pred_cls, pred_response, pred_bboxes = pred
        device = pred_cls.get_device()

        B_size,cls_num,h,w = pred_cls.shape
        bbox_num = pred_response.shape[1]
        cls_num = int(cls_num / bbox_num)

        ceil_size = (w,h)
        label_cls, label_response, label_bboxes = self._prepare_target(meta,ceil_size,cls_num,device)

        device = pred_cls.get_device()
        label_cls = label_cls.to(device)
        label_response = label_response.to(device)
        label_bboxes = label_bboxes.to(device)

        with torch.no_grad():
            tmp_response = label_response.sum(dim=1).unsqueeze(dim=1)
            k = (tmp_response>0.9).sum()
            x_list,y_list,c_list,b_list = get_kp_torch_batch(tmp_response,conf=0.5,topk=int(k))

        t_responses = label_response[b_list, :, y_list, x_list]
        p_responses = pred_response[b_list, :, y_list, x_list]

        t_offset = label_bboxes[b_list, :, y_list, x_list]
        p_offset = pred_bboxes[b_list, :, y_list, x_list]

        t_classes = label_cls[b_list, :, y_list, x_list]
        p_classes = pred_cls[b_list, :, y_list, x_list]


        t_boxes = t_offset.view(-1, 4).clone()
        p_boxes = p_offset.view(-1, 4).clone()
        loss_pos_offset = 0
        loss_pos_cls = 0
        loss_pos_response = 0
        if len(t_boxes) > 0:

            with torch.no_grad():
                t_boxes = self.offset2box(t_boxes.float()).to(device)
                p_boxes = self.offset2box(p_boxes.float()).to(device)

                #TODO fix
                iou = compute_iou(t_boxes, p_boxes)
                iou_mask = (t_boxes.mean(-1) > 0) * (iou < 0.01)
                iou[iou_mask] = 0.01
                iou = iou.view(-1,bbox_num)

            max_iou_idx = iou.argmax(dim=1)
            max_iou_idx = max_iou_idx.unsqueeze(dim=1)
            loss_pos_response = F.smooth_l1_loss(p_responses.gather(1,max_iou_idx),
                                           iou.gather(1,max_iou_idx),
                                           reduction='sum')


            idx = max_iou_idx.unsqueeze(dim=1)
            p_offset = p_offset.view(-1, bbox_num, 4)
            t_offset = t_offset.view(-1, bbox_num, 4)

            off_idx = idx.repeat(1,1,4)
            loss_pos_offset = F.smooth_l1_loss(p_offset.gather(1,off_idx),
                                         t_offset.gather(1,off_idx),
                                         reduction='sum')


            p_classes = p_classes.view(-1, bbox_num, cls_num)
            t_classes = t_classes.view(-1, bbox_num, cls_num)
            cls_idx = idx.repeat(1,1,cls_num)
            loss_pos_cls = F.mse_loss(p_classes.gather(1,cls_idx),
                                      t_classes.gather(1,cls_idx),
                                      reduction='sum')


        #TODO smooth regression
        neg_mask = label_response < 1
        neg_pred = pred_response[neg_mask]
        neg_target = label_response[neg_mask]

        loss_neg_response = F.smooth_l1_loss(neg_pred, neg_target, reduction='sum') / B_size * self.l_noobj
        loss_pos_response = loss_pos_response / B_size * self.l_obj
        loss_pos_offset = loss_pos_offset / B_size * self.l_coord
        loss_pos_cls = loss_pos_cls / B_size


        return {'pObj': loss_pos_response,
                'nObj':loss_neg_response,
                'cls': loss_pos_cls,
                'ofx': loss_pos_offset}




import math
class yolov2_loss_v2(nn.Module):
    def __init__(self, ratio,l_coord, l_obj, l_noobj , force_encoder, encoder_iou):
        super(yolov2_loss_v2, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj
        self.ratio = ratio
        self.force_encoder = force_encoder
        self.encoder_iou = encoder_iou
        self.box_num = len(ratio)

        torch.nn.CrossEntropyLoss()

    def _gen_anchor(self,ceil):
        anchor_xy = []
        anchor_wh = []
        w,h = ceil
        for r in self.ratio:
            x = torch.linspace(0, w-1, w).unsqueeze(dim=0).repeat(h, 1).unsqueeze(dim=0)
            y = torch.linspace(0, h-1, h).unsqueeze(dim=0).repeat(w, 1).unsqueeze(dim=0).permute(0, 2, 1)
            width = torch.Tensor([r[0]]).view(1, 1, 1).repeat(1, h, w)
            height = torch.Tensor([r[1]]).view(1, 1, 1).repeat(1, h, w)
            anchor_xy.append(torch.cat((x, y), dim=0).unsqueeze(dim=0))
            anchor_wh.append(torch.cat((width, height), dim=0).unsqueeze(dim=0))
        anchor_xy = torch.cat(anchor_xy, dim=0).view(-1, h, w).permute(1,2,0)
        anchor_wh = torch.cat(anchor_wh, dim=0).view(-1, h, w).permute(1,2,0)
        return anchor_xy,anchor_wh

    def bbox_ious(self, boxes1, boxes2):
        """ Compute IOU between all boxes from ``boxes1`` with all boxes from ``boxes2``.
        Args:
            boxes1 (torch.Tensor): List of bounding boxes
            boxes2 (torch.Tensor): List of bounding boxes
        Note:
            List format: [[xc, yc, w, h],...]
        """
        b1_len = boxes1.size(0)
        b2_len = boxes2.size(0)

        b1x1, b1y1 = (boxes1[:, :2]).split(1, 1)
        b1x2, b1y2 = (boxes1[:, :2] + (boxes1[:, 2:4] / 1)).split(1, 1)
        b2x1, b2y1 = (boxes2[:, :2]).split(1, 1)
        b2x2, b2y2 = (boxes2[:, :2] + (boxes2[:, 2:4] / 1)).split(1, 1)

        dx = (b1x2.min(b2x2.t()) - b1x1.max(b2x1.t())).clamp(min=0)
        dy = (b1y2.min(b2y2.t()) - b1y1.max(b2y1.t())).clamp(min=0)
        intersections = dx * dy

        areas1 = (b1x2 - b1x1) * (b1y2 - b1y1)
        areas2 = (b2x2 - b2x1) * (b2y2 - b2y1)
        unions = (areas1 + areas2.t()) - intersections
        return intersections / unions

    def _push_mask(self,box_list, ceil_size,cls_num):
        w,h = ceil_size

        bb_class = torch.zeros(( h, w,self.box_num,cls_num))
        bb_response = torch.zeros(( h, w,self.box_num,1))
        bb_boxes = torch.zeros(( h, w,self.box_num,4))

        box_list.resize(ceil_size)
        labels = box_list.get_field('labels')
        # TODO avoid loop
        for gt, l in zip(box_list.box, labels):
            local_x = min(int((gt[2] + gt[0]) / 2), int(w - 1))
            local_y = min(int((gt[3] + gt[1]) / 2), int(h - 1))
            gt_w = gt[2] - gt[0]
            gt_h = gt[3] - gt[1]

            max_iou = 0
            obj_idx = 0
            for j in range(self.box_num):
                anchor_w, anchor_h = self.ratio[j]
                area_anchor = anchor_w * anchor_h
                area_ground_truth = gt_w * gt_h
                inter = min(anchor_w, gt_w) * min(anchor_h, gt_h)
                iou = inter / (area_anchor + area_ground_truth - inter)
                if iou > max_iou:
                    max_iou = iou
                    obj_idx = j

                if iou > self.encoder_iou and not self.force_encoder and gt_h > 0.5 and gt_w > 0.5:
                    bb_response[local_y, local_x,j] = 1
                    bb_boxes[local_y, local_x,j, 0] = ((gt[2] + gt[0]) / 2) - local_x
                    bb_boxes[local_y, local_x,j ,1] = ((gt[3] + gt[1]) / 2) - local_y
                    bb_boxes[local_y, local_x, obj_idx, 2] = math.log(max(gt_w, 0.01) / self.ratio[j][0])
                    bb_boxes[local_y, local_x, obj_idx, 3] = math.log(max(gt_h, 0.01) / self.ratio[j][1])
                    bb_class[local_y, local_x,j ,l] = 1


            if self.force_encoder and gt_h > 0.5 and gt_w > 0.5:
                bb_response[ local_y, local_x,obj_idx] = 1
                bb_boxes[local_y, local_x, obj_idx,0] = ((gt[2] + gt[0]) / 2) - local_x
                bb_boxes[local_y, local_x, obj_idx,1] = ((gt[3] + gt[1]) / 2) - local_y
                bb_boxes[local_y, local_x, obj_idx,2] = math.log(max(gt_w, 0.01) / self.ratio[obj_idx][0])
                bb_boxes[local_y, local_x, obj_idx,3] = math.log(max(gt_h, 0.01) / self.ratio[obj_idx][1])
                bb_class[local_y, local_x, obj_idx,l] = 1


        gt_value = (bb_class,bb_response,bb_boxes)
        return gt_value

    def _prepare_target(self,meta,ceil_size,cls_num,device):
        bb_class_list = []
        bb_response_list = []
        bb_boxes_list = []

        for target in meta:
            t = target['boxlist'].copy()
            t.resize(ceil_size)
            #print(t.box)
            gt_value = self._push_mask(t, ceil_size,cls_num)
            bb_class,bb_response,bb_boxes = gt_value

            bb_class_list.append(bb_class.unsqueeze(dim=0))
            bb_response_list.append(bb_response.unsqueeze(dim=0))
            bb_boxes_list.append(bb_boxes.unsqueeze(dim=0))

        bb_class_list = torch.cat(bb_class_list,dim=0).to(device)
        bb_response_list = torch.cat(bb_response_list, dim=0).to(device)
        bb_boxes_list = torch.cat(bb_boxes_list, dim=0).to(device)
        gt_value = (bb_class_list,bb_response_list,bb_boxes_list)

        return gt_value

    def forward(self,pred,meta):
        #print("=" * 50)
        bs_pred_cls, bs_pred_conf, bs_pred_offset = pred
        device = bs_pred_cls.get_device()

        B_size,h,w,_,cls_num = bs_pred_cls.shape
        bbox_num = self.box_num

        ceil_size = (w,h)
        anchor_xy,anchor_wh = self._gen_anchor(ceil_size)
        gt_value = self._prepare_target(meta,ceil_size,cls_num,device)
        bs_target_cls,bs_target_conf,bs_target_offset = gt_value
        anchor_xy = anchor_xy.unsqueeze(dim=0).view(h,w,bbox_num, 2).to(device)
        anchor_wh = anchor_wh.unsqueeze(dim=0).view(h, w, bbox_num, 2).to(device)

        bs_conf_mask = bs_target_conf > 0

        if bs_conf_mask.sum() <= 0:
            loss_obj = F.binary_cross_entropy_with_logits(bs_pred_conf, bs_target_conf, reduction='sum')
            return{'nObj': loss_obj / B_size}


        bs_offset_mask = bs_conf_mask.repeat(1,1,1,1,4)
        bs_pred_box = bs_pred_offset.clone()
        loss_pos_obj = 0
        loss_neg_obj = 0
        loss_xy = 0
        loss_wh = 0

        #TODO no loop
        for bs_idx in range(B_size):
            target_offset = bs_target_offset[bs_idx]
            offset_mask = bs_offset_mask[bs_idx]

            pred_offset = bs_pred_offset[bs_idx]
            pred_box = bs_pred_box[bs_idx]

            target_conf = bs_target_conf[bs_idx]
            pred_conf = bs_pred_conf[bs_idx]
            conf_mask = bs_conf_mask[bs_idx]

            target_offset = target_offset[offset_mask].view(-1, 4)
            loc_pred_offset = pred_offset[offset_mask].view(-1, 4)
            pred_box = pred_box.view(-1, 4)

            with torch.no_grad():
                pred_box[:, 2:] = pred_box[:, 2:].exp() * anchor_wh.view(-1, 2)
                pred_box[:, :2] = pred_box[:, :2].sigmoid() + anchor_xy.view(-1,2)
                pred_box[:, :2] = pred_box[:, :2] - pred_box[:, 2:4] * 0.5
                anchor_mask = offset_mask[ :, :, :, :2]
                anchor_xy_gt = anchor_xy[anchor_mask].view(-1, 2)
                anchor_wh_gt = anchor_wh[anchor_mask].view(-1, 2)

                gt_box = target_offset.clone()
                gt_box[:, 2:] = gt_box[:, 2:].exp() * anchor_wh_gt
                gt_box[:, :2] = gt_box[:, :2] + anchor_xy_gt
                gt_box[:, :2] = gt_box[:, :2] - gt_box[:, 2:4] * 0.5
                iou = self.bbox_ious(pred_box.float(),gt_box.float())
                #print(gt_box)
               
            if len(gt_box) > 0:
                #print(len(gt_box))
                idx = iou.argmax(dim=1)
                idx = idx.unsqueeze(dim=1)
                iou = iou.gather(1, idx).view(-1,1)


                target_conf = target_conf.view(-1,1)
                conf_mask = conf_mask.view(-1,1)
                pred_conf = pred_conf.view(-1,1)

                neg_mask = (iou < 0.6) * (conf_mask==False)
                loss_pos_obj += F.binary_cross_entropy_with_logits(pred_conf[conf_mask],
                                                                  iou[conf_mask],
                                                                  reduction='sum')

                loss_neg_obj += F.binary_cross_entropy_with_logits(pred_conf[neg_mask],
                                                                  target_conf[neg_mask],
                                                                  reduction='sum')

                loss_xy += F.binary_cross_entropy_with_logits(loc_pred_offset[:,:2],target_offset[:,:2],reduction='sum')
                loss_wh += F.mse_loss(loc_pred_offset[:,2:], target_offset[:,2:], reduction='sum')
            else:
                loss_neg_obj += F.binary_cross_entropy_with_logits(pred_conf.view(-1),target_conf.view(-1),reduction='sum')
                
                
                
        cls_mask = bs_conf_mask.repeat(1, 1, 1, 1, cls_num)
        gt_cls = bs_target_cls[cls_mask].view(-1,cls_num).argmax(dim=1)
        pred_cls = bs_pred_cls[cls_mask].view(-1, cls_num)
        loss_cls = F.cross_entropy(pred_cls,gt_cls,reduction='sum')
        #print(gt_cls)

        return {'pObj': loss_pos_obj / B_size * self.l_obj,
                'nObj': loss_neg_obj / B_size * self.l_noobj,
                'cls': loss_cls / B_size,
                'xy':  loss_xy  / B_size * 0.5,
                'wh':  loss_wh  / B_size * self.l_coord
                }

