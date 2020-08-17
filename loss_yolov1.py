import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class loss_yolov1(nn.Module):
    def __init__(self, S, B, l_coord, l_noobj):
        super(loss_yolov1,self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def compute_iou(self, box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''
        N = box1.size(0)
        M = box2.size(0)

        lt = torch.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh<0] = 0  # clip at 0
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou
    def calloss(self,pred_tensor,target_tensor):
        N = pred_tensor.size()[0]
        obj_mask = target_tensor[..., 4] == 1  # N*14*14
        noobj_mask = target_tensor[..., 4] == 0  # N*14*14
        targ_obj = target_tensor[obj_mask]
        pred_obj = pred_tensor[obj_mask]
        pred_noobj = pred_tensor[noobj_mask]
        targ_boxes = targ_obj[..., :5 * 1].contiguous().view(-1, 5)
        pred_boxes = pred_obj[..., :5 * self.B].contiguous().view(-1, 5)

        row = torch.arange(14, dtype=torch.float).unsqueeze(-1).expand_as(obj_mask)[obj_mask].unsqueeze(-1)
        col = torch.arange(14, dtype=torch.float).unsqueeze(0).expand_as(obj_mask)[obj_mask].unsqueeze(-1)
        calrow_targ = torch.cat((col, row), dim=1).cuda()
        calrow_pred = torch.cat((calrow_targ, calrow_targ), dim=1).view(-1, 2).cuda()
        targ_boxes_xyxy = targ_boxes.clone()
        pred_boxes_xyxy = pred_boxes.clone()
        targ_boxes_xyxy[:,:2] = targ_boxes[:, :2] / self.S - 0.5 * targ_boxes[:, 2:4] + calrow_targ / self.S
        pred_boxes_xyxy[:,:2] = pred_boxes[:, :2] / self.S - 0.5 * pred_boxes[:, 2:4] + calrow_pred / self.S
        targ_boxes_xyxy[:,2:4] = targ_boxes[:, :2] / self.S + 0.5 * targ_boxes[:, 2:4] + calrow_targ / self.S
        pred_boxes_xyxy[:,2:4] = pred_boxes[:, :2] / self.S + 0.5 * pred_boxes[:, 2:4] + calrow_pred / self.S
        iou_res = self.compute_iou(targ_boxes_xyxy[:, :4], pred_boxes_xyxy[:, :4])
        targ_boxexkeep = torch.zeros_like(targ_boxes).cuda()
        pred_boxexkeep = torch.zeros_like(targ_boxes).cuda()
        pred_bestiou_obj = torch.zeros(targ_boxes.size(0)).cuda()
        pred_conf_noobj_1 = pred_noobj[..., [b*5+4 for b in range(self.B)]].view(-1)
        pred_conf_noobj_2 = torch.zeros(pred_boxes.size(0)-targ_boxes.size(0)).cuda()
        for i in range(targ_boxes_xyxy.size(0)):
            iouargmax = iou_res[i][i*self.B:(i+1)*self.B].argmax()
            targ_boxexkeep[i] = targ_boxes[i]
            pred_boxexkeep[i] = pred_boxes[i*self.B+iouargmax]
            pred_bestiou_obj[i] = iou_res[i][i*self.B+iouargmax]
            noobj_set = 0
            for b in range(self.B):
                if b != iouargmax:
                    pred_conf_noobj_2[(self.B-1)*i+noobj_set] = pred_boxes[i*self.B+b][-1]
                    noobj_set += 1
        pred_conf_noobj = torch.cat((pred_conf_noobj_1, pred_conf_noobj_2))
        loss_xy = F.mse_loss(targ_boxexkeep[..., :2], pred_boxexkeep[..., :2], reduction='sum') * self.l_coord/N
        loss_wh = F.mse_loss(targ_boxexkeep[..., 2:4].sqrt(), pred_boxexkeep[..., 2:4].sqrt(), reduction='sum')*self.l_coord/N
        # loss_wh = F.mse_loss(targ_boxexkeep[..., 2:4], pred_boxexkeep[..., 2:4], reduction='sum')*self.l_coord/N
        loss_Cobj = F.mse_loss(pred_bestiou_obj, pred_boxexkeep[..., -1], reduction='sum')/N
        loss_Cnoobj = F.mse_loss(torch.zeros_like(pred_conf_noobj), pred_conf_noobj, reduction='sum')*self.l_noobj/N

        con_pre_class = pred_obj[:, self.B * 5:]
        con_tar_class = targ_obj[:, self.B * 5:]
        loss_class = F.mse_loss(con_tar_class, con_pre_class, reduction='sum')/N

        loss_res = loss_xy+loss_wh+loss_Cobj+loss_Cnoobj+loss_class

        return loss_res, loss_xy, loss_wh, loss_Cobj, loss_Cnoobj, loss_class