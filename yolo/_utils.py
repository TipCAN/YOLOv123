import torch

def get_kp_torch_batch( pred, conf, topk=100):
    b, c, h, w = pred.shape
    pred = pred.contiguous().view(-1)
    pred[pred < conf] = 0
    score, topk_idx = torch.topk(pred, k=topk)

    batch = topk_idx / (h * w * c)

    cls = (topk_idx - batch * h * w * c) / (h * w)

    channel = (topk_idx - batch * h * w * c) - (cls * h * w)

    x = channel % w
    y = channel / w

    return x.view(-1), y.view(-1), cls.view(-1), batch.view(-1)



def compute_iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''

    lt = torch.max(
        box1[:, :2],  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2],  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:],  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:],  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, 0] * wh[:, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

    iou = inter / (area1 + area2 - inter + 1e-4)
    return iou