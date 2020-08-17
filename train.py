import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
import time
import tqdm
import cv2
from copy import deepcopy
import numpy as np
from torch.utils.data import DataLoader
from coco_dataset import cocoDataset, ToBbox, eval_results, make_val_map
from utils.YOLODataLoader import yoloDataset
from loss_yolov1 import loss_yolov1
from yolo.loss import yolov1_loss
from v1Loss import YOLOLossV1
from models.resnet_YOLO import resnet50
from models.mobilenetv2_YOLO import mobilenet_v2
import backbones.OriginResNet as OriginResNet
import backbones.OriginDenseNet as OriginDenseNet
import utils.utils as utils
import utils_out
import models.resnet_yolo as resyolo
import freezelayers


def main():
    B = 2
    S = 14
    learning_rate = 0.001
    num_epochs = 500
    bs = 8
    eval_epoch = 100
    save_epoch = 50
    decey = [200, 300]
    start_epoch = 0
    load_epoch = None
    save_dir = './save_person4_m'
    seed = 100
    device = 'cuda:0'
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = utils.create_logger(save_dir, 'train')
    criterion = loss_yolov1(S, B, 5, 0.5)
    # criterion = YOLOLossV1(_S=14, _B=2, _clsN=20, _l_coord=5., _l_noobj=0.5, _device='cuda:0')
    # model = resnet50()
    model = mobilenet_v2()
    # model = OriginResNet.resnet50(S=14)
    # model = OriginDenseNet.densenet121(S=14)
    # model = resyolo.resnet50()
    if load_epoch != None:
        model_path = './save_person2_m/epoch_%d.pth' % (load_epoch)
        model.load_state_dict(torch.load(model_path).state_dict())
    else:
        # orgnet = torchvision.models.resnet50(pretrained=True)
        orgnet = torchvision.models.mobilenet_v2(pretrained=True)
        new_state_dict = orgnet.state_dict()
        dd = model.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        model.load_state_dict(dd)
    #     freezelayers.freeze_by_idxs(model, [0,1,2,3,4])
    train_pipelines = ['Resize',
                       'RandomBrightness',
                       'RandomSaturation',
                       'RandomHue',
                       'randomBlur',
                       'random_flip',
                       'to_tensor',
                       'normalization',
                       ]
    # train_imgdirs_list = ['../data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/',
    #                       '../data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/',
    #                       '../data/vocdata/VOCdevkit/VOCtrainval_11-May-2012/JPEGImages/',
    #                       '../data/vocdata/VOCdevkit/VOCtrainval_11-May-2012/JPEGImages/',
    #                       ]
    # train_annfiles_list = ['../data/vocdata/tococo/voc2007_train.json',
    #                        '../data/vocdata/tococo/voc2007_val.json',
    #                        '../data/vocdata/tococo/voc2012_train.json',
    #                        '../data/vocdata/tococo/voc2012_val.json',
    #                        ]
    train_imgdirs_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/images/',]
    train_annfiles_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_add1_coco.json',]
    val_pipelines = ['Resize',
                     # 'BGR2RGB',
                     # 'subMean',
                     'to_tensor',
                     'normalization',
                     ]
    # val_imgdirs_list = ['../data/vocdata/VOCdevkit/VOCtest_06-Nov-2007/JPEGImages/']
    # val_annfiles_list = ['../data/vocdata/tococo/voc2007_test.json']
    val_imgdirs_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/images/',]
    val_annfiles_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_add1_coco.json',]
    # train_dataset = cocoDataset(imgdirs_list=train_imgdirs_list,
    #                             annfiles_list=train_annfiles_list,
    #                             trainorval='train',
    #                             transforms_pipeline=train_pipelines)
    # train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)

    val_dataset = cocoDataset(imgdirs_list=val_imgdirs_list,
                              annfiles_list=val_annfiles_list,
                              trainorval='val',
                              transforms_pipeline=val_pipelines)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    ###########
    train_transform = transforms.Compose([
        transforms.Lambda(utils.cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_dataset = yoloDataset(imgdirs_list=train_imgdirs_list,
                                annfiles_list=train_annfiles_list,
                                train=True,
                                transform=train_transform,
                                device='cuda',
                                little_train=False, with_file_path=False, S=14, B=2, C=20, test_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    ###########

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    model.cuda()
    e = start_epoch
    while e < num_epochs:
        e += 1
        if e-1 in decey:
            learning_rate = learning_rate/10
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        loss_total = 0.
        loss_xy_total = 0.
        loss_wh_total = 0.
        loss_Cobj_total = 0.
        loss_Cnoobj_total = 0.
        loss_class_total = 0.
        model.train()
        print('epoch:%d,lr:%.5f '% (e, learning_rate))
        logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        logger.info('epoch: %d,lr: %.5f , loss: %.5f' % (e, learning_rate, loss_total))
        with tqdm.tqdm(range(len(train_loader))) as pbar:
            for i, data_batch in enumerate(train_loader):
                #
                # for b in range(4):
                #     boxes, clss, confs = utils_out.decoder(data_batch[1][b], grid_num=S, gt=True)
                #     # print(boxes, clss, confs)
                #     print('~' * 50 + '\n\n\n')
                #     mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
                #     std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
                #     un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                #     img = un_normal_trans(data_batch[0][b].squeeze(0))
                #     utils.draw_debug_rect(img.squeeze(0).permute(1, 2, 0), boxes, clss, confs)
                #
                img = data_batch[0].to(device)
                SSgrid_gt = data_batch[1].to(device)
                SSgrid_pre = model(img)
                loss, loss_xy, loss_wh, loss_Cobj, loss_Cnoobj, loss_class = criterion.calloss(SSgrid_pre, SSgrid_gt)
                # loss = criterion.forward(SSgrid_pre.clone(), SSgrid_gt.clone())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_total += float(loss)
                loss_xy_total += float(loss_xy)
                loss_wh_total += float(loss_wh)
                loss_Cobj_total += float(loss_Cobj)
                loss_Cnoobj_total += float(loss_Cnoobj)
                loss_class_total += float(loss_class)
                lossinfo = 'l%0.2f,l_xy%0.2f,l_wh%0.2f,l_Cobj%0.2f,l_Cnoobj%0.2f,l_class%0.2f'
                pbar.set_description(lossinfo%(loss_total/(i+1),loss_xy_total/(i+1),loss_wh_total/(i+1),loss_Cobj_total/(i+1),loss_Cnoobj_total/(i+1),loss_class_total/(i+1)),False)
                pbar.update()
#         logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#         logger.info('epoch: %d,lr: %.5f , loss: %.5f' % (e, learning_rate, loss_total))
        if e % save_epoch == 0:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model, (save_dir+'/epoch_%d.pth')%(e))
        if e % eval_epoch == 0:
            # little_val_num = None
            # val_imgdirs_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/images/']
            # val_annfiles_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_coco.json']
            #
            # gt_test_map = make_val_map(annFile=val_annfiles_list[0], idval_num=little_val_num)
            # test_dataset = yoloDataset(imgdirs_list=val_imgdirs_list,
            #                            annfiles_list=val_annfiles_list,
            #                            train=False, transform=train_transform, device='cuda',
            #                            little_train=False, with_file_path=True, S=S)
            # data_len = int(len(test_dataset) / 1)
            # now_little_mAP = utils_out.run_test_mAP(model, deepcopy(gt_test_map), test_dataset, data_len, S=S,
            #                               logger=logger, little_test=little_val_num)
            # ##
            json_path = './result/eval.json'
            model.eval()
            eval_ap = eval_results(json_path, val_annfiles_list)
            with tqdm.tqdm(range(len(val_loader))) as pbar:
                for i, data_batch in enumerate(val_loader):
                    img_result, img_id, org_size_yx = data_batch
                    SSgrid_pre = model(img_result.cuda().float())
                    #
                    # boxes, clss, confs = utils_out.decoder(SSgrid_pre.cpu(), grid_num=S, gt=False)
                    # mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
                    # std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
                    # un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                    # img = un_normal_trans(img_result.cpu().squeeze(0))
                    # utils.draw_debug_rect(img.squeeze(0).permute(1, 2, 0), boxes, clss, confs)
                    #
                    Bboxresult = ToBbox(SSgrid_pre.cpu())
                    eval_ap.add_item(result_item=Bboxresult, data_batch=data_batch)
                    pbar.update()
            ap_res = eval_ap.eval()
            print(ap_res)
torch.cuda.is_available()

if __name__ == '__main__':
    main()
