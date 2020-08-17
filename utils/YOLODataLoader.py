# encoding:utf-8
import os, sys, numpy as np, random, time, cv2
import torch
import json
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import imgaug as ia
from imgaug import augmenters as iaa
#import utils.utils as utils
ia.seed(random.randint(1, 10000))
class yoloDataset(data.Dataset):
    image_size = 448
    def __init__(self,imgdirs_list, annfiles_list,train,transform, device, little_train=False, with_file_path=False, S=7, B = 2, C = 20, test_mode=False):
        print('data init')

        self.imgdirs_list = imgdirs_list
        self.anns_list = annfiles_list
        self.train = train
        self.transform=transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.dataset_list = []
        self.resize = 448
        self.S = S
        self.B = B
        self.C = C
        self.device = device
        self._test = test_mode
        self.with_file_path = with_file_path
        self.img_augsometimes = lambda aug: iaa.Sometimes(0.25, aug)
        self.bbox_augsometimes = lambda aug: iaa.Sometimes(0.5, aug)

        self.augmentation = iaa.Sequential(
            [
                # augment without change bboxes 
                self.img_augsometimes(
                    iaa.SomeOf((1, 3), [
                        iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
                        iaa.Sharpen((0.1, .8)),       # sharpen the image
                        # iaa.GaussianBlur(sigma=(2., 3.5)),
                        iaa.OneOf([
                            iaa.GaussianBlur(sigma=(2., 3.5)),
                            iaa.AverageBlur(k=(2, 5)),
                            iaa.BilateralBlur(d=(7, 12), sigma_color=(10, 250), sigma_space=(10, 250)),
                            iaa.MedianBlur(k=(3, 7)),
                        ]),
                        

                        iaa.AddElementwise((-50, 50)),
                        iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),
                        # iaa.JpegCompression(compression=(80, 95)),

                        iaa.Multiply((0.5, 1.5)),
                        iaa.MultiplyElementwise((0.5, 1.5)),
                        iaa.ReplaceElementwise(0.05, [0, 255]),
                        # iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                        #                 children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        iaa.OneOf([
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(1, iaa.Add((-10, 50)))),
                            iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                                            children=iaa.WithChannels(2, iaa.Add((-10, 50)))),
                        ]),

                    ], random_order=True)
                ),
                iaa.Fliplr(.5),
                iaa.Flipud(.125),
                # augment changing bboxes 
                self.bbox_augsometimes(
                    iaa.Affine(
                        # translate_px={"x": 40, "y": 60},
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        rotate=(-5, 5),
                    )
                )
            ],
            random_order=True
        )
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
        # # torch.manual_seed(23)
        # with open(list_file) as f:
        #     lines  = f.readlines()
        #
        # if little_train:
        #     lines = lines[:64*8]
        #
        # for line in lines:
        #     splited = line.strip().split()
        #     self.fnames.append(splited[0])
        #
        # self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        dataget = self.dataset_list[idx]
        if self._test:
            print(dataget['image_info']['file_name'])
        anns = dataget['ann']
        bboxes = torch.zeros(len(anns), 4)
        labels = torch.zeros(len(anns))
        img_org = cv2.imread(dataget['image_info']['file_name'])
        h, w = img_org.shape[:2]
        for i, ann in enumerate(anns):
            bboxes[i] = torch.tensor(ann['bbox'])
            bboxes[i][::2] = bboxes[i][::2]/w
            bboxes[i][1::2] = bboxes[i][1::2]/h
            bboxes[i][:2] += bboxes[i][2:]/2
            labels[i] = torch.tensor(int(ann['category_id']))
        itemdata = {'img_name': dataget['image_info']['file_name'],
                    'img_id': dataget['image_info']['id'],
                    'img_org': img_org,
                    'img': img_org,
                    'bboxes': bboxes,
                    'labels': labels}
        img = itemdata['img']
        boxes = itemdata['bboxes']
        labels = itemdata['labels'].int()-1
        # boxes_d = boxes.clone()
        # boxes_d[..., 2:] += boxes_d[..., :2]
        # utils.draw_debug_rect(img, boxes_d, labels, labels)
        # boxes, labels = self.get_boxes_labels(fname)
        # fname = self.fnames[idx]
        # if self._test:
        #     print(fname)
        # img = cv2.imread(fname)
        # boxes, labels = self.get_boxes_labels(fname)

        if self.train:
            # TODO
            # add data augument
            # print('before: ')
            # print(boxes)
            boxes = self.convert2augbbox(cv2.resize(img, (self.resize, self.resize)), self.convertXYWH2XYXY(boxes))
            seq_det = self.augmentation.to_deterministic()
            img = seq_det.augment_images([img])[0]
            # boxes = seq_det.augment_bounding_boxes([boxes])[0].remove_out_of_image().clip_out_of_image()
            boxes = seq_det.augment_bounding_boxes([boxes])[0].clip_out_of_image()
            # img = boxes.draw_on_image(cv2.resize(img, (self.resize, self.resize)), thickness=2, color=[0, 0, 255])

            boxes = self.convertAugbbox2XYWH(boxes)
            boxes = torch.Tensor(boxes)
            labels = labels[:boxes.size()[0]]
            # print('after: ')
            # print(boxes)

            # print(img.shape)
            # pass

        target = self.encoder(boxes, labels)  # 7x7x30
        target[:,:,[0,1,2,3,4,5,6,7,8,9]] = target[:,:,[2,3,4,5,0,6,7,8,9,1]]
        img = self.transform(img)
        # print(fname)
        # print(type(img))
        if self.with_file_path:
            return img, target, itemdata['img_name']
        return img, target

    def get_boxes_labels(self, in_path):
        bboxes = []
        labels = []
        with open(in_path.replace('JPEGImages', 'labels').replace('jpg', 'txt'), 'r') as f:
            for line in f:
                ll = line.strip().split(' ')
                labels.append(int(ll[0]))
                x = float(ll[1])
                y = float(ll[2])
                w = float(ll[3])
                h = float(ll[4])
                bboxes.append([x, y, w, h])
        return torch.Tensor(bboxes), torch.LongTensor(labels)
    
    def convertXYWH2XYXY(self, bboxes, im_size=(image_size, image_size)):
        '''
            input: tensor [Cx, Cy, w, h] normalized bboxes
            output: python list [x1, y1, x2, y2] abs bboxes

        '''
        t_boxes = []
        for i in range(bboxes.size()[0]):
            [x, y, w, h] = bboxes[i].tolist()
            x1 = int( (x - 0.5*w) * im_size[0] )
            y1 = int( (y - 0.5*h) * im_size[1] )
            x2 = int( (x + 0.5*w) * im_size[0] )
            y2 = int( (y + 0.5*h) * im_size[1] )
            t_boxes.append([x1, y1, x2, y2])
        return t_boxes
    
    def convert2augbbox(self, img, bboxes):
        '''
            input: tensor list [x1, y1, x2, y2] abs bboxes
            output: python imgaug bboxes object

        '''
        l = []
        for [x1, y1, x2, y2] in bboxes:
            l.append(ia.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
        return ia.BoundingBoxesOnImage(l, shape=img.shape)
    
    def convertaugbbox2X1Y1X2Y2(self, bboxes):
        t_boxes = []
        for i in range(len(bboxes.bounding_boxes)):
            t_boxes.append([bboxes.bounding_boxes[i].x1, bboxes.bounding_boxes[i].y1, bboxes.bounding_boxes[i].x2, bboxes.bounding_boxes[i].y2])
        return t_boxes

    def convertAugbbox2XYWH(self, bboxes, im_size=(image_size, image_size)):
        t_boxes = []
        bboxes = self.convertaugbbox2X1Y1X2Y2(bboxes)
        for [x1, y1, x2, y2] in bboxes:
            cx = (x2 + x1)/2. - 1
            cy = (y2 + y1)/2. - 1
            w = x2 - x1
            h = y2 - y1
            cx /= im_size[0]
            cy /= im_size[1]
            w /= im_size[0]
            h /= im_size[1]
            t_boxes.append([cx, cy, w, h])
        return t_boxes

    def __len__(self):
        return self.num_samples

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[cx,cy,w,h],[]]
        labels (tensor) [...]
        return 7x7x30
        '''
        target = torch.zeros((self.S, self.S, self.B*5 + self.C))
        cell_size = 1./self.S
        # wh = boxes[:,2:]-boxes[:,:2]
        # cxcy = (boxes[:,2:]+boxes[:,:2])/2
        if boxes.size()[0] == 0:
            return target
        try:
            wh = boxes[:, 2:]
            cxcy = boxes[:, :2]
        except:
            print(boxes)
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            target[int(ij[1]),int(ij[0]), :] = 0
            target[int(ij[1]),int(ij[0]),:self.B] = 1
            target[int(ij[1]),int(ij[0]),self.B*5 + int(labels[i])] = 1
            xy = ij*cell_size 
            delta_xy = (cxcy_sample -xy)/cell_size
            for b in range(self.B):
                target[int(ij[1]),int(ij[0]),self.B+b*4 : self.B+b*4+2] = delta_xy
                target[int(ij[1]),int(ij[0]),self.B+b*4+2 : self.B+b*4+4] = wh[i]
                
            
        return target


if __name__ == "__main__":

    import utils_out

    # train_imgdirs_list = ['../../data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/',
    #                       '../../data/vocdata/VOCdevkit/VOCtrainval_06-Nov-2007/JPEGImages/',
    #                       '../../data/vocdata/VOCdevkit/VOCtrainval_11-May-2012/JPEGImages/',
    #                       '../../data/vocdata/VOCdevkit/VOCtrainval_11-May-2012/JPEGImages/',
    #                       ]
    # train_annfiles_list = ['../../data/vocdata/tococo/voc2007_train.json',
    #                        '../../data/vocdata/tococo/voc2007_val.json',
    #                        '../../data/vocdata/tococo/voc2012_train.json',
    #                        '../../data/vocdata/tococo/voc2012_val.json',
    #                        ]
    train_imgdirs_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/images/',
                          ]
    train_annfiles_list = ['/home/yjh/catkin_ws/src/Firmware/yjhworlds/persons_coco.json',
                           ]
    train_transform = transforms.Compose([
        transforms.Lambda(utils_out.cv_resize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    S = 14

    train_dataset = yoloDataset(imgdirs_list=train_imgdirs_list,
                                annfiles_list=train_annfiles_list,
                                train=True,
                                transform=train_transform,
                                device='cuda',
                                little_train=False, with_file_path=False, S=14, B=2, C=20, test_mode=False)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    for i, data_batch in enumerate(train_loader):
        for b in range(1):
            boxes, clss, confs = utils_out.decoder(data_batch[1][b], grid_num=S, gt=True)
            # print(boxes, clss, confs)
            print('~' * 50 + '\n\n\n')
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
            un_normal_trans = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
            img = un_normal_trans(data_batch[0][b].squeeze(0))
            utils_out.draw_debug_rect(img.squeeze(0).permute(1, 2, 0), boxes, clss, confs)
