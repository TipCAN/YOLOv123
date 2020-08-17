
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn


def load_conv_bn(buf, start, conv_model, bn_model):
    num_w = conv_model.weight.numel()

    num_b = bn_model.bias.numel()
    bn_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_mean.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b
    bn_model.running_var.copy_(torch.from_numpy(buf[start:start + num_b]))
    start = start + num_b

    conv_weight = torch.from_numpy(buf[start:start + num_w])
    conv_model.weight.data.copy_(conv_weight.view_as(conv_model.weight))
    start = start + num_w

    return start
def load_conv(buf, start, conv_model):
    num_w = conv_model.weight.numel()
    num_b = conv_model.bias.numel()

    conv_model.bias.data.copy_(torch.from_numpy(buf[start:start + num_b]).view_as(conv_model.bias));
    start = start + num_b

    conv_model.weight.data.copy_(torch.from_numpy(buf[start:start + num_w]).view_as(conv_model.weight));
    start = start + num_w

    return start


class conv_block(nn.Module):

    def __init__(self, inplane, outplane, kernel_size, pool, stride=1):
        super(conv_block, self).__init__()

        pad = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)
        self.pool = pool  # MaxPool2d(2,stride = 2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)

        if self.pool:
            out = F.max_pool2d(out, kernel_size=2, stride=2)

        return out



layer_part_1 = [
        # Unit1 (2)
        (32, 3, True),
        (64, 3, True),
        # Unit2 (3)
        (128, 3, False),
        (64, 1, False),
        (128, 3, True),
        # Unit3 (3)
        (256, 3, False),
        (128, 1, False),
        (256, 3, True),
        # Unit4 (5)
        (512, 3, False),
        (256, 1, False),
        (512, 3, False),
        (256, 1, False),
        (512, 3, False),

]
layer_part_2 = [
        # Unit5 (5)
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
        (512, 1, False),
        (1024, 3, False),
]

class reorg_layer(nn.Module):
    def __init__(self,stride):
        super(reorg_layer,self).__init__()
        self.stride = stride

    def forward(self, x):

        B,C,H,W = x.shape[0],x.shape[1],x.shape[2],x.shape[3]

        x = x.view(B, C, H // self.stride, self.stride, H // self.stride, self.stride)

        data = []
        for i in range(self.stride):
            for j in range(self.stride):
                data.append(x[:, :, :, i, :, j])

        data = torch.cat([d for d in data], 1)

        return data

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class darknet_19(nn.Module):

    def __init__(self,gather_low_feat=False):
        super(darknet_19, self).__init__()
        self.feature_1 = self.make_layers(3,layer_part_1)
        self.feature_2 = self.make_layers(512,layer_part_2)
        self.gather_low_feat = gather_low_feat

    def make_layers(self,inplane,cfg):
        layers = []

        for outplane,kernel_size,pool in cfg:
            layers.append(conv_block(inplane,outplane,kernel_size,pool))
            inplane = outplane

        return nn.Sequential(*layers)

    def load_weight(self,weight_file):

        if weight_file is not None:
            print("Load pretrained models !")

            fp = open(weight_file, 'rb')
            header = np.fromfile(fp, count=4, dtype=np.int32)
            header = torch.from_numpy(header)
            buf = np.fromfile(fp, dtype = np.float32)

            start = 0
            for idx,m in enumerate(self.feature_1.modules()):
                if isinstance(m, nn.Conv2d):
                    conv = m
                if isinstance(m, nn.BatchNorm2d):
                    bn = m
                    start = load_conv_bn(buf,start,conv,bn)
                    # trick
                    m.eval()

            for idx,m in enumerate(self.feature_2.modules()):
                if isinstance(m, nn.Conv2d):
                    conv = m
                if isinstance(m, nn.BatchNorm2d):
                    bn = m
                    start = load_conv_bn(buf,start,conv,bn)
            assert start == buf.shape[0]

    def forward(self, x):

        out1 = self.feature_1(x)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = self.feature_2(out2)
        if self.gather_low_feat:
            return out1,out2
        else:
            return out2



class conv_blockv3(nn.Module):

    def __init__(self,inplane,outplane,kernel_size,stride=1,pad=1):
        super(conv_blockv3, self).__init__()

        self.conv = nn.Conv2d(inplane, outplane, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.act = nn.LeakyReLU(0.1)

    def forward(self,x):

        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)


        return out


class residual_block(nn.Module):
    expansion = 1

    def __init__(self, in_planes,out_planes, stride=1):
        super(residual_block, self).__init__()
        self.conv1 = conv_blockv3(in_planes, in_planes//2, kernel_size=1, stride=stride, pad=0)
        self.conv2 = conv_blockv3(in_planes//2, out_planes, kernel_size=3, stride=1, pad=1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = x + out
        return out



class pred_module(nn.Module):

    def __init__(self,inplance,plance,cls_num,bbox_num,ratio,stride):
        super(pred_module,self).__init__()

        self.cls_num = cls_num
        self.bbox_num = bbox_num
        self.ratio = ratio
        self.stride = stride
        assert len(ratio) == bbox_num
        self.extra_layer = conv_blockv3(inplance, plance, 3,stride=1,pad=1)

        self.cls_pred = nn.Conv2d(plance,self.cls_num * self.bbox_num, 1, stride=1, padding=0, bias=True)
        self.response_pred = nn.Conv2d(plance,self.bbox_num, 1, stride=1, padding=0, bias=True)
        self.offset_xy = nn.Conv2d(plance,2 * self.bbox_num, 1, stride=1, padding=0, bias=True)
        self.offset_wh = nn.Conv2d(plance,2 * self.bbox_num, 1, stride=1, padding=0, bias=True)


    def gen_anchor(self,ceil,ratio,stride):

        anchor_xy = []
        anchor_wh = []
        w,h = ceil

        for r in ratio:
            x = torch.linspace(0, w-1, w).unsqueeze(dim=0).repeat(h, 1).unsqueeze(dim=0)
            y = torch.linspace(0, h-1, h).unsqueeze(dim=0).repeat(w, 1).unsqueeze(dim=0).permute(0, 2, 1)
            width = torch.Tensor([r[0]/stride]).view(1, 1, 1).repeat(1, h, w)
            height = torch.Tensor([r[1]/stride]).view(1, 1, 1).repeat(1, h, w)

            anchor_xy.append(torch.cat((x, y), dim=0).unsqueeze(dim=0))
            anchor_wh.append(torch.cat((width, height), dim=0).unsqueeze(dim=0))

        anchor_xy = torch.cat(anchor_xy, dim=0).view(-1, h, w)
        anchor_wh = torch.cat(anchor_wh, dim=0).view(-1, h, w)

        return anchor_xy,anchor_wh



    def forward(self, x ,sigmoid_out):
        output = self.extra_layer(x)
        device = x.get_device()

        B,c,ceil_h,ceil_w = output.shape
        ceil = (ceil_w,ceil_h)

        anchor_xy,anchor_wh = self.gen_anchor(ceil,self.ratio,self.stride)
        anchor_xy = anchor_xy.repeat(B, 1, 1, 1).to(device)
        anchor_wh = anchor_wh.repeat(B, 1, 1, 1).to(device)

        pred_cls = self.cls_pred(output)
        pred_response = self.response_pred(output)
        pred_xy = self.offset_xy(output)
        pred_wh = self.offset_wh(output).exp()

        if sigmoid_out:
            pred_cls = pred_cls.view(B,-1,self.cls_num,ceil_h,ceil_w)
            pred_cls = pred_cls.softmax(dim=2)
            pred_cls = pred_cls.view(B,-1,ceil_h,ceil_w)
            pred_response = pred_response.sigmoid()
            pred_xy = pred_xy.sigmoid()
            pred_xy = pred_xy + anchor_xy


        pred_xy = pred_xy + anchor_xy
        pred_wh = pred_wh * anchor_wh
        pred_xy = pred_xy.view(-1,self.bbox_num,2,ceil_h,ceil_w)
        pred_wh = pred_wh.view(-1,self.bbox_num,2,ceil_h,ceil_w)
        pred_bbox = torch.cat([pred_xy,pred_wh],dim=2).view(-1,self.bbox_num*4,ceil_h,ceil_w)
        pred = (pred_cls, pred_response, pred_bbox)
        return pred






class conv_sets(nn.Module):
    def __init__(self,inplance,plance,outplance):
        super(conv_sets,self).__init__()

        self.conv1 = conv_blockv3(inplance,outplance,1,stride=1,pad=0)
        self.conv2 = conv_blockv3(outplance,plance,3,stride=1,pad=1)
        self.conv3 = conv_blockv3(plance,outplance,1,stride=1,pad=0)
        self.conv4 = conv_blockv3(outplance,plance,3,stride=1,pad=1)
        self.conv5 = conv_blockv3(plance,outplance,1,stride=1,pad=0)


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class up_sample(nn.Module):

    def __init__(self,inplance,outplance):
        super(up_sample,self).__init__()
        self.conv1 = conv_blockv3(inplance,outplance,1,stride=1,pad=0)

    def forward(self, x):

        out = self.conv1(x)

        out = nn.functional.interpolate(out,scale_factor=2,mode='nearest')

        return out

class darknet53(nn.Module):

    def __init__(self, in_planes=3):
        super(darknet53, self).__init__()

        self.conv1 = conv_blockv3(in_planes, 32, 3)
        self.conv2 = conv_blockv3(32, 64, 3, stride = 2, pad = 1)

        self.block1 = residual_block(64,64)
        self.conv3 = conv_blockv3(64, 128, 3, stride = 2, pad = 1)

        self.block2 = nn.ModuleList()
        self.block2.append(residual_block(128, 128))
        self.block2.append(residual_block(128, 128))

        self.conv4 = conv_blockv3(128, 256, 3, stride = 2, pad = 1)

        self.block3 = nn.ModuleList()
        for i in range(8):
            self.block3.append(residual_block(256, 256))

        self.conv5 = conv_blockv3(256, 512, 3, stride = 2, pad = 1)

        self.block4 = nn.ModuleList()
        for i in range(8):
            self.block4.append(residual_block(512, 512))

        self.conv6 = conv_blockv3(512,1024, 3, stride = 2, pad = 1)

        self.block5 = nn.ModuleList()
        for i in range(4):
            self.block5.append(residual_block(1024, 1024))

    def load_part(self,buf,start,part):
        for idx,m in enumerate(part.modules()):
            if isinstance(m, nn.Conv2d):
                conv = m
            if isinstance(m, nn.BatchNorm2d):
                bn = m
                start = load_conv_bn(buf,start,conv,bn)
        return start

    def load_weight(self, weight_file):

        if weight_file is not None:
            print("Load pretrained models !")

            fp = open(weight_file, 'rb')
            header = np.fromfile(fp, count=5, dtype=np.int32)
            header = torch.from_numpy(header)
            buf = np.fromfile(fp, dtype=np.float32)

            start = 0
            start = self.load_part(buf, start, self.conv1)
            start = self.load_part(buf, start, self.conv2)
            start = self.load_part(buf, start, self.block1)
            start = self.load_part(buf, start, self.conv3)
            start = self.load_part(buf, start, self.block2)
            start = self.load_part(buf, start, self.conv4)
            start = self.load_part(buf, start, self.block3)
            start = self.load_part(buf, start, self.conv5)
            start = self.load_part(buf, start, self.block4)
            start = self.load_part(buf, start, self.conv6)
            start = self.load_part(buf, start, self.block5)

            print(start,buf.shape[0])

    def forward(self,x):
        detect_feat = []
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.block1(out)
        out = self.conv3(out)

        for modu in self.block2:
            out = modu(out)

        out = self.conv4(out)
        for modu in self.block3:
            out = modu(out)
        detect_feat.append(out)

        out = self.conv5(out)
        for modu in self.block4:
            out = modu(out)
        detect_feat.append(out)

        out = self.conv6(out)
        for modu in self.block5:
            out = modu(out)
        detect_feat.append(out)

        return detect_feat
