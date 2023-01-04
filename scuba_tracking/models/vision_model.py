import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import  functional as F
from src.scuba_tracking.scuba_tracking.models.utils.utils import *
import time
# Fixme make the following import paths common like the one we have in vive robot
from src.scuba_tracking.scuba_tracking.models.yolo_model import YOLOv3, inference
from torchsummary import summary

class BlockTypeA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c1, out_c2, upscale = True):
        super(BlockTypeA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c2, out_c2, kernel_size=1),
            nn.BatchNorm2d(out_c2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c1, out_c1, kernel_size=1),
            nn.BatchNorm2d(out_c1),
            nn.ReLU(inplace=True)
        )
        self.upscale = upscale

    def forward(self, a, b):
        b = self.conv1(b)
        a = self.conv2(a)
        b = F.interpolate(b, scale_factor=2.0, mode='bilinear', align_corners=True)
        return torch.cat((a, b), dim=1)


class BlockTypeB(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x)
        return x

class BlockTypeC(nn.Module):
    def __init__(self, in_c, out_c):
        super(BlockTypeC, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, in_c,  kernel_size=3, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU()
        )
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        self.channel_pad = out_planes - in_planes
        self.stride = stride
        #padding = (kernel_size - 1) // 2

        # TFLite uses slightly different padding than PyTorch
        if stride == 2:
            padding = 0
        else:
            padding = (kernel_size - 1) // 2

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        # TFLite uses  different padding
        if self.stride == 2:
            x = F.pad(x, (0, 1, 0, 1), "constant", 0)
            #print(x.shape)

        for module in self:
            if not isinstance(module, nn.MaxPool2d):
                x = module(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, pretrained=True):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
        """
        super(MobileNetV2, self).__init__()

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        width_mult = 1.0
        round_nearest = 8

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            #[6, 96, 3, 1],
            #[6, 160, 3, 2],
            #[6, 320, 1, 1],
        ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(4, input_channel, stride=2)] ###################################### instead of 4 we used 3 for the input channels
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        self.features = nn.Sequential(*features)

        self.fpn_selected = [3, 6, 10]
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        #if pretrained:
        #    print('Loading the pretrained weights of the MobileNetV2')
        #    self._load_pretrained_model()
        #    print('Finished.')

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        fpn_features = []
        for i, f in enumerate(self.features):
            if i > self.fpn_selected[-1]:
                break
            x = f(x)
            if i in self.fpn_selected:
                fpn_features.append(x)

        c2, c3, c4 = fpn_features
        return c2, c3, c4


    def forward(self, x):
        return self._forward_impl(x)

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/mobilenet_v2-b0353104.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


class decoder_part(nn.Module):
    def __init__(self):
        super(decoder_part, self).__init__()
        self.block12 = BlockTypeA(in_c1= 32, in_c2= 64,
                                  out_c1= 64, out_c2=64)
        self.block13 = BlockTypeB(128, 64)

        self.block14 = BlockTypeA(in_c1 = 24,  in_c2 = 64,
                                  out_c1= 32,  out_c2= 32)
        self.block15 = BlockTypeB(64, 64)

        self.block16 = BlockTypeC(64, 16)

    def forward(self, c2, c3, c4):
        x = self.block12(c3, c4)
        x = self.block13(x)
        x = self.block14(c2, x)
        x = self.block15(x)
        x = self.block16(x)
        return x


class scuba_detector(nn.Module):
    def __init__(self):
        # line detector part
        super(scuba_detector, self).__init__()

        self.human_detector = YOLOv3().model

    def forward(self, x, line_detection=True):
        ################### Encoder, part 1: 0.076 on avg
        output = self.human_detector(x)

        return output

if __name__ == '__main__':
    import cv2
    import numpy as np
    # Saving each part weights independently
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = scuba_detector().cuda()

    summary(model, (3, 416, 416))
    # Test on a sample image to see the results
    frame = cv2.imread('/home/faraz/deep_learning_based_approaches/scuba_diver_tracking/3.jpg')

    w = frame.shape[1]
    h = frame.shape[0]
    width = w
    height = h
    img_, _, _, _ = letterbox(frame, img_size, mode='square')
    frame, ratio, padw, padh = letterbox(frame, img_size, mode='square')

    # cv2.imshow('Input to YOLO', frame)
    # cv2.waitKey(10000)
    frame = np.concatenate([frame, np.ones([img_size, img_size, 1])], axis=-1)
    frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    frame = np.ascontiguousarray(frame, dtype=np.float32)  # uint8 to float32
    frame /= 255.0  # 0 - 255 to 0.0 - 1.0
    imgs = []
    imgs.append(frame)
    imgs = torch.from_numpy(np.array(imgs)).cuda()
    x, pred = DNN_model(imgs)
    inf_out, train_out = pred
    # Run NMS
    output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres, max_num_of_boxes=10)
    # Statistics per image
    for si, pred in enumerate(output):
        if pred is None:
            break
        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
            # b1_x1, b1_x2 = pbox[0] - pbox[2] / 2, pbox[0] + pbox[2] / 2
            # b1_y1, b1_y2 = pbox[1] - pbox[3] / 2, pbox[1] + pbox[3] / 2
            b1_x1, b1_y1, b1_x2, b1_y2 = (torch.FloatTensor(pbox) * 512 / grid_dim).detach().cpu().numpy()
            if (pconf > 0.02):
                print(pconf, pcls, pcls_conf)
                img_ = cv2.rectangle(img_, (int(b1_x1), int(b1_y1)), (int(b1_x2), int(b1_y2)), (255, 0, 0), 1)

            cv2.imshow('Incoming frames', img_)
            cv2.waitKey(1)

    #img = cv2.resize(img, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    begin_time = time.time()

    train_out, inf_out = pred_lines(img, model, [512, 512], 0.25, 20)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    points = np.array(lines)

    for l in lines:
        cv2.line(img, (int(l[0]), int(l[1])), (int(l[2]), int(l[3])), (0, 200, 200), 1, 16)

    # plt.scatter(points[:,0],-points[:,1])
    # plt.scatter(points[:,2],-points[:,3])
    # plt.savefig(current_dir+'/data/scatter_'+str(count+10)+'.jpg')
    cv2.imwrite('result'+'.jpg', img)

    print(len(inference(inf_out)))
