import torch
import cv2
from torchsummary import summary

from src.scuba_tracking.scuba_tracking.models.darknet_models import *

CHECKPOINT_PATH = './' # FIXME


class YOLOv3:
    def __init__(self, img_shape=(416, 416), load_pretrained=False):
        self.checkpoint_path = CHECKPOINT_PATH
        self.predictions = None
        self.img_shape = img_shape
        # Fixme correct the following path
        cfg = '/home/faraz/sim_ws/src/scuba_tracking/scuba_tracking/models/cfg/yolov3_customized.cfg'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = Darknet(cfg, arc='default').to(self.device)

        if load_pretrained:
            self.load_weights()

    def __call__(self, img):
        # A simple test case...
        if len(img.shape) == 3:
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).float().cuda()
            img = img.unsqueeze(0)
        # In case of being used as a detection head
        # DO NOTHING
        self.predictions = self.model(img)
        return self.predictions

    def load_weights(self):
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)


    def read_labels(self, classes_file_path='classes.txt'):
        classes_list = []
        with open(classes_file_path, 'r') as f:
            lines_ = f.readlines()
            for line in lines_:
                classes_list.append(line[:-1])
        return classes_list

def inference(inf_out, image_size=(416,416), conf_threshold=0.001, nms_threshold=0.5, iou_thres=0.5):
    output = non_max_suppression(inf_out, conf_thres=conf_threshold, nms_thres=nms_threshold)
    # Statistics per image
    seen = 0
    detected = []
    width = image_size[0]
    height = image_size[1]
    for si, pred in enumerate(output):

        # Clip boxes to image bounds
        clip_coords(pred, (height, width))

        # Search for correct predictions
        for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
            detected.append([pbox,pconf, pcls_conf, pcls])

    return detected

if __name__ == '__main__':
    #image = cv2.imread('./Raw_image.png')
    detector = YOLOv3()
    detector.model.eval()
    input_shape = (416, 416, 3)
    summary(detector.model, (input_shape[2], input_shape[0], input_shape[1]))
    image = np.random.rand(*input_shape)
    inf_out, train_out = detector(image)
    inference(inf_out)
    #print(inf_out,train_out)
