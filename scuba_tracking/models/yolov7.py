import torch
import numpy as np
from src.scuba_tracking.scuba_tracking.models.experimental import attempt_load
from src.scuba_tracking.scuba_tracking.utils.general import check_img_size, non_max_suppression, \
    scale_coords, set_logging
from src.scuba_tracking.scuba_tracking.utils.plots import plot_one_box
from src.scuba_tracking.scuba_tracking.utils.torch_utils import time_synchronized, TracedModel
from src.scuba_tracking.scuba_tracking.utils.datasets import letterbox
from src.scuba_tracking.scuba_tracking.utils.sort import Sort, iou_batch, centroid_batch
import random
from src.scuba_tracking.scuba_tracking.config import config

class YoloV7:
    def __init__(self, imgsz = config.IMAGE_SIZE[0]):

        # Initialize parameters
        set_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = False # half precision only supported on CUDA
        self.augment = False
        self.conf_threshold = 0.1
        self.iou_threshold = 0.2
        self.verbose = False
        self.trace = True        
        self.no_detect_prob = 0.0 #probability in which detections are lost/thrown away. Set to 0 for 'perfect' detections

        #Tracking params
        self.track = True
        self.sort_max_age = 7
        self.sort_min_hits = 4
        self.track_id = None
        self.last_tracked_bb = None

        # Load model
        self.model = attempt_load(config.YOLO_WEIGHTS, map_location=self.device)  # load FP32 model
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
        if self.trace:
            self.model = TracedModel(self.model, self.device, imgsz)
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1
        
        #Initialize tracker
        self.sort_tracker = Sort(max_age=self.sort_max_age,
                        min_hits=self.sort_min_hits,
                        iou_threshold=self.iou_threshold)
        return

    def detect(self, img):
        img0 = img
        img = letterbox(img, self.imgsz, stride=self.stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
            self.old_img_b = img.shape[0]
            self.old_img_h = img.shape[2]
            self.old_img_w = img.shape[3]
            for i in range(3):
                self.model(img, augment=self.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img, augment=self.augment)[0]
        t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, classes=None)
        t3 = time_synchronized()
        outputs = []
        string_output = '#'
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                dets_to_sort = np.empty((0,6))
                for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                    if random.random() > self.no_detect_prob:
                        dets_to_sort = np.vstack((dets_to_sort, 
                                    np.array([x1, y1, x2, y2, conf, detclass])))
            else:
                dets_to_sort = np.empty((0,6))

            if self.track:
                tracked_dets = self.sort_tracker.update(dets_to_sort)
                
                if len(tracked_dets): #if at least one object is being tracked

                    if self.track_id is None: 
                        if self.last_tracked_bb is None: 
                            track = tracked_dets[np.argmax(tracked_dets[:,4]), :] #initialize track to bb with highest confidence
                        else: 
                            #two metrics to determine new bounding box to track. Pick one

                            #max iou similarity to previously detected track
                            # track = tracked_dets[np.argmax(iou_batch(tracked_dets, self.last_tracked_bb))] 
                            
                            #min centroid distance to previously detected track
                            track = tracked_dets[np.argmin(centroid_batch(tracked_dets, self.last_tracked_bb))] 
                    else: 
                        track = tracked_dets[np.where(tracked_dets[:,-1] == self.track_id)] #set track by id 
                        if len(track): #id was found
                            track = track[0]
                        else: #id not found - reset and continue
                            self.track_id = None
                            continue
                    
                    self.last_tracked_bb = track[0:4]
                    self.track_id = track[-1]
                    x1, y1,x2, y2 = self.last_tracked_bb
                    conf = track[4]
                    cls = self.names[int(track[5])]
                    label = f'{cls} {conf:.2f} {self.track_id}'
                    plot_one_box(self.last_tracked_bb, img0, label=label, color=self.colors[0], line_thickness=1)
                    outputs.append([int(x1),int(y1),int(x2),int(y2)])
                    string_output += str(int(x1)) + ',' + str(int(y1)) + ',' + str(int(x2)) + ',' + str(int(y2)) + '#'   
  
                else: #no tracked objects
                    self.track_id = None
            else:
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if random.random() > self.no_detect_prob:
                        x1, y1,x2, y2 = (torch.FloatTensor(xyxy)).detach().cpu().numpy()#*image_size/grid_dim
                        if(conf > self.conf_threshold):
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
                            outputs.append([int(x1),int(y1),int(x2),int(y2)])
                            string_output += str(int(x1)) + ',' + str(int(y1)) + ',' + str(int(x2)) + ',' + str(int(y2)) + '#'
            if self.verbose:
                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        return string_output, outputs, img0
            
