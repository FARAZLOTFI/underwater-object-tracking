from torchsummary import summary
from torch.utils.data import DataLoader

from src.scuba_tracking.scuba_tracking.models.vision_model import scuba_detector
from src.scuba_tracking.scuba_tracking.models.utils.datasets import *
from src.scuba_tracking.scuba_tracking.models.utils.utils import *

grid_dim = 52
image_size = 416
hyp = {'giou': 3.31,  # giou loss gain
       'cls': 42.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 52.0,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.213,  # iou training threshold
       'lr0': 0.00261,  # initial learning rate (SGD=1E-3, Adam=9E-5)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.949,  # SGD momentum
       'weight_decay': 0.000489,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0103,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.691,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.433,  # image HSV-Value augmentation (fraction)
       'degrees': 1.43,  # image rotation (+/- deg)
       'translate': 0.0663,  # image translation (+/- fraction)
       'scale': 0.11,  # image scale (+/- gain)
       'shear': 0.384}  # image shear (+/- deg)

def test_webcam(names_path=None, DNN_model=None, img_size=416,iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5):
    #names = load_classes(names_path)  # class names
    cap = cv2.VideoCapture(0)
    DNN_model.eval()
    while (1):
        ret, frame = cap.read()
        #frame = frame[300:350,200:400,:]
        #frame = cv2.imread('/home/faraz/coco_balls/images/train2014/COCO_train2014_000000166047.jpg')
        ##frame = cv2.imread('/home/faraz/Aqua_dataset/train_dataset/simulator_dataset/images/train/155_jpg.rf.809def085c9b603b66809cd4d2321f19.jpg')
        if ret:
            w = frame.shape[1]
            h = frame.shape[0]
            width = w
            height = h
            img_ , _,_ ,_  = letterbox(frame, img_size, mode='square')
            frame, ratio, padw, padh = letterbox(frame, img_size, mode='square')

            #cv2.imshow('Input to YOLO', frame)
            #cv2.waitKey(10000)
            frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            frame = np.ascontiguousarray(frame, dtype=np.float32)  # uint8 to float32
            frame /= 255.0  # 0 - 255 to 0.0 - 1.0
            imgs = []
            imgs.append(frame)
            imgs = torch.from_numpy(np.array(imgs)).cuda()
            pred = DNN_model(imgs)
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
                    #b1_x1, b1_x2 = pbox[0] - pbox[2] / 2, pbox[0] + pbox[2] / 2
                    #b1_y1, b1_y2 = pbox[1] - pbox[3] / 2, pbox[1] + pbox[3] / 2
                    b1_x1, b1_y1,b1_x2, b1_y2 = (torch.FloatTensor(pbox)).detach().cpu().numpy()#*image_size/grid_dim
                    if(pconf>0.02):
                        print(pconf, pcls,pcls_conf)
                        img_ = cv2.rectangle(img_, (int(b1_x1),int(b1_y1)), (int(b1_x2),int(b1_y2)), (255,0,0), 1)

                    cv2.imshow('Incoming frames', img_)
                    cv2.waitKey(1)
                    ###input(' ')
                    #print('predictions: ',b1_y1,b1_x2,b1_y1,b1_y2)
                    #print('Predicted class: ',pcls)

def process_single_frame(DNN_model=None,frame=None, img_size=416,iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5, debug_=False):

    DNN_model.eval()
    w = frame.shape[1]
    h = frame.shape[0]
    width = w
    height = h
    img_ , _,_ ,_  = letterbox(frame, img_size, mode='square')
    frame, ratio, padw, padh = letterbox(frame, img_size, mode='square')

    #cv2.imshow('Input to YOLO', frame)
    #cv2.waitKey(10000)
    frame = frame[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    frame = np.ascontiguousarray(frame, dtype=np.float32)  # uint8 to float32
    frame /= 255.0  # 0 - 255 to 0.0 - 1.0
    imgs = []
    outputs = []
    string_output = '#'
    imgs.append(frame)
    imgs = torch.from_numpy(np.array(imgs)).cuda()
    pred = DNN_model(imgs)
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
            #b1_x1, b1_x2 = pbox[0] - pbox[2] / 2, pbox[0] + pbox[2] / 2
            #b1_y1, b1_y2 = pbox[1] - pbox[3] / 2, pbox[1] + pbox[3] / 2
            b1_x1, b1_y1,b1_x2, b1_y2 = (torch.FloatTensor(pbox)).detach().cpu().numpy()#*image_size/grid_dim
            if(pconf>0.02):
                img_ = cv2.rectangle(img_, (int(b1_x1),int(b1_y1)), (int(b1_x2),int(b1_y2)), (255,0,0), 1)
                outputs.append([int(b1_x1),int(b1_y1),int(b1_x2),int(b1_y2)])
                # string_output e.g. '#1,2#10,20#100,200#'
                string_output += str(int(b1_x1)) + ',' + str(int(b1_y1)) + ',' + str(int(b1_x2)) + ',' + str(int(b1_y2)) + '#'
    if debug_:
        cv2.imshow('Incoming frames', img_)
        cv2.waitKey(1)
    return string_output, outputs, img_

def post_processing(pred,im0,ng = 32): # ng stands for the number of grids (default :32)
    # anchors
    for i, det in enumerate(pred):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(512, det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, _, cls in det:
                print(xyxy, cls, conf)

                if True:  # Add bbox to image
                    label = '%s %.2f' % (str(cls), conf)
                    plot_one_box(xyxy, im0, label=label, color=(255,0,0))

        print('%sDone. (%.3fs)' % (s, time.time() - t))

        # Stream results
        cv2.imshow('result', im0)
        cv2.waitKey(1)


def test(names_path=None, test_path=None,
         DNN_model=None,
         nc=2,
         weights=None,
         batch_size=1,
         img_size=512,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False):
    # Initialize/load model and set device

    # Configure run
    #test_path = data['valid']  # path to test images
    names = load_classes(names_path)  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    batch_size = min(batch_size, len(dataset))
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0,
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    DNN_model.eval()
    DNN_model.ball_detector.eval()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        x, pred = DNN_model(imgs)
        inf_out, train_out = pred

        # Compute loss
        if True:#hasattr(DNN_model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, DNN_model.ball_detector, nc=nc, hyp=hyp)[1][:3].cpu()  # GIoU, obj, cls

        # Run NMS
        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Append to pycocotools JSON dictionary

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)
            if nl:
                detected = []
                tcls_tensor = labels[:, 0]

                # target boxes

                tbox = xywh2xyxy(labels[:, 1:5])

                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height
                # Search for correct predictions
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # To modify the coordinates according to the 32by32 grid output
                    pbox = (torch.FloatTensor(pbox)) # image_size/grid_dim
                    
                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)
                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    load_from_checkpoint = True
    model = scuba_detector().cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if load_from_checkpoint:
        MODEL_CHECKPOINT = './trained_model/training_checkpoint'
        model.load_state_dict(torch.load(MODEL_CHECKPOINT, map_location=device)['model_state_dict'], strict=True)


    summary(model, (3, 416, 416))

    list_images_file = './images_list_aqua.txt'
    data_path = './aqua.names' # it contains the classes names
    model.eval()
    with torch.no_grad():
        #test(names_path=data_path, test_path=list_images_file, DNN_model=model,nc=1, weights=None, batch_size=12, img_size=512, iou_thres=0.01,
        #     conf_thres=0.001, nms_thres=0.5)
        test_webcam(names_path=data_path,DNN_model=model,nms_thres=0.001)

