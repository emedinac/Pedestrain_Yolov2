from utils import *
from darknet import Darknet
import cv2

import torch

class Network():
    def __init__(self, cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, batch_size=1,gpus=1):
        self.m = Darknet(cfgfile) # model <=> m
        self.m.print_network()
        self.m.load_weights(weightfile)
        self.batch = batch_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        print('Loading weights from %s... Done!' % (weightfile))

        if self.m.num_classes == 20:
            namesfile = 'data/voc.names'
        elif self.m.num_classes == 80:
            namesfile = 'data/coco.names'
        else:
            namesfile = 'data/person.names'
        class_names = load_class_names(namesfile)
     
        self.m.cuda()
        self.m.eval()
        if gpus>1: self.net = torch.nn.DataParallel(self.m, device_ids=range(torch.cuda.device_count()))
        else: self.net = self.m

    def get_region_boxes(self, output):
        anchor_step = len(self.m.anchors)//self.m.num_anchors
        assert(output.size(1) == (5+self.m.num_classes)*self.m.num_anchors)
        h = output.size(2)
        w = output.size(3)
        

        all_boxes = []
        output = output.view(self.batch*self.m.num_anchors, 5+self.m.num_classes, h*w).transpose(0,1).contiguous().view(5+self.m.num_classes, self.batch*self.m.num_anchors*h*w)

        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w).cuda()
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w).cuda()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y

        anchor_w = torch.Tensor(self.m.anchors).view(self.m.num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.m.anchors).view(self.m.num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch, 1).repeat(1, 1, h*w).view(self.batch*self.m.num_anchors*h*w).cuda()
        anchor_h = anchor_h.repeat(self.batch, 1).repeat(1, 1, h*w).view(self.batch*self.m.num_anchors*h*w).cuda()
        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax(dim=-1)(Variable(output[5:5+self.m.num_classes].transpose(0,1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        sz_hw = h*w
        sz_hwa = sz_hw*self.m.num_anchors
        det_confs = convert2cpu(det_confs)
        cls_max_confs = convert2cpu(cls_max_confs)
        cls_max_ids = convert2cpu_long(cls_max_ids)
        xs = convert2cpu(xs)
        ys = convert2cpu(ys)
        ws = convert2cpu(ws)
        hs = convert2cpu(hs)

        # cython optimization is required in this for sections.
        for b in range(self.batch):
            boxes = []
            for cy in range(h):
                for cx in range(w):
                    for i in range(self.m.num_anchors):
                        ind = b*sz_hwa + i*sz_hw + cy*w + cx
                        det_conf =  det_confs[ind]
                        conf =  det_confs[ind]
        
                        if conf > self.conf_thresh:
                            bcx = xs[ind]
                            bcy = ys[ind]
                            bw = ws[ind]
                            bh = hs[ind]
                            cls_max_conf = cls_max_confs[ind]
                            cls_max_id = cls_max_ids[ind]
                            box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]

                            boxes.append(box)
            all_boxes.append(boxes)
        return all_boxes

    def do_detect(self, img, use_cuda=1):
        if type(img) == np.ndarray: # cv2 image
            img = torch.from_numpy(img.transpose(0,3,1,2)).float().div(255.0)
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        output = self.net(img)

        output = output.data
        all_boxes = self.get_region_boxes(output)
        final_boxes = []
        for i in range(output.size(0)):
            boxes = all_boxes[i]
            final_boxes.append(nms(boxes, self.nms_thresh))
        return final_boxes

    def return_predict(self,img):
        if self.batch>1:
            img_vector = []
            for i in range(self.batch):
                img_vector.append( cv2.resize(img[i], (self.m.width, self.m.height)) )
            sized = np.array(img_vector)
        else:
            sized = cv2.resize(img, (self.m.width, self.m.height))
            sized = sized[None,:]
        bboxes = self.do_detect(sized, use_cuda=1)
        return bboxes


############################################
################### Test ###################
############################################
# if __name__ == '__main__':
#     # import torch
#     cfgfile = sys.argv[1] # config file
#     weightfile = sys.argv[2] # weights file
#     gpus = float(sys.argv[3]) # possible options: 1, other number to take all gpus
#     batch_size = int(float(sys.argv[4])) # batch size
#     if len(sys.argv) == 5:
#         net = Network(cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, batch_size=batch_size, gpus=gpus)
#         img = np.uint8(np.random.randn(batch_size,512,512,3)) # input: stacked camera images with size of 512x512x3 and type is np.uint8
#         bboxes = net.return_predict(img)
#     elif len(sys.argv) == 6:
#         net = Network(cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, batch_size=batch_size, gpus=gpus)
#         img = cv2.imread('./VOCdevkit/VOC2007/JPEGImages/000001.jpg')
#         if batch_size>1: img = np.repeat(img[None,:],batch_size,axis=0)
#         meant=0
#         bboxes = net.return_predict(img)
#         times = int(float(sys.argv[5]))
#         for i in range(times):
#             print(i)
#             t1=time.time()
#             bboxes = net.return_predict(img)
#             meant += time.time()-t1
#         meant /= times
#         print("{:.4f} fps".format(1/meant))
#     else:
#         print('Usage:')
#         print('    run main_module.py cfgfile weightfile GPUs BatchSize IterationTimes')
#         print('Exmaple:')
#         print('    run main_module.py cfg/yolo_person.cfg backup/yolo_person.weights 2 128 100')
#         print('    perform detection on multiple cameras using pipeline workflow')