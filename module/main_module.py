from .utils import *
from .darknet import Darknet
import cv2
from PIL import Image, ImageDraw, ImageFont
from .layer_to_boxes import layer_to_boxes as cuda
import torch

class Network():
    def __init__(self, cfgfile, weightfile, img_shape, conf_thresh=0.5, nms_thresh=0.4, batch_size=1, gpus=1):
        self.m = Darknet(cfgfile) # model <=> m
        self.m.print_network()
        self.m.load_weights(weightfile)
        self.batch = batch_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        if len(img_shape)==4:
            _, self.h, self.w, _ = img_shape
        else:
            self.h, self.w, _ = img_shape
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

        # 0.04523301124572754
        # t1 = time.time()
        output = output.view(self.batch*self.m.num_anchors, 5+self.m.num_classes, h*w).transpose(0,1).contiguous().view(5+self.m.num_classes, self.batch*self.m.num_anchors*h*w)
        # print(time.time()-t1)
        # t1 = time.time()
        grid_x = torch.linspace(0, w-1, w).cuda().repeat(h,1).repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w)
        grid_y = torch.linspace(0, h-1, h).cuda().repeat(w,1).t().repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w)
        # print(time.time()-t1)
        # t1 = time.time()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y
        # print(time.time()-t1)

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

        # t1 = time.time()
        all_boxes = cuda.layer_to_boxes(self.batch, w,h, self.m.num_anchors, self.w, self.h, self.conf_thresh, det_confs,cls_max_confs,cls_max_ids,xs,ys,ws,hs)
        # print(time.time()-t1)
        # print(np.array(all_boxes))
        # print(np.array(all_boxes).shape)
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

def plot_cv2_image(bboxes, img):
    draw = np.zeros_like(img)
    if len(bboxes)>1:
        for j in range(len(bboxes)):
            for i in bboxes[j]:
                print(i, tuple(i[:2]), tuple(i[2:4]))
                draw[j,...] = cv2.rectangle(img[j,...], tuple(i[:2]), tuple(i[2:4]), (255,0,0), 1)
    else:
         for i in bboxes[0]:
            print(i, tuple(i[:2]), tuple(i[2:4]))
            draw = cv2.rectangle(img, tuple(i[:2]), tuple(i[2:4]), (255,0,0), 1)       
    return draw


###########################################
################## Test ###################
###########################################
def test(*args):
    try:
        assert(len(args)==5)                                        # cfgfile, weightfile, gpus, batch_size, IterationTimes
        cfgfile, weightfile, gpus, batch_size, IterationTimes = args
        assert(type(cfgfile)==str)                                  # config file
        assert(type(weightfile)==str)                               # weights file
        assert(type(gpus)==int and gpus>0)                          # possible options: 1, other number to take all gpus
        assert(type(batch_size)==int and batch_size>0)              # batch size
        assert(type(IterationTimes)==int and IterationTimes>0)      # Iteration times

        img = cv2.imread('./data/person_1.jpg')
        # img = np.uint8(np.random.rand(512,512,3)*255) # input: stacked camera images with size of 512x512x3 and type is np.uint8
        if batch_size>1: img = np.repeat(img[None,:],batch_size,axis=0)
        net = Network(cfgfile, weightfile, img_shape=img.shape, conf_thresh=0.5, nms_thresh=0.4, batch_size=batch_size, gpus=gpus)

        if IterationTimes==1:
            bboxes = net.return_predict(img)
            output = plot_cv2_image(bboxes, img)
            if len(output.shape)==4: output_img = output[0,...]
            else: output_img = output
            cv2.imshow('   output image   ', output_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
        else:
            meant=0
            bboxes = net.return_predict(img)
            times = IterationTimes
            for i in range(times):
                print(i)
                t1=time.time()
                bboxes = net.return_predict(img)
                meant += time.time()-t1
            meant /= times
            print("{:.4f} fps".format(1/meant))

    except AssertionError:
        print('To test the variable -img- can be used over a real input image or a random noise')
        print('Usage:')
        print('    run main_module.py cfgfile weightfile GPUs BatchSize IterationTimes')
        print('')
        print('')
        print('cfgfile:\tYolo configuration file')
        print('weightfile:\tYolo weights file')
        print('GPUs:\t\tnumber of GPUs employed to run Yolo')
        print('BatchSize:\tNumber of camera images stacked in one batch')
        print('IterationTimes:\tK Iterations for time processing testing (optional)')
        print('')
        print('Exmaple:')
        print('    run main_module.py cfg/yolo_person.cfg backup/yolo_person.weights 2 128 10')
        print('    perform detection on multiple cameras using pipeline workflow and compute the mean time in 10 iterations')
        print('or')
        print('    run main_module.py cfg/yolo_person.cfg backup/yolo_person.weights 2 1 1 ')
        print('    perform detection on one camera using pipeline workflow')