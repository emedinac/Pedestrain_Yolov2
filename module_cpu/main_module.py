from .utils import *
from .darknet import Darknet
import cv2
from PIL import Image, ImageDraw, ImageFont
from .layer_to_boxes import layer_to_boxes as cuda
import torch

class Network():
    def __init__(self, cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, gpus=1):
        self.m = Darknet(cfgfile) # model <=> m
        self.m.print_network()
        self.m.load_weights(weightfile)
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
     
        self.m.eval()
        self.net = self.m

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
        grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w)
        grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(self.batch*self.m.num_anchors, 1, 1).view(self.batch*self.m.num_anchors*h*w)
        # print(time.time()-t1)
        # t1 = time.time()
        xs = torch.sigmoid(output[0]) + grid_x
        ys = torch.sigmoid(output[1]) + grid_y
        # print(time.time()-t1)

        anchor_w = torch.Tensor(self.m.anchors).view(self.m.num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
        anchor_h = torch.Tensor(self.m.anchors).view(self.m.num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
        anchor_w = anchor_w.repeat(self.batch, 1).repeat(1, 1, h*w).view(self.batch*self.m.num_anchors*h*w)
        anchor_h = anchor_h.repeat(self.batch, 1).repeat(1, 1, h*w).view(self.batch*self.m.num_anchors*h*w)
        ws = torch.exp(output[2]) * anchor_w
        hs = torch.exp(output[3]) * anchor_h

        det_confs = torch.sigmoid(output[4])

        cls_confs = torch.nn.Softmax(dim=-1)(Variable(output[5:5+self.m.num_classes].transpose(0,1))).data
        cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
        cls_max_confs = cls_max_confs.view(-1)
        cls_max_ids = cls_max_ids.view(-1)

        # t1 = time.time()
        all_boxes = cuda.layer_to_boxes(self.batch, w,h, self.m.num_anchors, self.vector_sizes, self.conf_thresh, det_confs,cls_max_confs,cls_max_ids,xs,ys,ws,hs)
        # print(time.time()-t1)
        return all_boxes

    def do_detect(self, img):
        if type(img) == np.ndarray: # cv2 image
            img = torch.from_numpy(img.transpose(0,3,1,2)).float().div(255.0)
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
        img_vector = []
        self.vector_sizes = []
        self.batch = img.shape[0]
        for i in range(self.batch):
            self.vector_sizes.append(img[i].shape[:2])
            img_vector.append( cv2.resize(img[i], (self.m.width, self.m.height)) )
        sized = np.array(img_vector)
        bboxes = self.do_detect(sized)
        return bboxes

def plot_cv2_image(bboxes, img):
    draw = np.zeros_like(img)
    for s, b in enumerate(bboxes):
        for i in b:
            print(i, tuple(i[:2]), tuple(i[2:4]))
            draw[s] = cv2.rectangle(img[s], tuple(i[:2]), tuple(i[2:4]), (255,0,0), 1)     
    return draw


###########################################
################## Test ###################
###########################################
def test(*args):
    try:
        assert(len(args)==5)                                                # cfgfile, weightfile, gpus, batch_size, IterationTimes
        cfgfile, weightfile, gpus, batch_size_FOR_TEST, IterationTimes = args
        assert(type(cfgfile)==str)                                          # config file
        assert(type(weightfile)==str)                                       # weights file
        assert(type(gpus)==int and gpus>0)                                  # possible options: 1, other number to take all gpus
        assert(type(batch_size_FOR_TEST)==int and batch_size_FOR_TEST>0)    # batch size
        assert(type(IterationTimes)==int and IterationTimes>0)              # Iteration times

        img = cv2.imread('./data/person_1.jpg')
        # img = np.uint8(np.random.rand(512,512,3)*255) # input: stacked camera images with size of 512x512x3 and type is np.uint8
        img = np.repeat(img[None,:],batch_size_FOR_TEST,axis=0)
        net = Network(cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, gpus=gpus)

        if IterationTimes==1:
            bboxes = net.return_predict(img)
            output = plot_cv2_image(bboxes, img)
            output_img = output[-1]
            cv2.imwrite('output_image.png', output_img)
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
        print('To test, the variable -img- can be used over a real input image or a random noise')
        print('The input arguments in test function are the following:                          ')
        print('cfgfile weightfile GPUs BatchSize IterationTimes')
        print('')
        print('cfgfile:\tYolo configuration file')
        print('weightfile:\tYolo weights file')
        print('GPUs:\t\tnumber of GPUs employed to run Yolo')
        print('BatchSize:\tNumber of camera images stacked in one batch')
        print('IterationTimes:\tK Iterations for time processing testing (optional)')
        print('')
        print('')
        print('Usage:')
        print('    import module.main_module as module')
        print('')
        print('    module.test(cfg/yolo_person.cfg, backup/yolo_person.weights, 2, 8, 10)  ')
        print('    perform detection on multiple cameras using pipeline workflow and compute the mean time in 10 iterations')
        print('or')
        print('    module.test(cfg/yolo_person.cfg, backup/yolo_person.weights, 2, 1, 1)   ')
        print('    perform detection on one camera using pipeline workflow')