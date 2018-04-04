import cv2
import sys
import time
import torch
import numpy as np
from main_module import Network
import unittest

class DetectionTests(unittest.TestCase):
    def __init__(self, testName, net, img, cfgfile, weightfile, gpus, batch_size, times):
        super(DetectionTests, self).__init__(testName)
        self.img = img
        self.cfgfile = cfgfile
        self.weightfile = weightfile
        self.gpus = gpus
        self.batch_size = batch_size
        self.times = times
        self.net = net
    def test_image_batch_size(self):
        self.assertEqual(self.img.shape[1:], (375, 500, 3))
        self.assertEqual(self.img.shape[0], self.batch_size)
    def test_do_detection_random_noise(self):
        TorchBatch = torch.FloatTensor(self.batch_size, 3, self.net.m.width, self.net.m.height)
        detections = net.do_detect(TorchBatch)
        self.assertEqual(detections, [[]]*len(detections))
    def test_do_detection_real_image(self):
        sized = []
        for i in range(len(img)):
            sized.append(cv2.resize(self.img[i], (self.net.m.width, self.net.m.height)) )
        sized = np.array(sized)
        detections = net.do_detect(sized)
        detections = np.array(detections).shape
        self.assertTrue(detections[1]>=  3  ) # Good Acurracy from 3 to 8, CHANGE THIS TO AVOID ERRORS
        self.assertEqual(detections[0],self.batch_size)
        self.assertEqual(detections[2],7)
        print("   --   Accuracy test: {}/8 Detections and batch size: {}".format(detections[1],detections), end='   ')
    def test_net(self):
        detections = net.return_predict(img)
        self.assertEqual(len(detections),self.batch_size)
    def test_time_processing_N_iterations(self):
        meant=0
        bboxes = net.return_predict(img)
        for i in range(times):
            t1=time.time()
            bboxes = net.return_predict(img)
            meant += time.time()-t1
        meant /= times
        print("{:.4f} fps  -- ".format(1/meant), end='   ')    

if __name__ == '__main__':
    cfgfile = sys.argv[1] # config file
    weightfile = sys.argv[2] # weights file
    gpus = float(sys.argv[3]) # possible options: 1, other number to take all gpus
    batch_size = int(float(sys.argv[4])) # batch size
    img = cv2.imread('./VOCdevkit/VOC2007/JPEGImages/000377.jpg') # input: stacked camera images with size of 512x512x3 and type is np.uint8
    if batch_size>1: img = np.repeat(img[None,:],batch_size,axis=0)
    times = int(float(sys.argv[5]))
    net = Network(cfgfile, weightfile, conf_thresh=0.5, nms_thresh=0.4, batch_size=batch_size, gpus=gpus)

    print('Initializing tests')
    if len(sys.argv) == 6:
        suite = unittest.TestSuite()
        suite.addTest(DetectionTests('test_image_batch_size', net, img, cfgfile, weightfile, gpus, batch_size, times))
        suite.addTest(DetectionTests('test_do_detection_random_noise', net, img, cfgfile, weightfile, gpus, batch_size, times))
        suite.addTest(DetectionTests('test_do_detection_real_image', net, img, cfgfile, weightfile, gpus, batch_size, times))
        suite.addTest(DetectionTests('test_net', net, img, cfgfile, weightfile, gpus, batch_size, times))
        suite.addTest(DetectionTests('test_time_processing_N_iterations', net, img, cfgfile, weightfile, gpus, batch_size, times))
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        print('Usage:')
        print('    run main_module.py cfgfile weightfile GPUs BatchSize IterationTimes')
        print('Exmaple:')
        print('    run main_module.py cfg/yolo_person.cfg backup/yolo_person.weights 2 128 10')
        print('    perform detection on multiple cameras using pipeline workflow')
        print('or')
        print('    run main_module.py cfg/yolo_person.cfg backup/yolo_person.weights 2 128 1  ')
        print('    perform detection on one camera using pipeline workflow')
