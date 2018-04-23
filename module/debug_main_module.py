import cv2
import sys
import time
import torch
import numpy as np
from .main_module import Network
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
        detections = self.net.do_detect(TorchBatch)
        self.assertEqual(detections, [[]]*len(detections))
    def test_do_detection_real_image(self):
        img_vector = []
        for i in range(self.batch_size):
            img_vector.append( cv2.resize(self.img[i], (self.net.m.width, self.net.m.height)) )
        sized = np.array(img_vector)
        detections = self.net.do_detect(sized)
        detections = np.array(detections).shape
        self.assertTrue(detections[1]>=  2  ) # Good Acurracy from 3 to 8, CHANGE THIS TO AVOID ERRORS
        self.assertEqual(detections[0],self.batch_size)
        self.assertEqual(detections[2],7)
        print("   --   Accuracy test: {}/8 Detections and batch size: {}".format(detections[1],detections), end='   ')
    def test_net(self):
        detections = self.net.return_predict(self.img)
        self.assertEqual(len(detections),self.batch_size)
    def test_time_processing_N_iterations(self):
        meant=0
        bboxes = self.net.return_predict(self.img)
        for i in range(self.times):
            t1=time.time()
            bboxes = self.net.return_predict(self.img)
            meant += time.time()-t1
        meant /= self.times
        print("{:.4f} fps  -- ".format(1/meant), end='   ')    




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
        img = np.repeat(img[None,:],batch_size,axis=0)
        net = Network(cfgfile, weightfile, img_shape=img.shape, conf_thresh=0.5, nms_thresh=0.4, batch_size=batch_size, gpus=gpus)

        print('Initializing tests')
        suite = unittest.TestSuite()
        suite.addTest(DetectionTests('test_image_batch_size', net, img, cfgfile, weightfile, gpus, batch_size, IterationTimes))
        suite.addTest(DetectionTests('test_do_detection_random_noise', net, img, cfgfile, weightfile, gpus, batch_size, IterationTimes))
        suite.addTest(DetectionTests('test_do_detection_real_image', net, img, cfgfile, weightfile, gpus, batch_size, IterationTimes))
        suite.addTest(DetectionTests('test_net', net, img, cfgfile, weightfile, gpus, batch_size, IterationTimes))
        suite.addTest(DetectionTests('test_time_processing_N_iterations', net, img, cfgfile, weightfile, gpus, batch_size, IterationTimes))
        unittest.TextTestRunner(verbosity=2).run(suite)

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
        print('    import module.debug_main_module as debug_module')
        print('')
        print('    debug_module.test(cfg/yolo_person.cfg, backup/yolo_person.weights, 2, 8, 10)  ')
        print('    perform detection on multiple cameras using pipeline workflow and compute the mean time in 10 iterations')
        print('or')
        print('    debug_module.test(cfg/yolo_person.cfg, backup/yolo_person.weights, 2, 1, 1)   ')
        print('    perform detection on one camera using pipeline workflow')