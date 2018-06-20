### Experimenting with Yolov2 in pytorch
This experiment was performed based on (almost totally) the code available in this link https://github.com/marvis/pytorch-yolo2. My contributions are main_module and debug_main_module files and 0.4 pytorch support (validated only in module) for a pipeline development, or in case Yolo works as a module (python module) in a higher top level code. Cut and Paste this module after follow the steps bellow.

##### Download Weights for pedestrian detection into backup folder
```
mkdir backup
cd backup
wget https://drive.google.com/open?id=1b-g-Jg6cN8Gya4yEsV-60odkJCzzVwWJ
```
If you will use the module, backup must be inside the module folder.
```
cd module
mkdir backup
cd backup
wget https://drive.google.com/open?id=1b-g-Jg6cN8Gya4yEsV-60odkJCzzVwWJ
```
---
##### Test the Pedestrian Detection Module Using a Pre-Trained Model
By now, you already can test this module by means of this command
```
import module_gpu.main_module as yolov2
yolov2.test('cfg/yolo_person.cfg', 'backup/yolo_person.weights', 2, 16, 10) # run this 10 times
```
or 
```
import module_gpu.main_module as yolov2
yolov2.test('cfg/yolo_person.cfg', 'backup/yolo_person.weights', 2, 16, 1) # run only 1 time
```
In case you want debug, try with the following code:
```
import module_gpu.debug_main_module as debug_yolov2
debug_yolov2.test('cfg/yolo_person.cfg', 'backup/yolo_person.weights', 2, 16, 1) # run only 1 time
```
---
#### Pedestrian Detection Using A Pre-Trained Model
From this point, the code will run in the main section like original github, but in this case used for pedestrian detection. be sure about the path of "yolo_person.weights" file.
```
python detect.py cfg/yolo_person.cfg yolo_person.weights data/person.jpg
```
---
#### Real-Time Detection on a Webcam
```
python demo.py cfg/yolo_person.cfg yolo_person.weights
```
---
#### Training YOLO on VOC
##### Get The Pascal VOC Data
```
wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
```
##### Generate Labels for VOC
run the following command to have the complete test set but only using person detections
```
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
run the following command to have a test set of person class only.
```
python voc_label_only_class.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Extract Pretrained Convolutional Weights
run the following command, where 32 is the conv-layer number:
```
python partial.py cfg/yolo_person.cfg backup/yolo_person.weights new_weights.conv.32 32
```
##### Train The Model
```
python train.py cfg/person.data cfg/yolo_person.cfg backup/yolo_person.weights
```
or
```
python train.py cfg/person.data cfg/yolo_person.cfg new_weights.conv.32
```
To freeze some layers and do a fine-tunning, just run the following command in train.py:
```
Darknet(cfgfile,FREEZE=n_layer=2) # n_layer=2, 6, 10, ...
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo_person.cfg backup/yolo_person.weights
python scripts/voc_eval.py results/comp4_det_test_
```

Precision (or mAP using the non official code) test on this release model is 76.5% for person class on the complete dataset.


##### To DO
- [X] Yolo Module codes to join the detector to tracking algorithm stages.
- [ ] Use the Non-Maximum Supression (nms) implemented in GPU instead of common CPU mode to speed up the training phase when training from scratch.


---
#### License
MIT License (see LICENSE file).
