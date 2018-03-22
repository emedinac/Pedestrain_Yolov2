### Experimenting with Yolov2 in pytorch
This experiment was performed based on (almost totally) the code available in this link https://github.com/marvis/pytorch-yolo2 .

##### Download Weights for pedestrian detection into backup folder
```
mkdir backup
cd backup
wget https://drive.google.com/open?id=1b-g-Jg6cN8Gya4yEsV-60odkJCzzVwWJ
```
---
#### Pedestrian Detection Using A Pre-Trained Model
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
```
python voc_label.py
cat 2007_train.txt 2007_val.txt 2012_*.txt > voc_train.txt
```
##### Extract Pretrained Convolutional Weights
run the following command, where 32 is the conv-layer number:
```
python partial.py cfg/yolo_person.cfg yolo_person.weights new_weights.conv.32 32
```
##### Train The Model
```
python train.py cfg/person.data cfg/yolo_person.cfg backup/yolo_person.weights
```
##### Evaluate The Model
```
python valid.py cfg/voc.data cfg/yolo_person.cfg backup/yolo_person.weights
python scripts/voc_eval.py results/comp4_det_test_
```

Precision (or mAP using the not official code) test on this release model is 76.5% for person class.


##### To DO
- [ ] Use the Non-Maximum Supression (nms) implemented in GPU instead of common CPU mode to speed up the training phase.


---
#### License
MIT License (see LICENSE file).
