import numpy as np
cimport cython

from utils import *

@cython.boundscheck(False)
@cython.wraparound(False)
def layer_to_boxes(batch, num_anchors, w,h, wi, hi, conf_thresh, det_confs,cls_max_confs,cls_max_ids,xs,ys,ws,hs):
    cdef Py_ssize_t b, cy, cx, i, sz_hw, sz_hwa
    # cdef long [:] det_confs, cls_max_confs, cls_max_ids, xs, ys, ws, hs

    all_boxes = []
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    # cython optimization is required in this for sections.
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    conf =  det_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        # box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        box = [ int(round((bcx-bw/2.0)/w*wi)), 
                                int(round((bcy-bh/2.0)/h*hi)), 
                                int(round((bcx+bw/2.0)/w*wi)), 
                                int(round((bcy+bh/2.0)/h*hi)), 
                                det_conf, cls_max_conf, cls_max_id]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes

def layer_to_boxes_o(batch, num_anchors, w,h, wi, hi, conf_thresh, det_confs,cls_max_confs,cls_max_ids,xs,ys,ws,hs):
    cdef Py_ssize_t b, cy, cx, i, sz_hw, sz_hwa

    all_boxes = []
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors

    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)
    # cython optimization is required in this for sections.
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    conf =  det_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]
                        # box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        box = [ int(round((bcx-bw/2.0)/w*wi)), 
                                int(round((bcy-bh/2.0)/h*hi)), 
                                int(round((bcx+bw/2.0)/w*wi)), 
                                int(round((bcy+bh/2.0)/h*hi)), 
                                det_conf, cls_max_conf, cls_max_id]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes