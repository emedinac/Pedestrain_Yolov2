from ..utils import *

def layer_to_boxes(batch, w,h, num_anchors, sizes, conf_thresh, det_confs,cls_max_confs,cls_max_ids,xs,ys,ws,hs):
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
    # cls_max_ids = cls_max_ids.long()
    # cython optimization is required in this for sections.
    for (hi, wi), b in zip(sizes, range(batch)):
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
                        cls_max_conf = cls_max_confs[ind].tolist()
                        cls_max_id = cls_max_ids[ind].tolist()
                        # box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        box = [ int(((bcx-bw/2.0)/w*wi).round()), 
                                int(((bcy-bh/2.0)/h*hi).round()), 
                                int(((bcx+bw/2.0)/w*wi).round()), 
                                int(((bcy+bh/2.0)/h*hi).round()), 
                                det_conf.tolist(), cls_max_conf, cls_max_id]
                        boxes.append(box)
        all_boxes.append(boxes)
    return all_boxes