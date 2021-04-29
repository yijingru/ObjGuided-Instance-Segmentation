import math
from .draw_gaussian import *
import numpy as np


def generate_ground_truth(gt_bboxes2,
                          num_classes,
                          image_h,
                          image_w,
                          max_objs):
    num_obj = min(gt_bboxes2.shape[0], max_objs)
    # generate ground truth
    hm = np.zeros((num_classes, image_h, image_w), dtype=np.float32)
    wh = np.zeros((max_objs, 2), dtype=np.float32)
    reg = np.zeros((max_objs, 2), dtype=np.float32)
    ind = np.zeros((max_objs), dtype=np.int64)
    reg_mask = np.zeros((max_objs), dtype=np.uint8)

    for k in range(num_obj):
        cen_x, cen_y, w, h = gt_bboxes2[k,:]
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        ct = np.asarray([cen_x, cen_y], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        draw_umich_gaussian(hm[0], ct_int, radius)
        wh[k] = 1. * w, 1. * h
        ind[k] = ct_int[1] * image_w + ct_int[0]
        reg[k] = ct - ct_int
        reg_mask[k] = 1

    ret = {'hm': hm,
           'reg_mask': reg_mask,
           'ind': ind,
           'wh': wh,
           'reg': reg,
           }

    return ret
