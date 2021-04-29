import numpy as np
import os
import cv2
from .base_dataset import BaseDataset
import glob


class Kaggle(BaseDataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(Kaggle, self).__init__(data_dir, phase, input_h, input_w, down_ratio)
        self.class_name = ['__background__', 'kaggle']
        self.num_classes = len(self.class_name)-1

    def load_image(self, index):
        img_id = self.img_ids[index]
        imgFile = os.path.join(self.img_dir, img_id, "images", img_id+'.png')
        img = cv2.imread(imgFile)
        return img

    def load_gt_bboxes(self, annopath):
        bboxes = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*.png"))):
            mask = cv2.imread(annoImg, -1)
            r,c = np.where(mask>0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2-y1)<=1 or abs(x2-x1)<=1):
                    continue
                bboxes.append([x1, y1, x2, y2])
        return np.asarray(bboxes, np.float32)

    def load_gt_masks(self, annopath):
        masks = []
        for annoImg in sorted(glob.glob(os.path.join(annopath, "*.png"))):
            mask = cv2.imread(annoImg, -1)
            r, c = np.where(mask > 0)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                masks.append(np.where(mask > 0, 1., 0.))
        return np.asarray(masks, np.float32)

    def load_annoFolder(self, img_id):
        return os.path.join(self.img_dir, img_id, "masks")
