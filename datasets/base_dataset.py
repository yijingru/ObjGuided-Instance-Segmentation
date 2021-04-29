import numpy as np
import os
import torch.utils.data as data
import torch
import cv2
from . import pre_proc, transforms, affine_funcs
from skimage import measure



class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=None, input_w=None, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ['__background__', 'cell']
        self.num_classes = len(self.class_name)-1
        self.img_dir = os.path.join(data_dir, phase)
        self.img_ids = sorted(os.listdir(self.img_dir))
        self.max_objs = 500

    def load_image(self, index):
        return ""


    def load_gt_masks(self, annopath):
        return []


    def load_annoFolder(self, img_id):
        return ""


    def load_gt_bboxes(self, annopath):
        bboxes = []
        masks = self.load_gt_masks(annopath)
        for mask in masks:
            r, c = np.where(mask > 0.)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])
        return np.asarray(bboxes, np.float32)

    def load_gt_masks_bboxes(self, annopath):
        bboxes = []
        masks = self.load_gt_masks(annopath)
        for mask in masks:
            r, c = np.where(mask > 0.)
            if len(r):
                y1 = np.min(r)
                x1 = np.min(c)
                y2 = np.max(r)
                x2 = np.max(c)
                if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
                    continue
                bboxes.append([x1, y1, x2, y2])
        return np.asarray(masks, np.float32), np.asarray(bboxes, np.float32)

    def find_maximum_mask(self, mask):
        mask = np.where(mask>0., 1., 0.)
        labels = measure.label(mask, connectivity=1)
        props = measure.regionprops(labels)
        props = sorted(props, key=lambda x: x.area, reverse=True)  # descending order
        if len(props)==0:
            return None
        else:
            return np.asarray(np.where(props[0].label == labels, 1., 0.), np.float32)

    def sample_ROI(self, mask, rbox):
        ROI = affine_funcs.sample_masks(mask, rbox, numpy=True)
        return ROI


    def masks_to_bboxes_rois(self, gt_masks):
        out_bboxes = []
        out_rois = []
        for mask in gt_masks:
            mask = self.find_maximum_mask(mask)
            if mask is None:
                continue
            r, c = np.where(mask==1.)
            h, w = mask.shape
            y1 = np.maximum(0, np.min(r))
            x1 = np.maximum(0, np.min(c))
            y2 = np.minimum(h-1, np.max(r))
            x2 = np.minimum(w-1, np.max(c))
            if y2-y1>2 and x2-x1>2:
                rbox = np.asarray([float(x1+x2)/2, float(y1+y2)/2,
                                   float(x2-x1+1), float(y2-y1+1)], np.float32)  #[cenx, ceny, w, h]
                rbox_padding = rbox.copy()
                rbox_padding[2:] *= 1.1
                roi = self.sample_ROI(mask, rbox_padding)
                if roi.shape[0]>1 and roi.shape[1]>1:
                    out_bboxes.append(rbox)
                    out_rois.append(roi)
        return np.asarray(out_bboxes, np.float32), out_rois


    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annoFolder = self.load_annoFolder(img_id)
        masks = self.load_gt_masks(annoFolder)
        return masks

    def image_trans(self, image):
        image = cv2.resize(image, (self.input_w, self.input_h))
        image = image.astype(np.float32) / 255.
        image = image - 0.5
        image = image.transpose(2, 0, 1)
        return torch.from_numpy(image)

    def data_preparation(self, image, gt_masks, augment=False):
        H,W,C = image.shape
        if augment:
            crop_size, crop_center = transforms.random_crop_info(W, H)
            image, gt_masks, crop_center = transforms.random_flip(image, gt_masks, crop_center)
        else:
            crop_center = np.array([W / 2., H / 2.], dtype=np.float32)
            crop_size = [max(W, H), max(W, H)]
        M = transforms.load_affine_matrix(crop_center, crop_size, (self.input_w, self.input_h))
        image = np.asarray(image, np.float32)
        image_warp = cv2.warpAffine(image, M, dsize=(self.input_w, self.input_h), flags=cv2.INTER_LINEAR)
        gt_masks = np.transpose(gt_masks, (1,2,0))
        gt_masks_warp = cv2.warpAffine(gt_masks, M,dsize=(self.input_w, self.input_h), flags=cv2.INTER_NEAREST)
        if gt_masks_warp.ndim == 2:
            gt_masks_warp = gt_masks_warp[:, :, np.newaxis]  # keep 3 ndims
        gt_masks_warp = np.transpose(gt_masks_warp, (2, 0, 1))
        return np.asarray(image_warp, np.float32), np.asarray(gt_masks_warp, np.float32)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = self.load_image(index)
        if self.phase == 'test':
            return {'image': self.image_trans(image),
                    'img_id': img_id}
        else:
            gt_masks = self.load_annotation(index)   # num_obj x h x w
            if self.phase == 'train':
                image, gt_masks = self.data_preparation(image, gt_masks, augment=True)
            else:
                image, gt_masks = self.data_preparation(image, gt_masks, augment=False)

            gt_bboxes, gt_rois = self.masks_to_bboxes_rois(gt_masks)

            image = self.image_trans(image)
            gt_bboxes2 = gt_bboxes.copy()
            gt_bboxes2 /= self.down_ratio

            data_dict = pre_proc.generate_ground_truth(gt_bboxes2=gt_bboxes2,
                                                       num_classes=self.num_classes,
                                                       image_h=self.input_h//self.down_ratio,
                                                       image_w=self.input_w//self.down_ratio,
                                                       max_objs=self.max_objs)
            for name in data_dict:
                data_dict[name] = torch.from_numpy(data_dict[name])
            data_dict['image'] = image
            data_dict['gt_bboxes'] = gt_bboxes
            data_dict['gt_rois'] = gt_rois
            return data_dict

    def __len__(self):
        return len(self.img_ids)
