import numpy as np
import cv2

def rescale_rects(rects, down_ratio):
    # rects: num x 5 [cenx,ceny,w,h,angle]
    if rects.ndim==1:
        return []
    elif rects.ndim == 2:
        rects[:, :4] /= down_ratio
    else:
        rects[0, :, :4] /= down_ratio
    return rects

def contours_to_rect(contours):
    # contours: dtype-np.int32
    return cv2.minAreaRect(contours)

def point_affine_transform(pt, M):
    # new_x = M11x+M12y+M13
    # new_y = M21x+M22y+M23
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(M, new_pt)
    return new_pt

def Rotation_Transform(src_point, degree):
    radian = np.pi * degree / 180
    R_matrix = [[np.cos(radian), -np.sin(radian)],
                [np.sin(radian), np.cos(radian)]]
    R_matrix = np.asarray(R_matrix, dtype=np.float32)
    R_pts = np.matmul(R_matrix, src_point)
    return R_pts

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def load_affine_matrix(crop_center, crop_size, dst_size, inverse=False):
    # image: h x w x c original size
    # mask_rects: num_obj x 5 (cenx, ceny, w, h, angle)
    # dst_size = [out_w, out_h]

    dst_center = np.array([dst_size[0]//2, dst_size[1]//2], dtype=np.float32)

    src_1 = crop_center
    src_2 = crop_center + Rotation_Transform([0, -crop_size[0]//2], degree=0)
    src_3 = get_3rd_point(src_1, src_2)
    src = np.asarray([src_1, src_2, src_3], np.float32)

    dst_1 = dst_center
    dst_2 = dst_center + [0, -dst_center[0]]
    dst_3 = get_3rd_point(dst_1, dst_2)
    dst = np.asarray([dst_1, dst_2, dst_3], np.float32)
    if inverse:
        M = cv2.getAffineTransform(dst, src)
    else:
        M = cv2.getAffineTransform(src, dst)
    return M

def _get_border(size, border):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def random_crop_info(h, w):
    max_wh = max(h, w)
    random_size = max_wh * np.random.choice(np.arange(0.6, 1.4, 0.1))
    w_border = _get_border(size=w, border=128)
    h_border = _get_border(size=h, border=128)
    random_center_x = np.random.randint(low=w_border, high=w - w_border)
    random_center_y = np.random.randint(low=h_border, high=h - h_border)
    return [random_size, random_size], [random_center_x, random_center_y]


def random_flip(image, gt_masks, crop_center=None):
    # image: h x w x c
    # gt_masks: num_obj x h x w
    h,w,c = image.shape
    if np.random.random() < 0.5:
        image = image[:, ::-1, :]
        gt_masks = gt_masks[:,:,::-1]
        if crop_center is not None:
            crop_center[0] = w - crop_center[0] - 1
    return image, gt_masks, crop_center


