import torch
import numpy as np
import torch.nn.functional as F
from scipy.sparse import coo_matrix

def create_rect_grids(bbox):
    # return: C x h x w x 2
    cen_x, cen_y, w, h = bbox
    x1 = cen_x - w / 2
    y1 = cen_y - h / 2
    x2 = cen_x + w / 2
    y2 = cen_y + h / 2
    x = np.linspace(x1, x2, num=int(w))
    y = np.linspace(y1, y2, num=int(h))
    grids = np.stack([np.repeat(x[np.newaxis, :], y.shape[0], axis=0),
                      np.repeat(y[:, np.newaxis], x.shape[0], axis=1)], axis=2)
    grids = grids[np.newaxis, :, :, :]  # (1, 18, 13, 2)
    return np.asarray(grids, np.float32)


def create_pytorch_grids(bbox, IW, IH):
    # input: N x C x H x W
    # grid shape: N x H_ x W_x 2
    grids = create_rect_grids(bbox)
    grids[:, :, :, 0] = (grids[:, :, :, 0] / (IW - 1)) * 2 - 1
    grids[:, :, :, 1] = (grids[:, :, :, 1] / (IH - 1)) * 2 - 1
    return torch.from_numpy(grids)

def sample_masks(mask, bbox, numpy=False):
    # mask: H x W
    # grid shape: N x H_ x W_x 2
    grid = create_pytorch_grids(bbox, IW=mask.shape[1], IH=mask.shape[0],)
    patch = F.grid_sample(input=torch.from_numpy(mask[np.newaxis, np.newaxis, :, :]), grid=grid, align_corners=True)
    if numpy:
        patch = patch.squeeze(0).squeeze(0).numpy()
    return patch

def accumulate_votes(votes, shape):
    # Hough Voting
    xs = votes[:, 0]
    ys = votes[:, 1]
    ps = votes[:, 2]
    tl = [np.floor(ys).astype('int32'), np.floor(xs).astype('int32')]
    tr = [np.floor(ys).astype('int32'), np.ceil(xs).astype('int32')]
    bl = [np.ceil(ys).astype('int32'), np.floor(xs).astype('int32')]
    br = [np.ceil(ys).astype('int32'), np.ceil(xs).astype('int32')]
    dx = xs - tl[1]
    dy = ys - tl[0]
    tl_vals = ps * (1. - dx) * (1. - dy)
    tr_vals = ps * dx * (1. - dy)
    bl_vals = ps * dy * (1. - dx)
    br_vals = ps * dy * dx
    data = np.concatenate([tl_vals, tr_vals, bl_vals, br_vals])
    I = np.concatenate([tl[0], tr[0], bl[0], br[0]])
    J = np.concatenate([tl[1], tr[1], bl[1], br[1]])
    good_inds = np.logical_and(I >= 0, I < shape[0])
    good_inds = np.logical_and(good_inds, np.logical_and(J >= 0, J < shape[1]))
    constructed_mask = np.asarray(coo_matrix((data[good_inds], (I[good_inds], J[good_inds])), shape=shape).todense())
    return constructed_mask

def glue_back_masks(mask, bbox, image_h, image_w, seg_thresh=0.5):
    grids = create_rect_grids(bbox)   # C x H x W x 2
    # hough voting
    xs = grids[:,:,:,0].flatten()
    ys = grids[:,:,:,1].flatten()
    ps = mask.flatten()
    # crop box boundary
    g_x1 = int(np.maximum(0, np.floor(np.min(xs)) - 1))
    g_y1 = int(np.maximum(0, np.floor(np.min(ys)) - 1))
    g_x2 = int(np.minimum(image_w - 1, np.ceil(np.max(xs)) + 1))
    g_y2 = int(np.minimum(image_h - 1, np.ceil(np.max(ys)) + 1))

    g_h = g_y2 - g_y1 + 1
    g_w = g_x2 - g_x1 + 1

    if g_h > 1 and g_w > 1:
        xs = xs - g_x1
        ys = ys - g_y1
        constructed_mask = accumulate_votes(np.stack([xs, ys, ps], axis=1), shape=[g_h, g_w])
        constructed_mask = np.asarray(np.where(constructed_mask>=seg_thresh, 1., 0.), np.float32)
        return constructed_mask, g_x1, g_y1, g_w, g_h
    else:
        return None




