import torch
import torch.nn as nn
import os
import numpy as np
import network
import loss
import decoder
import post_proc
from dataset_neural import Neural
from dataset_plant import Plant
from dataset_kaggle import Kaggle
import eval_parts
import nms

def collater(data):
    gt_patches = []
    gt_bboxes2 = []

    out_data_dict = {}
    for name in data[0][0]:
        out_data_dict[name] = []

    for sample in data:
        for name in sample[0]:
            out_data_dict[name].append(torch.from_numpy(sample[0][name]))
        gt_patches.append(sample[1])
        gt_bboxes2.append(sample[2])

    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)

    return out_data_dict, gt_patches, gt_bboxes2


class CenterNet(object):
    def __init__(self, args):
        torch.manual_seed(317)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'hm': args.num_classes, 'wh': 2, 'reg': 2}
        self.model = network.CTdetSeg(heads=heads,
                                      pretrained=True,
                                      down_ratio=args.down_ratio,
                                      final_kernel=1,
                                      head_conv=256)
        self.num_classes = args.num_classes
        self.decoder = decoder.DecDecoder(K=args.K, conf_thresh=args.conf_thresh)
        self.dataset = {'kaggle':Kaggle, 'plant':Plant, 'neural': Neural}


    def save_model(self, path, epoch, model):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        data = {'epoch': epoch, 'state_dict': state_dict}
        torch.save(data, path)

    def load_model(self, model, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        state_dict = {}

        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()

        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        return model

    def set_device(self, ngpus, device):
        if ngpus > 1:
            self.model = nn.DataParallel(self.model).to(device)
        else:
            self.model = self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device=device, non_blocking=True)

    def train_network(self, args):
        weights_file = 'weights_'+args.dataset
        if not os.path.exists(weights_file):
            os.mkdir(weights_file)
        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.set_device(args.ngpus, self.device)

        criterion = loss.CtdetLoss(hm_h=args.input_h//args.down_ratio,
                                   hm_w=args.input_w//args.down_ratio,
                                   device=self.device)
        """
        criterion_hausdorff = WeightedHausdorffDistance(h=args.input_h//args.down_ratio,
                                                        w=args.input_w//args.down_ratio,
                                                        alpha=-1,
                                                        device=self.device)
        """

        print('Setting up data...')

        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=args.down_ratio)
                 for x in ['train', 'val']}

        dsets_loader = {'train': torch.utils.data.DataLoader(dsets['train'],
                                                             batch_size=args.batch_size,
                                                             shuffle=True,
                                                             num_workers=args.num_workers,
                                                             pin_memory=True,
                                                             drop_last=True,
                                                             collate_fn=collater),

                        'val':torch.utils.data.DataLoader(dsets['val'],
                                                          batch_size=1,
                                                          shuffle=False,
                                                          num_workers=1,
                                                          pin_memory=True,
                                                          collate_fn=collater)}



        # import cv2
        # for item in range(100):
        #     data_dict, gt_patches, gt_bboxes2 = dsets['train'].__getitem__(item)
        #     image   = data_dict['input']
        #     heatmap = data_dict['hm']
        #     image = np.uint8((np.transpose(image, (1,2,0))+0.5)*255)
        #     heatmap = heatmap[0,:,:]
        #     heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        #     image = img_show.map_mask_to_image(heatmap,image, (0.98,0.53,0.))
        #     cv2.imshow('img', image)
        #     k = cv2.waitKey(0)&0xFF
        #     if k==ord('q'):
        #         cv2.destroyAllWindows()
        #         exit()

        print('Starting training...')
        train_loss = []
        val_loss = []
        ap_05 = []
        ap_07 = []
        iou_05 = []
        iou_07 = []
        for epoch in range(1, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            train_loss.append(epoch_loss)

            epoch_loss = self.run_epoch(phase='val',
                                        data_loader=dsets_loader['val'],
                                        criterion=criterion)
            val_loss.append(epoch_loss)

            np.savetxt('train_loss.txt', train_loss, fmt='%.6f')
            np.savetxt('val_loss.txt', val_loss, fmt='%.6f')

            if epoch % 10 == 0 or epoch ==1:
                self.save_model(os.path.join(weights_file, 'model_{}.pth'.format(epoch)), epoch, self.model)
                dsets = dataset_module(data_dir=args.data_dir,
                                       phase='test',
                                       input_h=args.input_h,
                                       input_w=args.input_w,
                                       down_ratio=args.down_ratio)
                ap_05_out, iou_05_out = self.seg_eval(dsets=dsets, ov_thresh=0.5, down_ratio=args.down_ratio)
                ap_07_out, iou_07_out = self.seg_eval(dsets=dsets, ov_thresh=0.7, down_ratio=args.down_ratio)
                ap_05.append(ap_05_out)
                ap_07.append(ap_07_out)
                iou_05.append(iou_05_out)
                iou_07.append(iou_07_out)
                np.savetxt('ap_05.txt', ap_05, fmt='%.6f')
                np.savetxt('ap_07.txt', ap_07, fmt='%.6f')
                np.savetxt('iou_05.txt', iou_05, fmt='%.6f')
                np.savetxt('iou_07.txt', iou_07, fmt='%.6f')

            self.save_model(os.path.join(weights_file, 'model_last.pth'), epoch, self.model)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        for data_dict, gt_patches, gt_bboxes2 in data_loader:
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    pr_decs, pr_patches, gt_patches = self.model(data_dict['input'], gt_bboxes2, gt_patches)
                    loss = criterion(pr_decs, pr_patches, data_dict, gt_patches)
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs, pr_patches, gt_patches = self.model(data_dict['input'], gt_bboxes2, gt_patches)
                    loss = criterion(pr_decs, pr_patches, data_dict, gt_patches)

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        print('{} loss: {}'.format(phase, epoch_loss))
        return epoch_loss


    def seg_eval(self, dsets, ov_thresh, down_ratio):
        self.model.eval()
        all_tp = []
        all_fp = []
        all_scores = []
        temp_overlaps = []
        npos = 0
        for index in range(len(dsets)):
            data_dict = dsets.__getitem__(index)
            images = data_dict['images'].to(self.device)
            img_id = data_dict['img_id']

            with torch.no_grad():
                z, feat = self.model.forward_dec(images)
                hm = z['hm']
                wh = z['wh']
                reg = z['reg']

            torch.cuda.synchronize()
            pr_bboxes2 = self.decoder.ctdet_decode(hm, wh, reg=reg)
            if np.any(pr_bboxes2):
                with torch.no_grad():
                    pr_patches, pr_bboxes0 = self.model.forward_seg(feat, pr_bboxes2, test=True)
                pr_masks, pr_bboxes = post_proc.affine_mask_process(dsets=dsets,
                                                                    img_id=img_id,
                                                                    pr_masks=pr_patches,
                                                                    pr_bboxes0=pr_bboxes0,
                                                                    seg_thresh=0.5)
                keep_index = nms.non_maximum_suppression_numpy_masks(pr_masks, pr_bboxes, nms_thresh=0.5)
                if keep_index is not None:
                    if len(keep_index)>2:
                        pr_masks = eval_parts.sorted_BB_mask(pr_masks, keep_index)
                        pr_bboxes = np.take(pr_bboxes, keep_index, axis=0)
            else:
                pr_masks = []
                pr_bboxes = []

            fp, tp, all_scores, npos, temp_overlaps = eval_parts.seg_evaluation(BB_mask=pr_masks,
                                                                                BB_bboxes=pr_bboxes,
                                                                                dsets=dsets,
                                                                                all_scores=all_scores,
                                                                                img_id=img_id,
                                                                                npos=npos,
                                                                                temp_overlaps=temp_overlaps,
                                                                                ov_thresh=ov_thresh)
            all_fp.extend(fp)
            all_tp.extend(tp)
        # step5: compute precision recall
        all_fp = np.asarray(all_fp)
        all_tp = np.asarray(all_tp)
        all_scores = np.asarray(all_scores)
        sorted_ind = np.argsort(-all_scores)
        all_fp = all_fp[sorted_ind]
        all_tp = all_tp[sorted_ind]
        all_fp = np.cumsum(all_fp)
        all_tp = np.cumsum(all_tp)
        rec = all_tp / float(npos)
        prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)

        ap = eval_parts.voc_ap(rec, prec, use_07_metric=True)
        print("ap@{} is {}, iou is {}".format(ov_thresh, ap, np.mean(temp_overlaps)))
        return ap, np.mean(temp_overlaps)

