# This code is referenced from MTI-Net
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import imageio
import numpy as np
import json
import torch
import scipy.io as sio
from utils.utils import get_output
import ipdb

nyuv2_palette = [
    0, 0, 0,
    255, 20, 23,
    255, 102, 17,
    255, 136, 68,
    255, 238, 85,
    254, 254, 56,
    255, 255, 153,
    170, 204, 34,
    187, 221, 119,
    200, 207, 130,
    146, 167, 126,
    85, 153, 238,
    0, 136, 204,
    34, 102, 136,
    23, 82, 121,
    85, 119, 119,
    221, 187, 51,
    211, 167, 109,
    169, 131, 75,
    118, 118, 118,
    81, 87, 74,
    68, 124, 105,
    116, 196, 147,
    142, 140, 109,
    228, 191, 128,
    233, 215, 142,
    226, 151, 93,
    241, 150, 112,
    225, 101, 82,
    201, 74, 83,
    190, 81, 104,
    163, 73, 116,
    153, 55, 103,
    101, 56, 125,
    78, 36, 114,
    145, 99, 182,
    226, 121, 163,
    224, 89, 139,
    124, 159, 176,
    86, 152, 196,
    154, 191, 136
]

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])#

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype = np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ ( np.uint8(str_id[-1]) << (7-j))
            g = g ^ ( np.uint8(str_id[-2]) << (7-j))
            b = b ^ ( np.uint8(str_id[-3]) << (7-j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap

def vis_semseg(_semseg):
    #if False: # NYUD
    if True:
        #Note: We ignore the background class as other related works. # nyud
        #_semseg += 1 #ignore
        new_cmap = labelcolormap(41)
        for i in range(new_cmap.shape[0]):
            new_cmap[i] = nyuv2_palette[i*3:(i+1)*3]
    else:
        # for pascal context, don't ignore background
        new_cmap = labelcolormap(21)
    #ipdb.set_trace()
    _semseg = new_cmap[_semseg]  # skip the backgournd class
    return _semseg

def vis_parts(_semseg):
    if False: # NYUD
        #Note: We ignore the background class as other related works. # nyud
        _semseg += 1 
        new_cmap = labelcolormap(41)
    else:
        # for pascal context, don't ignore background
        new_cmap = labelcolormap(7)

    _semseg = new_cmap[_semseg]  # skip the backgournd class
    return _semseg


class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, p, tasks):
        self.database = p['train_db_name']
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(p, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self, verbose=True):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score(verbose)

        return eval_dict

def get_single_task_meter(p, database, task):
    """ Retrieve a meter to measure the single-task performance """

    # ignore index based on transforms.AddIgnoreRegions
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(database, ignore_idx=p.ignore_index)

    elif task == 'human_parts':
        from evaluation.eval_human_parts import HumanPartsMeter
        return HumanPartsMeter(database, ignore_idx=p.ignore_index)

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter(ignore_index=p.ignore_index) 

    elif task == 'sal':
        from evaluation.eval_sal import  SaliencyMeter
        return SaliencyMeter(ignore_index=p.ignore_index, threshold_step=0.05, beta_squared=0.3)

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        return DepthMeter(ignore_index=p.ignore_index) 

    elif task == 'edge': # just for reference
        from evaluation.eval_edge import EdgeMeter
        return EdgeMeter(pos_weight=p['edge_w'], ignore_index=p.ignore_index)

    else:
        raise NotImplementedError

@torch.no_grad()
def save_model_pred_for_one_task(p, sample, output, save_dirs, task=None, epoch=None):
    """ Save model predictions for one task"""

    inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
    output_task = get_output(output[task], task)

    for jj in range(int(inputs.size()[0])):
        if len(sample[task][jj].unique()) == 1 and sample[task][jj].unique() == p.ignore_index:
            continue
        fname = meta['img_name'][jj]

        im_height = meta['img_size'][jj][0]
        im_width = meta['img_size'][jj][1]
        pred = output_task[jj] # (H, W) or (H, W, C)
        # if we used padding on the input, we crop the prediction accordingly
        if (im_height, im_width) != pred.shape[:2]:
            delta_height = max(pred.shape[0] - im_height, 0)
            delta_width = max(pred.shape[1] - im_width, 0)
            if delta_height > 0 or delta_width > 0:
                height_begin = torch.div(delta_height, 2, rounding_mode="trunc")
                height_location = [height_begin, height_begin + im_height]
                width_begin =torch.div(delta_width, 2, rounding_mode="trunc")
                width_location = [width_begin, width_begin + im_width]
                pred = pred[height_location[0]:height_location[1],
                            width_location[0]:width_location[1]]
        assert pred.shape[:2] == (im_height, im_width)

        result = pred.cpu().numpy()
        if task == 'depth':
            result = result-result.min()
            result = result/result.max()
            result = result*255.
            cv2.imwrite(os.path.join(save_dirs[task], fname + '.png'),cv2.applyColorMap(result.astype(np.uint8),cv2.COLORMAP_JET))   
        elif task=='sal':
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'),result.astype(np.uint8))
        elif task=='normals':
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'),result.astype(np.uint8))
        elif task=='semseg':
            result = vis_semseg(result)
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'),result.astype(np.uint8))
        elif task=='human_parts':
            result = vis_parts(result)
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'),result.astype(np.uint8))     
        else:
            imageio.imwrite(os.path.join(save_dirs[task], fname + '.png'), result.astype(np.uint8))
