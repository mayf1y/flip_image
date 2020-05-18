from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import cv2
import torch
import json
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
from utils.debugger import Debugger
import math
import random

class PointDataset(data.Dataset):
    num_classes = 12
    default_resolution = [512, 512]
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split):
        super(PointDataset, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'point')
        self.img_dir = os.path.join(self.data_dir, 'images')
        txt_dir = os.path.join(self.data_dir, '{}.txt'.format(split))
        fh = open(txt_dir, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip('\n')
            imgs.append(line)
        self.images = imgs
        self.annot_path = os.path.join(self.data_dir, 'annotations')
        self.max_objs = 64
        self.split = split
        self.opt = opt
        self.class_name = [
            'comma_0', 'comma_1', 'comma_2', 'comma_3',
            '3_0', '3_1', '3_2', '3_3',
            '4_0', '4_1', '4_2', '4_3',
            '7_0', '7_1', '7_2', '7_3',
            'ren_0', 'ren_1', 'ren_2', 'ren_3',
        ]
        self._valid_ids = np.arange(1, 4, dtype=np.int32)
        self.cat_ids = {v: i for i, v in enumerate(self.class_name)}
        self._data_rng = np.random.RandomState(123)
        self.num_samples = len(self.images)
        self.down_sample = 4
        self.resize = int(opt.resize)

    def __getitem__(self, index):
        radius = 20//self.resize
        file = self.images[index]
        img_path = os.path.join(self.img_dir, '{}.jpg'.format(file))
        ann_path = os.path.join(self.annot_path, '{}.json'.format(file[:-2]))
        img = cv2.imread(img_path)
        cls = int(file[-1])
        h, w, _ = img.shape
        vertical = False
        if cls == 0:
            if h > w:
                img = img[:2304, :-1, :]
                img = cv2.copyMakeBorder(img, 0, 0, 326, 326, cv2.BORDER_CONSTANT, value=0)
            else:
                vertical = True
                img = img[:-1, :2304, :]
                img = cv2.copyMakeBorder(img, 326, 326, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif cls == 1:
            if h < w:
                img = img[1:, :2304, :]
                img = cv2.copyMakeBorder(img, 326, 326, 0, 0, cv2.BORDER_CONSTANT, value=0)
            else:
                vertical = True
                img = img[35:, :-1, :]
                img = cv2.copyMakeBorder(img, 0, 0, 326, 326, cv2.BORDER_CONSTANT, value=0)
        elif cls == 2:
            if h > w:
                img = img[35:, 1:, :]
                img = cv2.copyMakeBorder(img, 0, 0, 326, 326, cv2.BORDER_CONSTANT, value=0)
            else:
                vertical = True
                img = img[1:, 35:, :]
                img = cv2.copyMakeBorder(img, 326, 326, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            if h < w:
                img = img[:-1, 35:, :]
                img = cv2.copyMakeBorder(img, 326, 326, 0, 0, cv2.BORDER_CONSTANT, value=0)
            else:
                vertical = True
                img = img[:2304, 1:, :]
                img = cv2.copyMakeBorder(img, 0, 0, 326, 326, cv2.BORDER_CONSTANT, value=0)
        h, w, _ = img.shape
        img = cv2.resize(img, (w//self.resize, h//self.resize), interpolation=cv2.INTER_AREA)
        input_h, input_w, _ = img.shape
        # img = cv2.resize(img, (1024, 1152), interpolation = cv2.INTER_AREA)

        c = np.array([input_w / 2., input_h / 2.], dtype=np.float32)
        s = max(input_h, input_w) * 1.0
        if self.split == 'train':
            sf = self.opt.scale
            cf = self.opt.shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        img = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (img.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.down_sample
        output_w = input_w // self.down_sample
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])
        hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
        draw_gaussian = draw_umich_gaussian

        f = open(ann_path, 'r')
        load_dict = json.load(f)
        for k in range(len(load_dict['shapes'])):
            ct = load_dict['shapes'][k]['points'][0]
            if vertical:
                ct[1] += 326
            else:
                ct[0] += 326
            if cls == 1:
                ct[0], ct[1] = ct[1], w - ct[0]
            elif cls == 2:
                ct[0], ct[1] = w - ct[0], h - ct[1]
            elif cls == 3:
                ct[0], ct[1] = h - ct[1], ct[0]
            ct = [ct[0]/self.resize, ct[1]/self.resize]
            ct = affine_transform(ct, trans_output)
            ct = np.array(ct, dtype=np.float32)
            ct_int = ct.astype(np.int32)
            if load_dict['shapes'][k]['label'] == 'point':
                draw_gaussian(hm[cls], ct_int, int(radius*(input_h/s)))
            elif load_dict['shapes'][k]['label'] == 'ren':
                draw_gaussian(hm[8+cls], ct_int, int(radius * (input_h/s) * 0.7))
            elif load_dict['shapes'][k]['label'] == 'lren':
                draw_gaussian(hm[8+cls], ct_int, int(radius * (input_h/s) * 0.85))
            elif load_dict['shapes'][k]['label'] == 'bren':
                draw_gaussian(hm[8+cls], ct_int, int(radius * (input_h/s)))
            else:
                draw_gaussian(hm[4 + cls], ct_int, int(radius * (input_h / s) * 0.65))


        ret = {'input': inp, 'hm': hm, 'cls': cls}
        if self.opt.draw:
            debugger = Debugger(dataset='point',
                ipynb=False, theme='white')
            img = np.clip(img, 0, 255).astype(np.uint8)
            gt = debugger.gen_colormap(ret['hm'])
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.show_all_imgs(pause=True)
        return ret

    def __len__(self):
        return len(self.images)











