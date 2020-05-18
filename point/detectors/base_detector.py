from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import random

class BaseDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
        self.max_per_image = 100
        self.num_classes = opt.num_classes
        self.opt = opt
        self.pause = True
        self.resize = int(opt.resize)
        self.results = {}

    def pre_process(self, image):
        input_h, input_w, _ = image.shape
        image = ((image / 255. - self.mean) / self.std).astype(np.float32)
        image = image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)

        c = np.array([input_w // 2, input_h // 2], dtype=np.float32)
        s = np.array([input_w, input_h], dtype=np.float32)
        image = torch.from_numpy(image)
        meta = {'c': c, 's': s,
                'out_height': input_h // self.opt.down_ratio,
                'out_width': input_w // self.opt.down_ratio}
        return image, meta

    def process(self, images, return_time=False):
        raise NotImplementedError

    def post_process(self, dets, meta, scale=1):
        raise NotImplementedError

    def debug(self, debugger, images, dets, output, scale=1):
        raise NotImplementedError

    def show_results(self, debugger, image, results, img_id):
        raise NotImplementedError

    def run(self, image_or_path_or_tensor, meta=None):
        load_time, pre_time, net_time = 0, 0, 0
        tot_time = 0
        start_time = time.time()
        # 预处理
        image = cv2.imread(image_or_path_or_tensor)
        loaded_time = time.time()
        load_time += (loaded_time - start_time)
        scale_start_time = time.time()

        h, w, _ = image.shape
        # image = cv2.resize(image, (w // self.resize, h // self.resize), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (w // self.resize, h // self.resize), interpolation=cv2.INTER_AREA)
        # pre_process图像预处理，图像的尺寸resize，相应的batch调整，得到输入图像。
        if h > w:
            image = cv2.copyMakeBorder(image, 0, 640 - h//self.resize, 0, 512 - w//self.resize, cv2.BORDER_CONSTANT, value=0)
        else:
            image = cv2.copyMakeBorder(image, 0, 512 - h//self.resize, 0, 640 - w//self.resize, cv2.BORDER_CONSTANT, value=0)
        # image = np.rot90(image, cls, (0, 1))
        images, meta = self.pre_process(image)

        images = images.to(self.opt.device)
        torch.cuda.synchronize()
        pre_process_time = time.time()
        pre_time += pre_process_time - scale_start_time

        # process 得到输出output，并decode成预测框
        hm, forward_time = self.process(images, return_time=True)
        hm = hm.squeeze(0)
        hm[:4, :, :][hm[:4, :, :] < 0.04] = 0
        hm[4:8, :, :][hm[4:8, :, :] < 0.1] = 0
        hm[8:, :, :][hm[8:, :, :] < 0.1] = 0
        result = torch.sum(hm, dim=1)
        result = torch.sum(result, dim=1)
        result = result.cpu().numpy()
        list_result = result.tolist()
        result = [0 for _ in range(4)]
        for i in range(self.opt.num_classes):
            result[i % 4] += list_result[i]

        max_result = max(result)
        result = result.index(max_result)
        self.results[image_or_path_or_tensor[-20:]] = result
        torch.cuda.synchronize()
        net_time += forward_time - pre_process_time
        tot_time = time.time() - start_time
        if self.opt.draw:
            debugger = Debugger(dataset='point', ipynb=False, theme='white')
            img = np.clip(image, 0, 255).astype(np.uint8)
            hm = hm.cpu().numpy()
            gt_comma = debugger.gen_colormap(hm[:4, :, :])
            gt_347 = debugger.gen_colormap(hm[4:8, :, :])
            gt_ren = debugger.gen_colormap(hm[8:, :, :])
            debugger.add_blend_img(img, gt_comma, 'comma')
            debugger.add_blend_img(img, gt_347, '347')
            debugger.add_blend_img(img, gt_ren, 'ren')
            if not result:
                debugger.save_all_imgs(prefix=image_or_path_or_tensor[44:-4])
            if self.opt.debug == 1:
                debugger.show_all_imgs(pause=True)

        return {'result': result, 'tot': tot_time, 'load': load_time,
                'pre': pre_time, 'net': net_time}