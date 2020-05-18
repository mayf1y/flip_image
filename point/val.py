from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from datasets.point_data import PointDataset
from detectors.detector_factory import detector_factory


def val(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    Dataset = PointDataset
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
    Logger(opt)
    Detector = detector_factory[opt.task]

    split = 'test'
    dataset = Dataset(opt, split)
    detector = Detector(opt)

    results = {}
    num_iters = len(dataset)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    count = 0
    for ind in range(num_iters):
        img_id = dataset.images[ind]
        image_path = os.path.join(dataset.img_dir, img_id+'.jpg')
        ret = detector.run(image_path)

        if ret['result']:
            count += 1
        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                   ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].val)
        print(Bar.suffix)
        bar.next()
    with open('results.json', 'w') as fp:
        fp.write(json.dumps(detector.results, indent=4))
    bar.finish()

def generate_txt():
    fileList_images = (os.listdir('../data/point/images'))
    test_file = open('../data/point/test.txt', 'w')

    for fileName in fileList_images:
        fileName_ = fileName[:-4]
        test_file.write(fileName_ + '\n')
    test_file.close()

if __name__ == '__main__':
    generate_txt()
    opt = opts().parse()
    val(opt)



