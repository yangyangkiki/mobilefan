from __future__ import print_function, absolute_import

import os
import numpy as np
import json
import random
import math
import matplotlib.pyplot as plt
# from skimage import io
import cv2

import torch
import torch.utils.data as data

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import pandas as pd
import hdf5storage

class Cofw_no_vector(data.Dataset):
    def __init__(self, mat_file, inp_res=256, out_res=64, train=True, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian'):

        # self.landmarks_frame = pd.read_csv(csv_file)

        self.mat = hdf5storage.loadmat(mat_file)

        if train:
            self.images = self.mat['IsTr']
            self.pts = self.mat['phisTr']
        else:
            self.images = self.mat['IsT']
            self.pts = self.mat['phisT']

        # # debug
        # self.images = self.images[1:2]
        # self.pts = self.pts[1:2]

        print('total %d images' % len(self.images))

        self.is_train = train           # training set or test set
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.mean = [0.485, 0.456, 0.406]  # RGB
        self.std = (0.229, 0.224, 0.225)  # RGB

    def get_img(self,index):
        img = self.images[index][0]
        return img

    def __getitem__(self, index):
        # index = 2
        sf = self.scale_factor
        rf = self.rot_factor

        img = self.images[index][0]  # RGB

        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
            img = np.repeat(img, 3, axis=2)

        pts = self.pts[index][0:58].reshape(2,-1).transpose()

        xmin = np.min(pts[:, 0])
        xmax = np.max(pts[:, 0])
        ymin = np.min(pts[:, 1])
        ymax = np.max(pts[:, 1])

        # center
        center_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        center_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0

        c_w = (math.floor(xmin) + math.ceil(xmax)) / 2.0
        c_h = (math.floor(ymin) + math.ceil(ymax)) / 2.0
        # scale
        s = max(math.ceil(xmax) - math.floor(xmin), math.ceil(ymax) - math.floor(ymin)) / 200.0
        c = torch.Tensor([c_w, c_h])

        pts = torch.Tensor(pts.tolist())

        # # Adjust center/scale slightly to avoid cropping limbs
        # ------------------------
        # dis_diagonal = np.linalg.norm(np.array([xmin, ymin]) - np.array([xmax, ymax]))
        # s = dis_diagonal / 200.0
        # -------------------------
        s = s * 1.25  # no problem
        # -------------------------

        # For single-person pose estimation with a centered/scaled figure
        nparts = pts.size(0)
        img = load_image(img)  # (0-1)

        # # show
        # plt.imshow(img.numpy().transpose((1,2,0)))
        # plt.scatter(np.array(pts)[:, 0], np.array(pts)[:, 1], s=10, marker='.', c='r')
        # plt.show()

        r = 0
        if self.is_train:

            s = s * (random.uniform(1-sf, 1+sf))
            r = random.uniform(-rf, rf) if random.random() <= 0.6 else 0

            # Flip
            if random.random() <= 0.5:
                img = torch.from_numpy(fliplr(img.numpy())).float()
                pts = shufflelr(pts, width=img.size(2), dataset='cofw')
                c[0] = img.size(2) - c[0]
                center_w = img.size(2) - center_w

            # # Color
            # img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            # img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        # 3*256*256
        inp = crop(img, c, s, [self.inp_res, self.inp_res], rot=r)  # H*W*C
        # # # show
        # plt.imshow(inp)
        # plt.show()

        # tpts = pts.clone()
        # for ii in range(nparts):
        #     tpts[ii, 0:2] = to_torch(transform(tpts[ii, 0:2] + 1 , [c_w, c_h], s, [256, 256], rot=r))
        #
        # plt.imshow(inp)
        # plt.scatter(tpts.numpy()[:, 0], pts.numpy()[:, 1], s=10, marker='.', c='r')
        # plt.show()

        # plt.imshow(cv2.resize(inp,(64,64)))
        # plt.show()

        # inp = im_to_torch(inp)  # C*H*W (0-1)
        # norm
        inp = inp / 255.0
        inp -= self.mean
        inp /= self.std
        inp = inp.transpose(2, 0, 1)
        inp = torch.from_numpy(inp).float()

        # Generate heatmap ground truth
        tpts = pts.clone()
        target = torch.zeros(nparts, self.out_res, self.out_res)  # 68*64*64
        for i in range(nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2]+1, c, s, [self.out_res, self.out_res], rot=r))
                target[i] = draw_labelmap(target[i], tpts[i]-1, self.sigma, type=self.label_type)

        # for ii in range(2):
        #     plt.imshow(target.numpy()[ii])
        #     plt.show()

        # Meta info
        meta = {'index': index, 'center' : torch.Tensor(c), 'scale' : s, #torch.Tensor((center_w, center_h))
        'pts' : pts, 'tpts' : tpts}

        return inp, target, meta  # 3*256*256  68*64*64

    def __len__(self):
        return len(self.images)