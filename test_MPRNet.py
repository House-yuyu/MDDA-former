"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import utils

import utils
from collections import OrderedDict
from natsort import natsorted
from glob import glob

from model.MDDA import RCUNet
from skimage import img_as_ubyte

from datasets.data_RGB import get_test_data
from CnnModel.model.MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='../data/Derain/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='../pretrained_models/model_best.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='5', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = MPRNet()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ",args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

datasets = ['200']
# datasets = ['Rain100L']


for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, dataset, 'input')  # 'test', dataset
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_)) / 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            factor = 8
            # Padding in case images are not multiples of 8
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            restored = model_restoration(input_)

            restored = restored[0]

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]
            restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
                           img_as_ubyte(restored))
