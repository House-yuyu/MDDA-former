"""
## Reference from: Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""

import numpy as np
import os
import argparse

from tqdm import tqdm
import torch
import utils

from model.SIDD_test1029 import RCUNet
from skimage import img_as_ubyte
import scipy.io as sio
import cv2
from collections import OrderedDict


os.environ["CUDA_VISIBLE_DEVICES"] = "5"
parser = argparse.ArgumentParser(description='Image Denoising using RCUNet')

parser.add_argument('--input_dir', default="../data/SIDD_test", type=str,
                    help='Directory of validation images')
parser.add_argument('--result_dir', default='../Results/SIDD/', type=str, help='Directory for results')
parser.add_argument('--weights', default="../checkpoints/SIDD/models/model_bestPSNR.pth", type=str,
                    help='Path to weights')
parser.add_argument('--save_images', default=True, help='Save denoised images in result directory')
args = parser.parse_args()


result_dir = os.path.join(args.result_dir, 'mat')
utils.mkdir(result_dir)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cuda:0')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if args.save_images:
    result_dir_img = os.path.join(args.result_dir, 'png')
    utils.mkdir(result_dir_img)

model_restoration = RCUNet()

print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
load_checkpoint(model_restoration, args.weights)
model_restoration.eval()

# Process data
filepath = os.path.join(args.input_dir, 'ValidationNoisyBlocksSrgb.mat')
img = sio.loadmat(filepath)
Inoisy = np.float32(np.array(img['ValidationNoisyBlocksSrgb']))
Inoisy /= 255.
restored = np.zeros_like(Inoisy)

import time
start = time.time()
with torch.no_grad():
    for i in tqdm(range(40)):
        for k in range(32):
            noisy_patch = torch.from_numpy(Inoisy[i, k, :, :, :]).unsqueeze(0).permute(0, 3, 1, 2).cuda()
            restored_patch = model_restoration(noisy_patch)
            restored_patch = torch.clamp(restored_patch, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
            restored[i, k, :, :, :] = restored_patch

            if args.save_images:
                save_file = os.path.join(result_dir_img, '%04d_%02d.png' % (i + 1, k + 1))
                save_img(save_file, img_as_ubyte(restored_patch))

print('Process time each patch:', (time.time() - start)/1280)
sio.savemat(os.path.join(result_dir, 'Idenoised.mat'), {"Idenoised": restored, })
