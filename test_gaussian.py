## Modified from Restormer. https://arxiv.org/abs/2111.09881, [CVPR 2022 Oral] ##


import os.path
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import img_as_ubyte
from collections import OrderedDict
from natsort import natsorted
from glob import glob
from tqdm import tqdm
import utils

from model.MDDA import RCUNet


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

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

def main():

    # ----------------------------------------
    # Preparation
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default='../checkpoints/Gaussian/models/model_bestPSNR_gaussian.pth', help='path of log files')
    parser.add_argument('--input_dir', default='../data/Denoise/test/Gaussian', type=str, help='Input images')
    parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')
    parser.add_argument('--result_dir', default='../Results/Denoise/Gaussian', type=str, help='directory of test dataset')

    args = parser.parse_args()

    datasets = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']

    sigmas = np.int_(args.sigmas.split(','))

    for sigma_test in sigmas:
        print("Compute results for noise level", sigma_test)
        model = RCUNet()
        model.cuda()
        load_checkpoint(model, args.weights)
        model.eval()

        for dataset in datasets:
            inp_dir = os.path.join(args.input_dir, dataset)
            files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.tif')))
            print(len(files))
            result_dir_tmp = os.path.join(args.result_dir, dataset, str(sigma_test))
            os.makedirs(result_dir_tmp, exist_ok=True)

            if len(files) == 0:
                raise Exception(f"No files found at {inp_dir}")

            factor = 8
            with torch.no_grad():  # jide
                for file_ in tqdm(files):
                    torch.cuda.ipc_collect()
                    torch.cuda.empty_cache()
                    img = np.float32(utils.load_img(file_)) / 255.

                    np.random.seed(seed=1111)  # for reproducibility
                    img += np.random.normal(0, sigma_test / 255., img.shape)
                    img = np.clip(img, 0, 1)
                    img = torch.from_numpy(img).permute(2, 0, 1)

                    input_ = img.unsqueeze(0).cuda()

                    # Padding in case images are not multiples of 8
                    h, w = input_.shape[2], input_.shape[3]
                    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                    padh = H - h if h % factor != 0 else 0
                    padw = W - w if w % factor != 0 else 0
                    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                    restored = model(input_)

                    # Unpad images to original dimensions
                    restored = restored[:, :, :h, :w]

                    restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                    save_file = os.path.join(result_dir_tmp, os.path.split(file_)[-1])
                    utils.save_img(save_file, img_as_ubyte(restored))


main()
