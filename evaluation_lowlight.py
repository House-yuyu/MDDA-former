## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
from glob import glob

import numpy as np
from natsort import natsorted
import argparse
import concurrent.futures
import utils
import lpips
import torch
from sklearn.metrics import mean_absolute_error

loss_fn = lpips.LPIPS(net='alex', spatial=True)

# mae_fn = torch.nn.L1Loss(reduction='sum')

def proc(filename):
    tar, prd = filename
    tar_img = utils.load_img(tar)
    prd_img = utils.load_img(prd)

    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    # niqe = utils.calculate_niqe(prd_img, crop_border=0)
    # MAE = torch.from_numpy(MAE)
    return PSNR, SSIM


parser = argparse.ArgumentParser(description='Deblurring scores')
parser.add_argument('--input_dir', default='../data/Low_light', type=str, help='Input images')
parser.add_argument('--result_dir', default='../Results/Low_light', type=str, help='directory of test dataset')


args = parser.parse_args()

datasets = ['LOL_v2']   # ['Cap', 'Syn']  ,  'LOL_v2'    ,   'LOL'

for dataset in datasets:

    gt_path = os.path.join(args.input_dir, dataset, 'test', 'target')
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png'))
                        + glob(os.path.join(gt_path, '*.tif'))
                        + glob(os.path.join(gt_path, '*.jpg'))
                        + glob(os.path.join(gt_path, '*.bmp')))
    assert len(gt_list) != 0, "Target files not found"

    file_path = os.path.join(args.result_dir)
    path_list = natsorted(glob(os.path.join(file_path, dataset, '*.png'))
                          + glob(os.path.join(file_path, '*.tif'))
                          + glob(os.path.join(gt_path, '*.jpg'))
                          + glob(os.path.join(gt_path, '*.bmp')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim, dist_, niqe = [], [], [], []

    for i in range(len(gt_list)):
        dummy_img0 = lpips.im2tensor(lpips.load_image(gt_list[i]))
        dummy_img1 = lpips.im2tensor(lpips.load_image(path_list[i]))

        dist = loss_fn.forward(dummy_img0, dummy_img1)
        dist_.append(dist.mean().item())
        LPIPS = sum(dist_) / len(gt_list)

    img_files = [(i, j) for i, j in zip(gt_list, path_list)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for filename, Value in zip(img_files, executor.map(proc, img_files)):
            psnr.append(Value[0])
            ssim.append(Value[1])
            # niqe.append(Value[2])

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)
    # avg_mae = sum(niqe) / len(niqe)

    print('For {:s} dataset PSNR: {:.4f} SSIM: {:.4f} LPIPS: {:.4f} \n'.format(dataset, avg_psnr, avg_ssim,
                                                                                          LPIPS))

