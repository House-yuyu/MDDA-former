## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
from glob import glob
from natsort import natsorted
import argparse
import concurrent.futures
import utils

def proc(filename):
    tar, prd = filename
    tar_img = utils.load_img(tar)
    prd_img = utils.load_img(prd)

    PSNR = utils.calculate_rgb_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR, SSIM


parser = argparse.ArgumentParser(description='Deblurring scores')
parser.add_argument('--input_dir', default='../data/Deblur', type=str, help='Input images')
parser.add_argument('--result_dir', default='../Results/Deblur', type=str, help='directory of test dataset')


args = parser.parse_args()

datasets = ['HIDE']  # 'GoPro', 'HIDE'

for dataset in datasets:

    gt_path = os.path.join(args.input_dir, 'test', dataset,  'target')
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png'))
                        + glob(os.path.join(gt_path, '*.tif'))
                        + glob(os.path.join(gt_path, '*.jpg'))
                        + glob(os.path.join(gt_path, '*.bmp')))
    assert len(gt_list) != 0, "Target files not found"

    file_path = os.path.join(args.result_dir, dataset)
    path_list = natsorted(glob(os.path.join(file_path, '*.png'))
                          + glob(os.path.join(file_path, '*.tif'))
                          + glob(os.path.join(gt_path, '*.jpg'))
                          + glob(os.path.join(gt_path, '*.bmp')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim = [], []

    img_files = [(i, j) for i, j in zip(gt_list, path_list)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])

    avg_psnr = sum(psnr) / len(psnr)
    avg_ssim = sum(ssim) / len(ssim)

    print('For {:s} dataset PSNR: {:.4f} SSIM: {:.4f}\n'.format(dataset, avg_psnr, avg_ssim))
