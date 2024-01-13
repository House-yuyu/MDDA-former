
import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from tqdm import tqdm
import concurrent.futures
import utils

def proc(filename):
    tar, prd = filename
    tar_img = utils.load_img(tar)
    prd_img = utils.load_img(prd)
        
    PSNR = utils.calculate_psnr(tar_img, prd_img)  # rgb_psnr is need for color
    # SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR

parser = argparse.ArgumentParser(description='Gasussian Color Denoising using Restormer')
parser.add_argument('--input_dir', default='../data/Denoise/test/Gaussian', type=str, help='Input images')
parser.add_argument('--result_dir', default='../Results/Denoise/Gaussian', type=str, help='directory of test dataset')
parser.add_argument('--sigmas', default='15,25,50', type=str, help='Sigma values')

args = parser.parse_args()

sigmas = np.int_(args.sigmas.split(','))

datasets = ['CBSD68', 'Kodak', 'McMaster', 'Urban100']

for dataset in datasets:

    gt_path = os.path.join(args.input_dir, dataset)
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
    assert len(gt_list) != 0, "Target files not found"

    for sigma_test in sigmas:
        file_path = os.path.join(args.result_dir, dataset, str(sigma_test))
        path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
        assert len(path_list) != 0, "Predicted files not found"

        psnr, ssim = [], []

        img_files = [(i, j) for i, j in zip(gt_list, path_list)]
        with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
            for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
                psnr.append(PSNR_SSIM)
                # ssim.append(PSNR_SSIM[1])

        avg_psnr = sum(psnr)/len(psnr)
        # avg_ssim = sum(ssim)/len(ssim)

        print('For {:s} dataset Noise Level {:d} PSNR: {:.4f}\n'.format(dataset, sigma_test, avg_psnr))
        # print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))