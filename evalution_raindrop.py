import cv2
import os
import argparse

import numpy as np
from natsort import natsorted
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return peak_signal_noise_ratio(im1_y, im2_y)

def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return structural_similarity(im1_y, im2_y)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='../Results/Deraining/test_a', type=str)
    parser.add_argument("--gt_dir", default='../data/Derain/Raindrop/test_a/target', type=str)
    args = parser.parse_args()
    return args

def align_to_four(img):
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col]
    return img


if __name__ == '__main__':
    args = get_args()

    input_list = natsorted(os.listdir(args.input_dir))
    gt_list = natsorted(os.listdir(args.gt_dir))
    num = len(input_list)
    cumulative_psnr = 0
    cumulative_ssim = 0
    for i in range(num):
        print('Processing image: %s'%(input_list[i]))
        img = cv2.imread(os.path.join(args.input_dir, input_list[i]))
        gt = cv2.imread(os.path.join(args.gt_dir, gt_list[i]))
        img = align_to_four(img)
        gt = align_to_four(gt)
        result = img
        cur_psnr = calc_psnr(result, gt)
        cur_ssim = calc_ssim(result, gt)
        print('PSNR is %.4f and SSIM is %.4f'%(cur_psnr, cur_ssim))
        cumulative_psnr += cur_psnr
        cumulative_ssim += cur_ssim
    print('In testing dataset, PSNR is %.4f and SSIM is %.4f'%(cumulative_psnr/num, cumulative_ssim/num))
