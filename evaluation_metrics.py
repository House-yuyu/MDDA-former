'''
calculate the PSNR and SSIM.
same as MATLAB's results
'''
import math
import numpy as np
import cv2
import glob
import os
import argparse
from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(description="metrics test")
    parser.add_argument('--input_dir', default='../Datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='../Results/', type=str, help='Directory for results')

    parser.add_argument('--encoding_format', default='H265', type=str, help='H264, H265')
    parser.add_argument('--dataset', default='newspaper', type=str, help='[kendo, lovebird, newspaper, hall, champagne, pantomime]')
    parser.add_argument('--noisy_level', default='1,2,3,4,5', type=str, help='[1,2,3,4,5]')
    opt = parser.parse_args()


    crop_border = 0  # same with scale
    test_Y = True  # True: test Y channel only; False: test RGB channels

    PSNR_all = []
    SSIM_all = []

    sigmas = np.int_(opt.noisy_level.split(','))

    datasets = ['kendo', 'lovebird', 'newspaper', 'hall', 'pantomime']

    if test_Y:
        print('Testing Y channel.')
    else:
        print('Testing RGB channels.')
    for dataset in datasets:
        for sigma_test in sigmas:
            print("Compute results for noise level", sigma_test)
            test_files_gt = natsorted(
                glob.glob(os.path.join(opt.input_dir, opt.encoding_format, dataset, 'gt', '*.bmp')) +
                glob.glob(os.path.join(opt.input_dir, opt.encoding_format, dataset, 'gt',
                                       '*.png')))  # CBM3D use sorted. love--> bmp注意换PNG,
            test_files = natsorted(
                glob.glob(os.path.join(opt.result_dir, opt.encoding_format, dataset, str(sigma_test), '*.png')))

            for idx in range(len(test_files)):
                im_GT = cv2.imread(test_files_gt[idx]) / 255.
                im_Gen = cv2.imread(test_files[idx]) / 255.

                if test_Y and im_GT.shape[2] == 3:  # evaluate on Y channel in YCbCr color space
                    im_GT_in = bgr2ycbcr(im_GT)
                    im_Gen_in = bgr2ycbcr(im_Gen)
                else:
                    im_GT_in = im_GT
                    im_Gen_in = im_Gen

                # crop borders
                if crop_border == 0:
                    cropped_GT = im_GT_in
                    cropped_Gen = im_Gen_in
                else:
                    if im_GT_in.ndim == 3:
                        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border, :]
                        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border, :]
                    elif im_GT_in.ndim == 2:
                        cropped_GT = im_GT_in[crop_border:-crop_border, crop_border:-crop_border]
                        cropped_Gen = im_Gen_in[crop_border:-crop_border, crop_border:-crop_border]
                    else:
                        raise ValueError('Wrong image dimension: {}. Should be 2 or 3.'.format(im_GT_in.ndim))

                # 不同通道数（Y通道和RGB三个通道），需要更改
                # calculate PSNR and SSIM
                PSNR = calculate_psnr(cropped_GT * 255, cropped_Gen * 255)
                SSIM = calculate_ssim(cropped_GT * 255, cropped_Gen * 255)
                print("%s PSNR_Y %.4f  SSIM_Y %.4f" % (test_files[idx], PSNR, SSIM))
                PSNR_all.append(PSNR)
                SSIM_all.append(SSIM)

    Mean_PSNR = sum(PSNR_all) / len(PSNR_all)
    Mean_SSIM = sum(SSIM_all) / len(SSIM_all)

    PSNR_var = np.var(PSNR_all, ddof=1)
    SSIM_var = np.var(SSIM_all, ddof=1)

    print("\nPSNR_Y_average %.4f \nSSIM_Y_average %.4f \nPSNR_var %.4f \nSSIM_var %.6f"
          % (Mean_PSNR, Mean_SSIM, PSNR_var, SSIM_var))


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_rgb_psnr(img1, img2):
    """calculate psnr among rgb channel, img1 and img2 have range [0, 255]
    """
    n_channels = np.ndim(img1)
    sum_psnr = 0
    for i in range(n_channels):
        this_psnr = calculate_psnr(img1[:, :, i], img2[:, :, i])
        sum_psnr += this_psnr
    return sum_psnr / n_channels


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(img1.shape[2]):
                ssims.append(ssim(img1[..., i], img2[..., i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def bgr2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


if __name__ == '__main__':
    main()
