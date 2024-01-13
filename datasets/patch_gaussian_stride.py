from glob import glob

import random
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--input_dir', default='../../data/Flickr2K', type=str, help='Directory for gt images')
parser.add_argument('--patchN_dir', default='../../data/patch_gaussian_new/distortion', type=str, help='Directory for image patches')
parser.add_argument('--patchC_dir', default='../../data/patch_gaussian_new/gt', type=str, help='Directory for image patches')

parser.add_argument('--patchsize', default=256, type=int, help='Image Patch Size')
parser.add_argument('--stride', default=160, type=int, help='')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

PS        = args.patchsize
std       = args.stride
NUM_CORES = args.num_cores

os.makedirs(args.patchC_dir, exist_ok=True)
os.makedirs(args.patchN_dir, exist_ok=True)


#get sorted folders
files_gt = natsorted(glob(os.path.join(args.input_dir,  '*.png')))  # natsorted会比sorted排序更自然,赋值给了files才生效
files_no = natsorted(glob(os.path.join(args.input_dir, '*.png')))

noisy_files, clean_files = [], []
for m in files_gt:
    clean_files.append(m)
for n in files_no:
    noisy_files.append(n)

def data_aug(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def save_files(index):
    clean_file = clean_files[index]
    clean_img = cv2.imread(clean_file)
    noisy_file = noisy_files[index]
    noisy_img = cv2.imread(noisy_file)

    H = clean_img.shape[0]
    W = clean_img.shape[1]

    for i in range(0, H - PS + 1, std):
        for j in range(0, W - PS + 1, std):
            x = clean_img[i:i + PS, j:j + PS, :]
            y = noisy_img[i:i + PS, j:j + PS, :]

            # 做一次增广
            idx = np.random.randint(0, 8)  # 必须保持2次随机模式一样
            x_aug = data_aug(x, idx)
            y_aug = data_aug(y, idx)

            # 生成高斯噪声
            mean = 0
            sigma = round(random.random() * 50)
            gauss = np.random.normal(mean, sigma, (PS, PS, 3))
            noisy = x_aug + gauss
            noisy = np.clip(noisy, 0, 255)

            cv2.imwrite(os.path.join(args.patchN_dir, 'Flickr2K_{}_{}_{}.png'.format(index + 1, j + 1, idx)), noisy)
            cv2.imwrite(os.path.join(args.patchC_dir, 'Flickr2K__{}_{}_{}.png'.format(index + 1, j + 1, idx)), y_aug)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))

files_length = natsorted(glob(os.path.join(args.patchC_dir, '*.png')))
print(len(files_length))
