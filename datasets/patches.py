from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--gt_dir', default='../../data/Low_light/LOL/train/target', type=str, help='Directory for gt images')
parser.add_argument('--input_dir', default='../../data/Low_light/LOL/train/input', type=str, help='Directory for gt images')
parser.add_argument('--patch_dir', default='../../data/Low_light/LOL/train1', type=str, help='Directory for image patches')

parser.add_argument('--patchsize', default=512, type=int, help='Image Patch Size')
parser.add_argument('--stride', default=10, type=int, help='')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

PS        = args.patchsize
std       = args.stride
NUM_CORES = args.num_cores

noisy_patchDir = os.path.join(args.patch_dir, 'input')
clean_patchDir = os.path.join(args.patch_dir, 'target')



#get sorted folders
files_gt = natsorted(glob(os.path.join(args.gt_dir,    '*.png')))
files_no = natsorted(glob(os.path.join(args.input_dir, '*.png')))

noisy_files, clean_files = [], []
for m in files_gt:
    clean_files.append(m)
for n in files_no:
    noisy_files.append(n)


def data_aug(img, mode=0):
    # data augmentation
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

    H = clean_img.shape[0]  # 此时的img是依次读取
    W = clean_img.shape[1]

    for i in range(0, H - PS + 1, std):
        for j in range(0, W - PS + 1, std):
            x = clean_img[i:i + PS, j:j + PS, :]
            y = noisy_img[i:i + PS, j:j + PS, :]


            cv2.imwrite(os.path.join(clean_patchDir,
                                     '{}_{}.png'.format(index+1, (i+1))), x)
            cv2.imwrite(os.path.join(noisy_patchDir,
                                     '{}_{}.png'.format(index+1, (i+1))), y)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))

# files_length = natsorted(glob(os.path.join('/home/zhanghuang_701/zx/Restormer-test/data/patch_flusion/vi', '*.png')))
# print(len(files_length))
