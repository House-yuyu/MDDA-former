from glob import glob
from tqdm import tqdm
import numpy as np
import os
from natsort import natsorted
import cv2
from joblib import Parallel, delayed
import multiprocessing
import argparse
import random
import skimage

parser = argparse.ArgumentParser(description='Generate patches from Full Resolution images')
parser.add_argument('--input_dir', default='../../data/Denoise/test/Gaussian/CBSD68', type=str, help='Directory for gt images')
parser.add_argument('--patchN_dir', default='../../data/Denoise/val/Gaussian/input', type=str, help='Directory for image patches')
parser.add_argument('--patchC_dir', default='../../data/Denoise/val/Gaussian/target', type=str, help='Directory for image patches')

parser.add_argument('--patchsize', default=256, type=int, help='Image Patch Size')
parser.add_argument('--num_patches', default=10, type=int, help='Number of patches per image')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

PS        = args.patchsize
NUM_CORES = args.num_cores
NUM_PATCHES = args.num_patches


os.makedirs(args.patchC_dir, exist_ok=True)
os.makedirs(args.patchN_dir, exist_ok=True)

#get sorted folders
files_gt = natsorted(glob(os.path.join(args.input_dir,  '*.png'))
                     + glob(os.path.join(args.input_dir, '*.bmp'))
                     + glob(os.path.join(args.input_dir, '*.jpg')))
files_no = natsorted(glob(os.path.join(args.input_dir,  '*.png'))
                     + glob(os.path.join(args.input_dir, '*.bmp'))
                     + glob(os.path.join(args.input_dir, '*.jpg')))

noisy_files, clean_files = [], []
for m in files_gt:
    clean_files.append(m)
for n in files_no:
    noisy_files.append(n)

def save_files(index):
    clean_file = clean_files[index]
    clean_img = cv2.imread(clean_file)
    noisy_file = noisy_files[index]
    noisy_img = cv2.imread(noisy_file)

    H = clean_img.shape[0]
    W = clean_img.shape[1]

    np.random.seed(seed=0)  # for reproducibility
    for j in range(NUM_PATCHES):
        rr = np.random.randint(0, H - PS)
        cc = np.random.randint(0, W - PS)
        noisy_patch = noisy_img[rr:rr + PS, cc:cc + PS, :]
        clean_patch = clean_img[rr:rr + PS, cc:cc + PS, :]

        # 生成高斯噪声
        mean = 0
        sigma = round(random.random()*50)
        gauss = np.random.normal(mean, 50, (PS, PS, 3))
        noisy = noisy_patch + gauss
        noisy = np.clip(noisy, 0, 255)  # 这里对应测试时的clip

        cv2.imwrite(os.path.join(args.patchN_dir, 'CBSD_50_{}_{}.png'.format(index + 1, j + 1)), noisy)
        cv2.imwrite(os.path.join(args.patchC_dir, 'CBSD_50_{}_{}.png'.format(index + 1, j + 1)), clean_patch)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))

files_length = natsorted(glob(os.path.join(args.patchC_dir, '*.png')))
print(len(files_length))


#['DIV2K', 'Flickr2K', 'WaterlooED', 'BSD500']:
