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
parser.add_argument('--input_dir', default='../../data/CBSD68', type=str, help='Directory for gt images')
parser.add_argument('--noisy_dir', default='../../Datasets/Gaussian/CBSD68/25', type=str, help='Directory for image patches')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

NUM_CORES = args.num_cores

#get sorted folders
files_gt = natsorted(glob(os.path.join(args.input_dir,  '*.png')))  # natsorted会比sorted排序更自然,赋值给了files才生效

clean_files = []
for m in files_gt:
    clean_files.append(m)


def save_files(index):
    clean_file = clean_files[index]
    clean_img = cv2.imread(clean_file)


    H = clean_img.shape[0]
    W = clean_img.shape[1]

    # for j in range(len(clean_img)):

    # 生成高斯噪声
    mean = 0
    # sigma = round(random.random()*45+5)
    gauss = np.random.normal(mean, 25, (H, W, 3))
    noisy = clean_img + gauss
    noisy = np.clip(noisy, 0, 255)

    cv2.imwrite(os.path.join(args.noisy_dir, 'CBSD68_{}.png'.format(index + 1)), noisy)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(clean_files))))

files_length = natsorted(glob(os.path.join(args.noisy_dir, '*.png')))
print(len(files_length))
