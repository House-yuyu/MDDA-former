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
parser.add_argument('--vi_dir',    default='../../data/MSRS/Visible/train/MSRS', type=str, help='Directory for gt images')
parser.add_argument('--ir_dir', default='../../data/MSRS/Infrared/train/MSRS', type=str, help='Directory for gt images')
parser.add_argument('--label_dir', default='../../data/MSRS/Label/train/MSRS', type=str, help='Directory for gt images')

parser.add_argument('--patch_dir', default='../../data/patch_fusion', type=str, help='Directory for image patches')

parser.add_argument('--patchsize', default=64, type=int, help='Image Patch Size')
parser.add_argument('--stride', default=64, type=int, help='')

parser.add_argument('--num_cores', default=16, type=int, help='Number of CPU Cores')
args = parser.parse_args()

PS        = args.patchsize
std       = args.stride
NUM_CORES = args.num_cores

ir_patchDir = os.path.join(args.patch_dir, 'ir')
vi_patchDir = os.path.join(args.patch_dir, 'vi')
label_patchDir = os.path.join(args.patch_dir, 'label')

#get sorted folders
files_vi = natsorted(glob(os.path.join(args.vi_dir,    '*.png')))
files_ir = natsorted(glob(os.path.join(args.ir_dir, '*.png')))
files_label = natsorted(glob(os.path.join(args.label_dir, '*.png')))

vi_files, ir_files, label_files = [], [], []
for i in files_vi:
    vi_files.append(i)
for i in files_ir:
    ir_files.append(i)
for i in files_label:
    label_files.append(i)

def save_files(index):
    vi_file = vi_files[index]
    vi_img = cv2.imread(vi_file)
    ir_file = ir_files[index]
    ir_img = cv2.imread(ir_file, 0)
    label_file = label_files[index]
    label_img = cv2.imread(label_file, 0)

    H = vi_img.shape[0]  # 此时的img是依次读取
    W = vi_img.shape[1]

    for i in range(0, H - PS + 1, std):
        for j in range(0, W - PS + 1, std):
            x = vi_img[i:i + PS, j:j + PS, :]
            y = ir_img[i:i + PS, j:j + PS, :]
            z = label_img[i:i + PS, j:j + PS, :]

            cv2.imwrite(os.path.join(vi_patchDir,
                                     '{}_{}_{}.png'.format(index+1, i+1, (j+1)) ), x)
            cv2.imwrite(os.path.join(ir_patchDir,
                                     '{}_{}_{}.png'.format(index+1, (i+1), (j+1)) ), y)
            cv2.imwrite(os.path.join(label_patchDir,
                                     '{}_{}_{}.png'.format(index+1, (i+1), (j+1)) ), z)


Parallel(n_jobs=NUM_CORES)(delayed(save_files)(i) for i in tqdm(range(len(vi_files))))

files_length = natsorted(glob(os.path.join('/home/zhanghuang_701/zx/Restormer-test/data/patch_fusion/vi', '*.png')))
print(len(files_length))
