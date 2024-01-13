import argparse

import os

from utils.image_utils import *
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from datasets.dataset_zx import Dataset_test
from tqdm import tqdm
from collections import OrderedDict
import torch
import math


from model.Uformer import Uformer


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


def expand2square(timg, factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h, w) / float(factor)) * factor)  # 向上取整

    img = torch.zeros(1, 3, X, X).type_as(timg)  # 3, h,w
    mask = torch.zeros(1, 1, X, X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
    mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

    return img, mask


def run():
    parser = argparse.ArgumentParser(description="DIBR_Test")
    parser.add_argument("--logdir", type=str, default="../checkpoints/Hilo/models/model_bestPSNR.pth", help='path of log files')

    parser.add_argument('--input_dir', default='../Datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='../Results/', type=str, help='Directory for results')

    parser.add_argument('--encoding_format', default='H264', type=str, help='H264, H265')
    parser.add_argument('--dataset', default='lovebird', type=str, help='[kendo, lovebird, newspaper, hall, champagne, pantomime]')
    parser.add_argument('--noisy_level', default='1', type=str, help='[1,2,3,4,5]')
    parser.add_argument('--nums', default=100, type=int, help='image number of every folder')  # for test time
    opt = parser.parse_args()


    ####### load model ###############
    print('Loading model ...\n')

    model = Uformer()

    model.cuda()
    load_checkpoint(model, opt.logdir)
    model.eval()

    rgb_dir_test = os.path.join(opt.input_dir, opt.encoding_format, opt.dataset)
    test_dataset = Dataset_test(rgb_dir_test, opt.noisy_level)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False,
                             pin_memory=False)

    save_dir = os.path.join(opt.result_dir, opt.encoding_format,
                            opt.dataset, opt.noisy_level)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print('Loading data info ...\n')
    import time
    start = time.time()
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            rgb_gt = data_test[0].numpy().squeeze().transpose((1, 2, 0))
            rgb_noisy, mask = expand2square(data_test[1].cuda(), factor=16)

            rgb_restored = model(rgb_noisy, 1 - mask)

            rgb_restored = torch.masked_select(rgb_restored, mask.bool()).reshape(1, 3, rgb_gt.shape[0],
                                                                                  rgb_gt.shape[1])
            rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))

            save_img(os.path.join(save_dir, 'denoised_%d.png' % ii), img_as_ubyte(rgb_restored))



    print('Process time each image:', (time.time() - start) / opt.nums)


if __name__ == "__main__":
    run()
