from collections import OrderedDict

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from datasets.loader import TestDataset
from model import MDDA
from model.Dehaze.DEA import Backbone
from model.Dehaze.DehazeFormer import dehazeformer_s

import os, argparse

parser = argparse.ArgumentParser()

parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='RESIDE')
parser.add_argument('--val_dataset_dir', type=str)
# parser.add_argument('--model_name', type=str, default='DEA-Net', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='../Results/Dehaze/RESIDE/SOTS-Outdoor')

# only need for evaluation                    # model_bestPSNR_OTS
parser.add_argument('--pre_trained_model', type=str, default='../checkpoints/Dehazing/models/model_bestPSNR_OTS.pth', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', default=True, help='save the infer results during validation')
opt = parser.parse_args()

opt.val_dataset_dir = os.path.join('../data/Dehaze', opt.dataset, 'SOTS-Test/outdoor')

import cv2
from PIL import Image
import math
from math import exp
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def val_ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def val_psnr(pred, gt):
    pred = pred.clamp(0, 1).cpu().numpy()
    gt = gt.clamp(0, 1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(1.0 / rmse)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def norm_zero_to_one(x):
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))


def save_heat_image(x, save_path, norm=False):
    if norm:
        x = norm_zero_to_one(x)
    x = x.squeeze(dim=0)
    C, H, W = x.shape
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    if C == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.applyColorMap(x, cv2.COLORMAP_JET)[:, :, ::-1]
    x = Image.fromarray(x)
    x.save(save_path)


def eval(val_loader, network):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    for batch in tqdm(val_loader, desc='evaluation'):
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cuda()

        with torch.no_grad():
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 8)  # 修改
            output = network(hazy_img)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]

            if opt.save_infer_results:
                save_image(output, os.path.join(opt.saved_infer_dir, batch['filename'][0]))

        psnr_tmp = val_psnr(output, clear_img)
        ssim_tmp = val_ssim(output, clear_img).item()            # utils.calculate_psnr

        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':

    val_dataset = TestDataset(os.path.join(opt.val_dataset_dir, 'input'), os.path.join(opt.val_dataset_dir, 'target'))
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)
    val_loader.num_workers = 16


    def load_checkpoint(model, weights):
        checkpoint = torch.load(weights, map_location='cuda:0')
        try:
            model.load_state_dict(checkpoint["state_dict"])  # checkpoint["state_dict"]
        except:
            state_dict = checkpoint["state_dict"]  # state_dict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    # load pre-trained model
    model = MDDA.RCUNet()
    model.cuda()
    load_checkpoint(model, opt.pre_trained_model)

    # ckpt = torch.load(os.path.join('../checkpoints/Dehazing/models/model_bestPSNR_ITS.pth'), map_location='cpu')
    # model.load_state_dict(ckpt["state_dict"])

    # start evaluation
    avg_psnr, avg_ssim = eval(val_loader, model)
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))
