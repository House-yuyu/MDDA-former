import os
from collections import OrderedDict

import torch

import utils
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets.loader import TestDataset
from tqdm import tqdm

from model.MDDA import RCUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cuda:0')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]   # state_dict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

model = RCUNet()
model.cuda()
load_checkpoint(model, '../checkpoints/Dehazing/models/model_bestPSNR_OTS.pth')
model.eval()

## DataLoaders
val_dataset = TestDataset(os.path.join('../data/Dehaze/RESIDE/SOTS-Test/outdoor', 'input'),
                          os.path.join('../data/Dehaze/RESIDE/SOTS-Test/outdoor', 'target'))
val_loader = DataLoader(val_dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False)
val_loader.num_workers = 16

## Validation
with torch.no_grad():
    model.eval()
    psnr_val_rgb = []
    ssim_val_rgb = []
    for k, data in enumerate(tqdm(val_loader), 0):
        target = data['clear'].cuda()
        input_ = data['hazy'].cuda()

        factor = 8
        h, w = input_.shape[2], input_.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
        restored = model(input_)
        restored = restored[:, :, :h, :w]

        for res, tar in zip(restored, target):
            psnr_val_rgb.append(utils.torchPSNR(res, tar))
            ssim_val_rgb.append(utils.torchSSIM(restored, target))

    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

print('\nPSNR:{}\nSSIM:{}'.format(psnr_val_rgb, ssim_val_rgb))

