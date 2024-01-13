import numpy as np
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import utils
from collections import OrderedDict
from natsort import natsorted
from glob import glob

from model.MDDA import RCUNet
from skimage import img_as_ubyte


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights, map_location='cuda:0')
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


parser = argparse.ArgumentParser(description='Image Deraining')

parser.add_argument('--input_dir', default='../data/Deblur/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../Results/Deblur', type=str, help='Directory for results')
parser.add_argument('--weights', default='../checkpoints/Deblurring/models/model_bestPSNR.pth', type=str, help='Path to weights')


args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

model = RCUNet()
model.cuda()
load_checkpoint(model, args.weights)
model.eval()

factor = 8
datasets = ['HIDE', 'RealBlur_R', 'RealBlur_J']  # 'GoPro', 'HIDE', 'RealBlur_R', 'RealBlur_J'

for dataset in datasets:
    result_dir  = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir, exist_ok=True)

    inp_dir = os.path.join(args.input_dir, 'test', dataset, 'input')
    files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
    with torch.no_grad():
        for file_ in tqdm(files):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            img = np.float32(utils.load_img(file_))/255.
            img = torch.from_numpy(img).permute(2, 0, 1)
            input_ = img.unsqueeze(0).cuda()

            # Padding in case images are not multiples of 8
            h,w = input_.shape[2], input_.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_ = F.pad(input_, (0,padw,0,padh), 'reflect')

            restored = model(input_)

            # Unpad images to original dimensions
            restored = restored[:,:,:h,:w]

            restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
