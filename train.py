import os
import torch
import yaml

from utils import network_parameters, losses
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

import numpy as np
import random
from datasets.data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
# from model.test827 import RCUNet
from model.Uformer import Uformer
from model.RDNet import RDN
from model.DnCNN import DnCNN
from model.VRCNN import VRCNN
# from model.TCUNet import TCUNet
# from model.NAFNet import NAFNet
from model.Ablation import RCUNet
from model.SIDD_test1029 import RCUNet

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('options/Derain.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']

## Build Model
print('==> Build the model')
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

model = RCUNet()

p_number = network_parameters(model)

## Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
utils.mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']
save_dir = Train['SAVE_DIR']
betas = OPT['betas']
weight_decay = OPT['weight_decay']

## GPU
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model).cuda()
else:
    model.cuda()


## Optimizer
start_epoch = 1
lr_initial = float(OPT['LR_INITIAL'])
if OPT['type'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=betas, weight_decay=weight_decay)
elif OPT['type'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr_initial, betas=betas, weight_decay=weight_decay)

## Scheduler
if OPT['Scheduler'] == 'cosine':
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(OPT['LR_MIN']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
elif OPT['Scheduler'] == 'step':
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)
    scheduler.step()
elif OPT['Scheduler'] == 'none':
    pass

## Resume
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    if OPT['Scheduler'] != 'none':
        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------')

## Loss
Charloss = losses.CharbonnierLoss()
PSNR_loss = losses.PSNRLoss()
SSIM_loss = losses.SSIMLoss()
EDGE_loss = losses.EdgeLoss()
L1loss = nn.L1Loss()
mseloss = nn.MSELoss()

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                          shuffle=True, num_workers=16, drop_last=False)
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=8,
                        drop_last=False)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
best_iter = 0

mixup = utils.MixUp_AUG()

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    for i, data in enumerate(tqdm(train_loader), 0):
        model.train()
        model.zero_grad()
        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = mixup.aug(target, input_)

        restored = model(input_)
        loss = SSIM_loss(restored, target) / (-PSNR_loss(restored, target) + 0.005) + 0.05\
               * EDGE_loss(restored, target)        # 0.005

        # loss = Charloss(restored, target)
        # loss = PSNR_loss(restored, target)
        # loss = L1loss(restored, target)
        # loss = mseloss(restored, target)

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=10, norm_type=2)  #
        optimizer.step()

        # Results
        psnr_train = utils.torchPSNR(restored, target)
        print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
              (epoch, i + 1, len(train_loader), loss.item(), psnr_train))
        writer.add_scalar('train/loss', loss.item(), (epoch * len(train_loader) + i) // 1000)

    ## Validation
        if (i+1) % Train['TRAINING.VAL_AFTER_EVERY'] == 0 and epoch > 5:
            with torch.no_grad():
                model.eval()
                psnr_val_rgb = []
                ssim_val_rgb = []
                for k, data in enumerate(tqdm(val_loader), 0):
                    input_ = data[1].cuda()
                    target = data[0].cuda()

                    restored = model(input_)

                    for res, tar in zip(restored, target):
                        psnr_val_rgb.append(utils.torchPSNR(res, tar))

                psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
                # Save the best PSNR of validation
                if psnr_val_rgb > best_psnr:
                    best_psnr = psnr_val_rgb
                    best_epoch_psnr = epoch
                    best_iter = i
                    torch.save({'epoch': epoch,
                                'state_dict': model.state_dict()}, os.path.join(model_dir, "model_bestPSNR.pth"))
                print("[Epoch %d iter %d PSNR: %.4f --- best_Epoch %d best_iter %d Best_PSNR: %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch_psnr, best_iter, best_psnr))

                writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)

                if OPT['Scheduler'] != 'none':
                    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
                else:
                    pass

                torch.cuda.empty_cache()

    if OPT['Scheduler'] != 'none':
        scheduler.step()
    else:
        pass

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
