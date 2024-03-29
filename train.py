import os
import torch
import yaml
import argparse
import numpy as np
import random

import utils
from utils import losses, get_temperature, mkdir
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from datasets.data_RGB import get_training_data, get_validation_data
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from model.MDDA_former import MDDA_former

from torch.cuda.amp.grad_scaler import GradScaler

# Parse arguments
parser = argparse.ArgumentParser(description='Training')
parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: %(default)s)")
parser.add_argument('--use_amp', default=True, action='store_true', help='use automatic mixed precision')
parser.add_argument("--local_rank", default=-1, type=int)

# odconv
parser.add_argument('--temp_epoch', type=int, default=10, help='number of epochs for temperature annealing')
parser.add_argument('--temp_init', type=float, default=30.0, help='initial value of temperature')
parser.add_argument('--reduction', type=float, default=0.25, help='reduction ratio used in the attention module')
parser.add_argument('--kernel_num', type=int, default=2, help='number of convolutional kernels in ODConv')

args = parser.parse_args()

dist.init_process_group(backend='nccl', init_method='env://')

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

print('==> Build the model')

## Training model path direction
mode = opt['MODEL']['MODE']
model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
mkdir(model_dir)
train_dir = Train['TRAIN_DIR']
val_dir = Train['VAL_DIR']
save_dir = Train['SAVE_DIR']
betas_ = OPT['betas']
weight_decay = OPT['weight_decay']

## GPU
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()

model = RCUNet()
model.to(device)
model = nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[args.local_rank],
                                            output_device=args.local_rank, find_unused_parameters=True)

## Optimizer
start_epoch = 1
lr_initial = float(OPT['LR_INITIAL'])

if OPT['type'] == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=lr_initial, betas=betas_, weight_decay=weight_decay)
elif OPT['type'] == 'AdamW':
    optimizer = optim.AdamW(model.parameters(), lr=lr_initial, betas=betas_, weight_decay=weight_decay)

if args.use_amp:
    scaler = GradScaler()
else:
    scaler = None

## Scheduler
if OPT['Scheduler'] == 'cosine':
    warmup_epochs = 5
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
    path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')  # bestPSNR or latest
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
PSNR_loss = losses.PSNRLoss()

## DataLoaders
print('==> Loading datasets')
train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']})
val_dataset = get_validation_data(val_dir, {'patch_size': Train['VAL_PS']})

train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, sampler=train_sampler, drop_last=True)

val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=10,
                        num_workers=8, pin_memory=True,
                        sampler=val_sampler, drop_last=False)

train_loader_len, val_loader_len = len(train_loader), len(val_loader)

# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {args.batch_size}
    Learning rate:      {OPT['LR_INITIAL']}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
best_iter = 0

eval_now = train_loader_len//2
print(f"\nEval after every {eval_now} Iterations !!!\n")
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

        if epoch < args.temp_epoch and hasattr(model.module, 'net_update_temperature'):
            temp = get_temperature(i + 1, epoch, train_loader_len,
                                   temp_epoch=args.temp_epoch, temp_init=args.temp_init)
            model.module.net_update_temperature(temp)

        if args.use_amp:
            with torch.cuda.amp.autocast():
                pred_ = model(input_)
                loss = PSNR_loss(pred_, target)  
        else:
            pred_ = model(input_)
            loss = PSNR_loss(pred_, target)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.01)
        optimizer.step()

        psnr_train = utils.torchPSNR(pred_, target)

        if args.local_rank == 0:
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                 (epoch, i + 1, len(train_loader), loss.item(), psnr_train))
        writer.add_scalar('train/loss', loss.item(), (epoch * len(train_loader) + i) // 1000)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    ## Validation
        if (i+1) % eval_now == 0 and epoch > 30:
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
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_bestPSNR.pth"))

                if args.local_rank == 0:
                    print("[Epoch %d iter %d PSNR: %.4f --- best_Epoch %d best_iter %d Best_PSNR: %.4f] " % (
                        epoch, i, psnr_val_rgb, best_epoch_psnr, best_iter, best_psnr))

                writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)

                torch.cuda.empty_cache()

    if OPT['Scheduler'] != 'none':
        scheduler.step()
    else:
        pass

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}, os.path.join(model_dir, "model_latest.pth"))
