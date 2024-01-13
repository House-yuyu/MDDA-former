import os
import os.path
import random
import torch

import torch.utils.data as udata  #
import torchvision.transforms.functional as TF
import cv2
from natsort import natsorted
from PIL import Image


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class Dataset_train(udata.Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(Dataset_train, self).__init__()

        noisy_files = natsorted(os.listdir(os.path.join(rgb_dir, 'input')))
        clean_files = natsorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input',  x) for x in noisy_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target',        x) for x in clean_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps[0]

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class Fullsize_train(udata.Dataset):
    def __init__(self, rgb_dir, img_options=None):
        super(Fullsize_train, self).__init__()

        noisy_files = natsorted(os.listdir(os.path.join(rgb_dir, 'input')))
        clean_files = natsorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input',  x) for x in noisy_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target',          x) for x in clean_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index = index % self.sizex
        #clean = TF.to_tensor(load_img(self.clean_filenames[index]))  # c,h,w [0-1]range,RGB
        #noisy = TF.to_tensor(load_img(self.noisy_filenames[index]))

        inp_path = self.inp_filenames[index]
        tar_path = self.tar_filenames[index]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)   # c,h,w [0-1]range,RGB
        tar_img = TF.to_tensor(tar_img)

        aug = random.randint(0, 8)

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class Dataset_val(udata.Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        inp_files = natsorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = natsorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps
        ps1 = ps[0]
        ps2 = ps[1]

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps1, ps2))
            tar_img = TF.center_crop(tar_img, (ps1, ps2))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class Fullsize_val(udata.Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'target')))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'target', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return tar_img, inp_img, filename


class Dataset_test(udata.Dataset):
    def __init__(self, inp_dir, img_options):
        super(Dataset_test, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename


if __name__ == '__main__':
    dataset_train = Dataset_train('../data/patch_train')
    test = dataset_train.__getitem__(2)
    print(test)
