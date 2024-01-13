import argparse
from utils.image_utils import *
from skimage import img_as_ubyte
from torch.utils.data import DataLoader
from datasets.dataset_zx import Dataset_test
from tqdm import tqdm
from collections import OrderedDict

# from model.RCUNet_db import RCUNet
from model.RDNet import RDN
from model.TSAN import TSAN
from model.VRCNN import VRCNN
from model.MIRNet import MIRNet
from model.DnCNN import DnCNN
from model.NAFNet import NAFNet
from model.Restormer import Restormer
from model.TCUNet import TCUNet
from model.QECNN import QECNN
# from model.Hilo import RCUNet
from model.RDEN import RDEN
from model.Ablation import RCUNet
# from model.test827 import RCUNet

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


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


def run():
    parser = argparse.ArgumentParser(description="DIBR_Test")
    parser.add_argument("--logdir", type=str, default="../checkpoints/TCUNet_db/models/model_bestPSNR.pth", help='path of log files')

    parser.add_argument('--input_dir', default='../Datasets/', type=str, help='Directory of validation images')
    parser.add_argument('--result_dir', default='../Results/', type=str, help='Directory for results')

    parser.add_argument('--encoding_format', default='H264', type=str, help='H264, H265')
    parser.add_argument('--dataset', default='newspaper', type=str, help='[kendo, lovebird, newspaper, hall, champagne, pantomime]')
    parser.add_argument('--noisy_level', default='4', type=str, help='[1,2,3,4,5]')
    parser.add_argument('--nums', default=100, type=int, help='image number of every folder')  # for test time
    opt = parser.parse_args()


    ####### load model ###############
    print('Loading model ...\n')

    model = RCUNet()

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

            input_ = data_test[1].cuda()


            restored = model(input_)

            denoised_RGB = torch.clamp(restored[0], 0, 1).cpu().numpy().transpose((1, 2, 0))
            denoised_RGB = img_as_ubyte(denoised_RGB)

            denoised_BGR  = denoised_RGB[..., ::-1]  # rgb->bgr, for img save
            cv2.imwrite(os.path.join(save_dir, 'denoised_%d.png' % ii), denoised_BGR)

    print('Process time each image:', (time.time() - start) / opt.nums)


if __name__ == "__main__":
    run()
