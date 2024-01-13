
import torchvision.transforms.functional as TF
import torch
from collections import OrderedDict
from model.AFD_former import RCUNet
import os

from skimage import img_as_ubyte
import cv2

# ----------------------------------- feature map visualization -----------------------------------
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    return img


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


path_img = "/home/zhanghuang_701/zx/Restormer-test/Datasets/H264/hall/3/199.bmp"  # your path to image
inp_img = TF.to_tensor(load_img(path_img))
inp_img = inp_img.unsqueeze_(0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# inputs = img_tensor.to(device)

model = RCUNet()
load_checkpoint(model, "../checkpoints/TCUNet_db/models/model_bestPSNR1.pth")
print(model)


activation = {}  # 保存获取的输出
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.eval()
model.last.register_forward_hook(get_activation(''))  # 注册钩子
_ = model(inp_img)

last = activation['']  # 结果将保存在activation字典中

print(last.shape)

denoised_RGB = last.cpu().numpy().squeeze(0).transpose((1, 2, 0))
denoised_RGB = img_as_ubyte(denoised_RGB)
denoised_BGR = denoised_RGB[..., ::-1]  # rgb->bgr, for img save

cv2.imwrite(("../Results/visual/hall.png"), denoised_BGR)
