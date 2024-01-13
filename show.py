
from collections import OrderedDict

from CnnModel.torch_cka.utils import add_colorbar
from model.MSD_Former import RCUNet



# os.environ["CUDA_VISIBLE_DEVICES"] = "5"


from torchvision.transforms import transforms
import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


# 导入数据
def get_image_info(image_dir):
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    image_info = Image.open(image_dir).convert('RGB')
    # 数据预处理方法
    image_transform = transforms.Compose([
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_info = image_transform(image_info)
    image_info = image_info.unsqueeze(0)
    return image_info


# 获取第k层的特征图
def get_k_layer_feature_map(feature_extractor, k, x):
    with torch.no_grad():
        for index, layer in enumerate(feature_extractor):
            x = layer(x)
            if k == index:
                return x


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


#  可视化特征图
def show_feature_map(feature_map):
    feature_map = feature_map.squeeze(0)
    feature_map = feature_map.cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    feature_map = normalization(feature_map)
    for index in range(1, 9):
        plt.subplot(3, 3, index+1)
        # feature_map = normalization(feature_map[index-1])
        a = plt.imshow(feature_map[index-1], origin='lower', cmap='rainbow')
        plt.axis('off')

        add_colorbar(a)
        plt.savefig("feature.jpg", dpi=700)
    plt.show()



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


# new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'encoder_level2': '1'})  # 'encoder_level2.1'   'encoder_level1': '1'
# out = new_model(img)
#
# tensor_ls = [(k, v) for k, v in out.items()]
#
# # 这里选取layer2的输出画特征图
# v = tensor_ls[1][1]

if __name__ == '__main__':

    image_dir = "00004.bmp"

    # 定义提取第几层的feature map
    k = 1

    # model = RCUNet()
    # # model.cuda()
    # load_checkpoint(model, '../checkpoints/Deraining/models/model_latest2.pth')
    model = models.alexnet(pretrained=True)

    use_gpu = torch.cuda.is_available()
    use_gpu =False

    # 读取图像信息
    image_info = get_image_info(image_dir)

    if use_gpu:
        model = model.cuda()
        image_info = image_info.cuda()
    # alexnet只有features部分有特征图
    # classifier部分的feature map是向量
    feature_extractor = model.features
    feature_map = get_k_layer_feature_map(feature_extractor, k, image_info)
    show_feature_map(feature_map)

