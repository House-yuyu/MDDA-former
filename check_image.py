"""
检测某个文件夹中不完整的图片，并删除
"""
import os


"""
检测图片完整性
"""


class CheckImage(object):

    def __init__(self, img):
        with open(img, "rb") as f:
            f.seek(-2, 2)
            self.img_text = f.read()
            f.close()

    def check_jpg_jpeg(self):
        """检测jpg图片完整性，完整返回True，不完整返回False"""
        buf = self.img_text
        return buf.endswith(b'\xff\xd9')

    def check_png(self):
        """检测png图片完整性，完整返回True，不完整返回False"""

        buf = self.img_text
        return buf.endswith(b'\xaeB`\x82')

class CheckBrockImage(object):
    def __init__(self, train_dir):
        self.train_dir = train_dir
        self.completeFile = 0
        self.incompleteFile = 0

    def get_imgs(self):
        """遍历某个文件夹下的所有图片"""
        for file in os.listdir(self.train_dir):
            if os.path.splitext(file)[1].lower() == '.jpg' or os.path.splitext(file)[1].lower() == ".jpeg":
                ret = self.check_img(file)
                if ret:
                    self.completeFile += 1

                else:
                    self.incompleteFile = self.incompleteFile + 1
                    self.img_remove(file)  # 删除不完整图片

    def img_remove(self, file):
        """删除图片"""
        os.remove(self.train_dir + file)

    def check_img(self, img_file):
        """检测图片完整性，图片完整返回True,图片不完整返回False"""
        return CheckImage(self.train_dir + img_file).check_jpg_jpeg()

    def run(self):
        """执行文件"""
        self.get_imgs()
        print('不完整图片 : %d个' % self.incompleteFile)
        print('完整图片 : %d个' % self.completeFile)


if __name__ == '__main__':
    train_dir = '../data/Dehaze/RESIDE/OTS-Train/clear/target/'  # 检测文件夹
    imgs = CheckBrockImage(train_dir)
    imgs.run()

