# coding=utf-8
# 将jpg复制并重命名
import os, shutil

path_src = '../../data/Dehaze/RESIDE/SOTS/indoor/target'
path_dst = '../../Results/Dehaze/RESIDE/SOTS/indoor_gt'
n = os.listdir(path_src)
len0 = len(n)
# print(len0)
bias = 0  # 起始序列
ends = 500  # 结束序列
cnt = 1


for i in range(0, 50):
    for j in range(0, 10):
        file_src = path_src + 'image{:08d}.'.format(i)  # str(i) + '.jpg'
        print(file_src)
        file_dst = path_dst + str(cnt) + '.png'
        if os.path.exists(file_src):  # 判断文件是否存在，以防中间序列不连续
            shutil.copyfile(file_src, file_dst)
            cnt = cnt + 1


