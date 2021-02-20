import json
import os
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/home/greatbme/mydata/data_EndoCV2021/'
# 用于创建训练集或验证集
phase = 'train'
# 训练集和验证集划分的界线
split = 2000
dataset = {'info':[],'categories':[],'images':[],'annotations':[]}
# 打开类别标签
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()

# 建立类别标签和数字id的对应关系
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

# 读取images文件夹的图片名称
_indexes = [f for f in os.listdir(os.path.join(root_path, 'images'))]

# 判断是建立训练集还是验证集
if phase == 'train':
    indexes = [line for i, line in enumerate(_indexes) if i <= split]
elif phase == 'val':
    indexes = [line for i, line in enumerate(_indexes) if i > split]

# 读取Bbox信息
annos=[]
count=[0]*12
for i, index in enumerate(indexes):
    if index[:2]!='C3':
        tempname=index[:-4]+'_mask.txt'
    else:
        tempname=index[:-4]+'.txt'
    with open(os.path.join(root_path, 'bbox',tempname)) as tr:
        anno = tr.readlines()
        if anno[0]=='\n':
            count[0]+=1
        else:
            count[len(anno)]+=1
print('count:',count)