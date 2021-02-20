import json
import os
import cv2

# 根路径，里面包含images(图片文件夹)，annos.txt(bbox标注)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
root_path = '/home/greatbme/mydata/data_EndoCV2021/'
# 用于创建训练集或验证集
phase = 'val'
# 训练集和验证集划分的界线
split = 1200
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
for i, index in enumerate(indexes):
    if index[:2]!='C3':
        tempname=index[:-4]+'_mask.txt'
    else:
        tempname=index[:-4]+'.txt'
    with open(os.path.join(root_path, 'bbox',tempname)) as tr:
        anno = tr.readlines()
        annos.append(anno)

for k, index in enumerate(indexes):
    # 用opencv读取图片，得到图像的宽和高
    im = cv2.imread(os.path.join(root_path, 'images/') + index)
    height, width, _ = im.shape

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                              'id': k,
                              'width': width,
                              'height': height})
anno_id=0
for k, anno_it in enumerate(annos):
    if anno_it[0]=='\n':
        continue
    for i,anno in enumerate(anno_it):
        parts = anno.strip().split()
        cls_id = 1
        # x_min
        x1 = float(parts[1])
        # y_min
        y1 = float(parts[2])
        # x_max
        x2 = float(parts[3])
        # y_max
        y2 = float(parts[4])
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        anno_id+=1
        dataset['annotations'].append({
            'area': width * height,
            'bbox': [x1, y1, width, height],
            'category_id': int(cls_id),
            'id': anno_id,
            'image_id': k,
            'iscrowd': 0,
            # mask, 矩形是从左上角点按顺时针的四个顶点
            'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
        })

# 保存结果的文件夹
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
  os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
  json.dump(dataset, f)