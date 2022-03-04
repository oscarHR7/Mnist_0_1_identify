'''MNIST 转换为 .jpg格式
import keras
from keras.datasets import mnist
import numpy as np
from PIL import Image, ImageOps
import os
def save_image(filename, data_array):
    im = Image.fromarray(data_array.astype('uint8'))
    im_invert = ImageOps.invert(im)
    im_invert.save(filename)
# 加载 MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()
DIR_NAME = "mnist_images"
if os.path.exists(DIR_NAME) == False:
    os.mkdir(DIR_NAME)
# 保存图片
i = 0
for data_x,data_y in [[x_train,y_train], [x_test,y_test]]:
    print("[---------------------------------------------------------------]")
    for x,y in zip(data_x,data_y):
        file_path=os.path.join(DIR_NAME,str(y))
        os.makedirs(file_path,exist_ok=True)
        filename = "{0}/{1:05d}.jpg".format(file_path,i)
        print(filename)
        save_image(filename, x)
        i += 1
'''
'''
#对数据集中的0、1文件夹单独进行命名
import os
def rename(path,label):
        filelist = os.listdir(path)
        i = 0
        # 仅用于数字开头的图片命名方法
        for item in filelist:
                if item.endswith('.jpg'):
                        i = i + 1
                        name = str(i)
                        src = os.path.join(os.path.abspath(path),item)
                        # 原始图像的路径
                        dst = os.path.join(os.path.abspath(path), str(label) + '_' + name + '.jpg')
                        # 目标图像路径
                try:
                        os.rename(src, dst)#格式重命名为0_1其中第一个0表示当前图片显示的数字方便后续拆解为label
                        print('rename from %s to %s'%(src,dst))
                        # 将转换结果在终端打印出来以便检查
                except:
                        continue
path_MNIST_0 = 'D:\py_test\Mnist_0_1_identify\mnist_images\MNIST_0'
rename(path_MNIST_0, 0)
path_MNIST_1 = 'D:\py_test\Mnist_0_1_identify\mnist_images\MNIST_1'
rename(path_MNIST_1, 1)
'''
'''
#说明：为了方便处理此处直接将两个MNIST_1和MNIST_0的处理好标签的图片直接放入文件夹
import os
from shutil import copy

def copy_to_goal_dir(ori_dir,new_dir,new_dir_name):
    filelist = os.listdir(ori_dir)
    for item in filelist:
        ori_path = os.path.join(ori_dir, item)
        new_path = new_dir + "\\" + new_dir_name
        copy(ori_path, new_path)

ori_dir_0 = 'D:\py_test\Mnist_0_1_identify\mnist_images\MNIST_0'
ori_dir_1 = 'D:\py_test\Mnist_0_1_identify\mnist_images\MNIST_1'
new_dir = 'D:\py_test\Mnist_0_1_identify'
new_dir_name = 'Mnist_Data'#新文件夹中包含了0和1的各类图像
copy_to_goal_dir(ori_dir_0, new_dir, new_dir_name)
copy_to_goal_dir(ori_dir_1, new_dir, new_dir_name)
'''

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPool2D, GlobalMaxPool2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
#对于相关参数的设定
DATA_DIR = 'D:\py_test\Mnist_0_1_identify\Mnist_Data'

TRAIN_TEST_SPLIT = 0.7
IMAGE_HEIGHT, IMAGE_WIDTH = 28, 28
IMAGE_CHANNELS = 1


def parse_filepath(filepath):#通过解析数据集照片命名获得相对应的所需的数字信息
    try:
        path, filename = os.path.split(filepath)#提取出路径和照片的名称
        filename, ext = os.path.splitext(filename)#分离照片名中的扩展名和原始照片名
        number, counter = filename.split("_")
        return int(number)
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None

files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))#解析所有照片命名所对应的所需信息
df_origin = pd.DataFrame(attributes)
df_origin['file'] = files
df_origin.columns = ['number', 'file']#添加列标签
#print(df_origin)

df = df_origin.copy()
df_random = np.random.permutation(len(df))#对原有数据集合打乱
#print(df_random)

train_up_to = int(len(df)*TRAIN_TEST_SPLIT)#训练集占原数据集的百分比
#制作训练集和测试集 切片只切的是序号而没有其他信息
train_idx = df_random[:train_up_to]
test_idx = df_random[train_up_to:]
#将 train_idx 进一步拆分为训练和验证集
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

#生成器的制作
def get_data_number(df, indices, for_training, batch_size=16):
    images, numbers = [], []
    counter = 0
    while True:
        for i in indices:
            r = df.iloc[i]#对应行号的所有信息
            file, number = r['file'], r['number']
            im = Image.open(file)
            im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
            im = np.array(im)/255.0#归一化
            images.append(im)
            numbers.append(number)
            if len(images) >= batch_size:
                yield np.array(images), [np.array(numbers)]
                images, numbers = [], []
        if not for_training:
            break

#model的单层构建
def conv_block(input_data, filters=32, bn=True, pool=True, kernel_size=3, activation='relu'):
    return_x = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation)(input_data)
    if bn:
        return_x = BatchNormalization()(return_x)
    if pool:
        return_x = MaxPool2D()(return_x)
    return return_x



input_layer = Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))
x = conv_block(input_layer, filters=32, bn=False, pool=False)
x = conv_block(x, filters=32*2)
x = conv_block(x, filters=32*3)
bottleneck = GlobalMaxPool2D()(x)
number_x = Dense(16, activation='relu')(bottleneck)
number_output = Dense(1, activation='sigmoid', name='number_output')(number_x)
model = Model(inputs=input_layer, outputs=[number_output])
model.compile(optimizer='rmsprop',
              loss={
                  'number_output': 'mse'},
              loss_weights={
                  'number_output': 2.},
              metrics={
                  'number_output': 'mae'})


from tensorflow.keras.callbacks import ModelCheckpoint
batch_size = 64
valid_batch_size = 64

train_gen = get_data_number(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_number(df, valid_idx, for_training=True, batch_size=valid_batch_size)

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx) // batch_size,
                    epochs=5,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx) // valid_batch_size)
import matplotlib
import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')

plt.show()
#测试预测集
test_gen = get_data_number(df, test_idx, for_training=False, batch_size=128)
print(dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//128))))
