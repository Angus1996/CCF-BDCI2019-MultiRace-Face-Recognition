import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
import math
import insightface
import sklearn
from sklearn import preprocessing
from PIL import Image, ImageEnhance

model = insightface.app.FaceAnalysis()
ctx_id = 0
model.prepare(ctx_id = ctx_id, nms=0.4)

model2 = insightface.model_zoo.get_model('arcface_r100_v1')
model2.prepare(ctx_id = 0)

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def get_featurs(model, test_list):
    # pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        # pbar.update(1)

        # 自动gamma校正
        # img_gray=cv2.imread(img_path,0)   # 灰度图读取，用于计算gamma值
        img = cv2.imread(img_path)    # 原图读取
        img_Guassian = cv2.medianBlur(img, 3)
        img_gray = cv2.cvtColor(img_Guassian, cv2.COLOR_BGR2GRAY)
        mean = np.mean(img_gray)
        gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
        # image_gamma_correct = gamma_trans(img, gamma_val)   # gamma变换
        image_gamma_correct = gamma_trans(img_Guassian, gamma_val)
        tmp_result = cv2.flip(image_gamma_correct, 1, dst=None)
        # tmp_result = image_gamma_correct

        # 图像锐化
        # tmp_result = img_Guassian
        # tmp_result = cv2.flip(img_Guassian, 1, dst=None)
        image = Image.fromarray(cv2.cvtColor(tmp_result,cv2.COLOR_BGR2RGB))
        im_30 = ImageEnhance.Sharpness(image).enhance(3.0)
        result = cv2.cvtColor(np.asarray(im_30),cv2.COLOR_RGB2BGR)

        try:
            face = model.get(result)
            if idx == 0:
                feature = face[0].embedding.reshape(1,512)
                features = feature
            else:
                feature = face[0].embedding.reshape(1,512)
                features = np.concatenate((features, feature), axis=0)
        except:
            print(img_path)
            image = cv2.resize(result, (112,112))
            feature = model2.get_embedding(image)
            features = np.concatenate((features, feature), axis=0)
            
    return features

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

data_dir = '../Baseline/test/'                      # testset dir
name_list = [name for name in os.listdir(data_dir)]
img_paths = [data_dir + name for name in os.listdir(data_dir)]
print('Images number:', len(img_paths))

s = time.time()
features = get_featurs(model, img_paths)
t = time.time() - s
print(features.shape)
print('total time is {}, average time is {}'.format(t, t / len(img_paths)))

fe_dict = get_feature_dict(name_list, features)
print('Output number:', len(fe_dict))
sio.savemat('blur_gamma_flip_sharp3_arcface_app_embedding_test.mat', fe_dict)
