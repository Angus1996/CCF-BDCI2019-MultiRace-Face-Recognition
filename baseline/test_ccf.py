import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
import dlib
import math
# from models import *
# import torch
# from config import Config
# from torch.nn import DataParallel
# from data import Dataset
# from torch.utils import data
# from models import resnet101
# from utils import parse_args
from scipy.spatial.distance import pdist
import insightface
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") # 用来预测关键点

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def face_alignment(faces):

    # predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
        order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            # cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned


def load_image(img_path, filp=False):
    # im_raw = cv2.imread(img_path).astype('uint8')
    # # print(img_path.split('/')[-1])

    # detector = dlib.get_frontal_face_detector()
    # gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    # rects = detector(gray, 1)
    # if len(rects)>0:
    #     src_faces = []
    #     try:
    #         (x, y, w, h) = rect_to_bb(rects[0])
    #         detect_face = im_raw[y:y+h,x:x+w]
    #         src_faces.append(detect_face)
    #         # cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         # cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #         faces_aligned = face_alignment(src_faces)
    #         # cv2.imshow("src", im_raw)
    #         if faces_aligned[0].shape[0]>10:
    #             # result_image = faces_aligned[0][-96:, :, :]
    #             result = cv2.resize(faces_aligned[0], (112, 112))
    #         else:
    #             result = cv2.resize(im_raw, (112, 112))
    #     except:
    #         result = cv2.resize(im_raw, (112, 112))
    # else:
    #     result = cv2.resize(im_raw, (112, 112))
    # cv2.imwrite('./new_test/'+img_path.split('/')[-1], result)
    # return result

    image = cv2.imread(img_path, 3)
    imgYUV = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # cv2.imshow("src", img)
    channelsYUV = cv2.split(imgYUV)
    channelsYUV[0] = cv2.equalizeHist(channelsYUV[0])

    channels = cv2.merge(channelsYUV)
    result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
    image = result[-96:, :, :]
    image = cv2.resize(image, (112, 112))
    if image is None:
        return None
    if filp:
        image = cv2.flip(image, 1, dst=None)
    return image

model = insightface.model_zoo.get_model('arcface_r100_v1')
model.prepare(ctx_id = 0)

# def get_featurs(model, test_list):

#     device = torch.device("cuda")

#     pbar = tqdm(total=len(test_list))
#     for idx, img_path in enumerate(test_list):
#         pbar.update(1)


#         dataset = Dataset(root=img_path,
#                       phase='test',
#                       input_shape=(1, 112, 112))

#         trainloader = data.DataLoader(dataset, batch_size=1)
#         for img in trainloader:
#             img = img.to(device)
#             if idx == 0:
#                 feature = model(img)
#                 feature = feature.detach().cpu().numpy()
#                 features = feature
#             else:
#                 feature = model(img)
#                 feature = feature.detach().cpu().numpy()
#                 features = np.concatenate((features, feature), axis=0)
#     return features

def get_featurs(model, test_list):
    pbar = tqdm(total=len(test_list))
    for idx, img_path in enumerate(test_list):
        pbar.update(1)
        img = load_image(img_path)
        if idx == 0:
            feature = model.get_embedding(img)
            features = feature
        else:
            feature = model.get_embedding(img)
            features = np.concatenate((features, feature), axis=0)
    return features

def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        fe_dict[each] = features[i]
    return fe_dict

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def cosine_similarity(x1, x2):
    X = np.vstack([x1, x2])
    d2 = 1 - pdist(X, 'cosine')
    return d2

def euclidean_distance(face_encodings, face_to_compare):
    return np.linalg.norm(face_encodings - face_to_compare, axis=0)


# 加载训练过得模型
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# checkpoint = '40checkpoint.tar'
# print('loading model: {}...'.format(checkpoint))
# checkpoint = torch.load(checkpoint)
# model = checkpoint['model'].module.to(device)

# model.eval()

data_dir = './test/'                      # testset dir
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
sio.savemat('histrogramed_arcface_embedding_test.mat', fe_dict)

######## cal_submission.py #########

face_features = sio.loadmat('histrogramed_arcface_embedding_test.mat')
# face_features2 = sio.loadmat('face_embedding_test.mat')
print('Loaded mat')
sample_sub = open('./submission_template.csv', 'r')  # sample submission file dir
sub = open('[angusaha]_results.csv', 'w', encoding='utf-8')
print('Loaded CSV')

lines = sample_sub.readlines()
# pbar = tqdm(total=len(lines))
# all_Euclidean_score = []
# for line in lines:
#     pair = line.split(',')[0]
#     a, b = pair.split(':')
#     all_Euclidean_score.append(euclidean_distance(face_features[a][0], face_features[b][0]))
#     # pbar.update(1)
# minvalue_Euclidean_score = min(all_Euclidean_score)
# maxvalue_Euclidean_score = max(all_Euclidean_score)
# print(maxvalue_Euclidean_score)
# print(minvalue_Euclidean_score)

pbar = tqdm(total=len(lines))
for i in range(len(lines)):
    line = lines[i]
# for line in lines:
    pair = line.split(',')[0]
    # print(pair)
    sub.write(pair + ',')
    a, b = pair.split(':')
    # face_features_a = np.concatenate((face_features[a][0], face_features2[a][0]),axis=0)
    # face_features_b = np.concatenate((face_features[b][0], face_features2[b][0]),axis=0)
    # cos_score = 0.5 + 0.5 * (cosin_metric(face_features[a][0], face_features[b][0]))
    # score2 = (float(score)+0.3)/1.3

    # Euclidean_score = (all_Euclidean_score[i]-minvalue_Euclidean_score)/(maxvalue_Euclidean_score-minvalue_Euclidean_score)

    # score = '%.5f' % (0.5 * cos_score + 0.5 * Euclidean_score)

    score = '%.5f' % cosin_metric(face_features[a][0], face_features[b][0])
    # if float(score) < 0.0:
    #     score = '0.0's
    # score = '%.5f' % cosine_similarity(face_features_a, face_features_b)
    # sub.write(a + ',' + b)
    sub.write(score + '\n')
    # sub.writerow(pair + ',' + score)
    pbar.update(1)


sample_sub.close()
sub.close()