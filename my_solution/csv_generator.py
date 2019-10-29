import os
import cv2
import numpy as np
import time
import scipy.io as sio
from collections import OrderedDict
from tqdm import tqdm
import math
from scipy.spatial.distance import pdist

def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def get_sim(x1, x2):
    return np.dot(x1, x2.T)

arc_face_features = sio.loadmat('blur_gamma_flip_sharp3_arcface_app_embedding_test.mat')
print('Loaded mat')
arc_face_features2 = sio.loadmat('blur_gamma_sharp3_arcface_app_embedding_test.mat')
print('Loaded mat')
sample_sub = open('./submission_template.csv', 'r')  # sample submission file dir
sub = open('[angusaha]_results.csv', 'w', encoding='utf-8')
print('Loaded CSV')

lines = sample_sub.readlines()

pbar = tqdm(total=len(lines))
for line in lines:
    pair = line.split(',')[0]
    # print(pair)
    sub.write(pair + ',')
    a, b = pair.split(':')
    # feature_a = arc_face_features[a][0] + arc_face_features2[a][0]
    # feature_b = arc_face_features[b][0] + arc_face_features2[b][0]
    # feature_a = arc_face_features2[a][0]
    # feature_b = arc_face_features2[b][0]

    # sim = (cosin_metric(feature_a, feature_b))
    sim1 = cosin_metric(arc_face_features[a][0], arc_face_features[b][0])
    sim2 = cosin_metric(arc_face_features2[a][0], arc_face_features2[b][0])
    sim = (sim1 + sim2) / 2.
    score = '%.5f' % sim
    sub.write(score + '\n')
    # sub.writerow(pair + ',' + score)
    pbar.update(1)
sample_sub.close()
sub.close()