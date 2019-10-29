import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

img_path = '../Baseline/test/Caucasian_1184.jpg'
img = cv2.imread(img_path)    # 原图读取
cv2.imshow('Ori', img)
# img_Guassian = cv2.medianBlur(img, 3)
# cv2.imshow('blur', img_Guassian)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mean = np.mean(img_gray)
gamma_val = math.log10(0.5)/math.log10(mean/255)    # 公式计算gamma
# image_gamma_correct = gamma_trans(img, gamma_val)   # gamma变换
image_gamma_correct = gamma_trans(img, gamma_val)
# tmp_result = cv2.flip(image_gamma_correct, 1, dst=None)
tmp_result = image_gamma_correct
cv2.imshow('gamma', tmp_result)

# 图像锐化
# tmp_result = img_Guassian
image = Image.fromarray(cv2.cvtColor(tmp_result,cv2.COLOR_BGR2RGB))
im_30 = ImageEnhance.Sharpness(image).enhance(3.0)
result = cv2.cvtColor(np.asarray(im_30),cv2.COLOR_RGB2BGR)

cv2.imshow("Sharp", result)

cv2.waitKey(0)