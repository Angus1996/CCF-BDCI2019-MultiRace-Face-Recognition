# encoding:utf-8

import dlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def face_alignment(faces):

    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") # 用来预测关键点
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0,0,face.shape[0],face.shape[1])
        shape = predictor(np.uint8(face),rec) # 注意输入的必须是uint8类型
        order = [36,45,30,48,54] # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(face, (x, y), 2, (0, 0, 255), -1)

        eye_center =((shape.part(36).x + shape.part(45).x) * 1./2, # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1./2)
        dx = (shape.part(45).x - shape.part(36).x) # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy,dx) * 180. / math.pi # 计算角度
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1) # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1])) # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
    return faces_aligned

def demo():
    im_raw = cv2.imread('./test/African_3052.jpg').astype('uint8')

    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(im_raw, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    src_faces = []
    (x, y, w, h) = rect_to_bb(rects[0])
    print(x,y,w,h)
    try:
        detect_face = im_raw[y:y+h,x:x+w]
        src_faces.append(detect_face)
        cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # for (i, rect) in enumerate(rects):
    #     (x, y, w, h) = rect_to_bb(rect)
    #     detect_face = im_raw[y:y+h,x:x+w]
    #     src_faces.append(detect_face)
    #     cv2.rectangle(im_raw, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #     cv2.putText(im_raw, "Face: {}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("src", im_raw)
        cv2.imshow("detect_face", detect_face)
        faces_aligned = face_alignment(src_faces)
        if faces_aligned[0].shape[0]>10:
            result = cv2.resize(faces_aligned[0], (112, 112))
        else:
            result = cv2.resize(im_raw, (112, 112))
    except:
        result = cv2.resize(im_raw, (112, 112))
    # print(faces_aligned[0].shape)
    print(result.shape)
    cv2.imshow("det", result)
    # for face in faces_aligned:
    #     cv2.resize(face, (112, 112))
    #     cv2.imshow("det_{}".format(i), face)
    #     i = i + 1
    cv2.waitKey(0)

def test(img_path, flip):
    image = cv2.imread(img_path, 3)
    print(image)
    image = image[-96:, :, :]
    print(image)
    cv2.imshow("image", image)
    # image = cv2.resize(image, (112, 112))
    if image is None:
        return None
    if flip:
        image = cv2.flip(image, 1, dst=None)
    cv2.imshow("flip", image)
    cv2.waitKey(0)
    # return image


if __name__ == "__main__":

    # demo()
    test('./African_0001.jpg', True)

