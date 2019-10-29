import insightface
import cv2
import numpy as np

model = insightface.app.FaceAnalysis()
ctx_id = 0

model.prepare(ctx_id = ctx_id, nms=0.4)
img = cv2.imread('./African_0001.jpg')
faces = model.get(img)
for idx, face in enumerate(faces):
    # print(face)
    print(face.normed_embedding)
    # print("\t embedding :%s"%np.array(face.embedding))
    # print("\t normed_embedding :%s"%face.normed_embedding)
    

# 自己resize后的图像送进网络的输出向量与网络自己处理的图像的输出向量完全不一样
# img = cv2.imread('./Indian_1605.jpg')
# img = cv2.resize(img, (112,112))
# model2 = insightface.model_zoo.get_model('arcface_r100_v1')
# model2.prepare(ctx_id = ctx_id)
# emb = model2.get_embedding(img)
# print(type(emb))
# # print(emb[0])

# features = np.concatenate((faces[0].embedding.reshape(1,512), emb), axis=0)
# print(features.shape)
# print(faces[0].embedding == emb)