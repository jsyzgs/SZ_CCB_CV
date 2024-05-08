import cv2
import requests
import json
import numpy as np
from PIL import Image
from skimage import transform as trans

from model.face_detection import model_face_detector
from model.face_recognition import (model_sphereface, get_feature_2d_array)

'''Mock'''
with open('/hostmount/errorPicture/face_db_info.txt', 'r', encoding='utf-8') as f:
    results = f.read()
    json_result = json.loads(results)


def get_face_info(url=''):
    # 定义请求header

    HEADERS = {'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8'}
    content = requests.post(url=url, headers=HEADERS).text
    # logger.info("get_face_info content:" + content)
    if (content is not None and len(content) > 0):
        content = json.loads(content)
    return content

# 解析人脸数据库，只保留特征向量！
def get_face_feats_database(database_result):
    database_2d_array = np.empty((len(database_result), 512))

    for i, face in enumerate(database_result):
        face["tensor"] = str(face["tensor"])
        if face["tensor"] is None:
            continue
        if i != len(database_2d_array) - 1:
            database_2d_array[i:i + 1] = eval(face["tensor"])
        if i == len(database_2d_array) - 1:
            database_2d_array[i:len(database_2d_array)] = eval(face["tensor"])
    return database_2d_array


class FaceSys():
    database = json_result  # get_face_info()
    best_threshold = 1.1182
    feats_database = get_face_feats_database(database)

    def __init__(self, model_det, model_rec, similarity):
        self.model_det = model_det
        self.model_rec = model_rec
        self.similarity = similarity

    def __call__(self, bgr_3d_array):
        ...

    def get_feat(self, bgr_3d_array):
        box_2d_array, point_2d_array = self.model_det.infer(bgr_3d_array)
        # 场景：人脸比对默认单图单张人脸 （动态人脸识别可能是单图多张人脸）！！！
        if box_2d_array.shape[0] != 0:
            affine_bgr_image = get_affine_image_3d_array(
                bgr_3d_array, point_2d_array[0])
            affine_rgb_image = cv2.cvtColor(
                affine_bgr_image, cv2.COLOR_BGR2RGB)
            recognize_result = self.model_rec.infer(
                Image.fromarray(affine_rgb_image))
            feat = get_feature_2d_array(recognize_result)
            return feat

        else:
            return None

    def get_name_from_database(self, bgr_3d_array):
        face_feats = self.get_feat(bgr_3d_array)
        if face_feats is None:
            return None
        else:
            diffValue_2d_array = np.subtract(
                FaceSys.feats_database, face_feats)
            distance_1d_array = np.sum(np.square(diffValue_2d_array), 1)
            isSame_1d_array = np.less(distance_1d_array, FaceSys.best_threshold)
            max_person_score = self.similarity.calc_similarity(
                np.min(distance_1d_array))
            min_distance_index = np.argmin(distance_1d_array)
            # 根据布尔索引找结果
            isSame_result_list = list(
                np.compress(
                    isSame_1d_array,
                    FaceSys.database))
            if len(isSame_result_list) == 0 or max_person_score <= 0.9:
                person = 'unknown'
            else:
                person = FaceSys.database[min_distance_index]['person_id']

            return person,max_person_score





def get_affine_image_3d_array(image_3d_array, point_1d_array):
    """
    获取原始人脸图像仿射变换后的新图像--用于注册库
    :param original_image_3d_array: 原始图像
    :param box_1d_array: 检测扩充后的人脸框--剪裁图像
    :param point_1d_array: 人脸5个关键点
    :return: 仿射变换后的正脸图像
    """

    src = np.array([
        [30.2946, 51.6963],  # 左眼
        [65.5318, 51.5014],  # 右眼
        [48.0252, 71.7366],  # 鼻子
        [33.5493, 92.3655],  # 左嘴角
        [62.7299, 92.2041]], dtype=np.float32)  # 右嘴角
    src[:, 0] += 8.0

    # 左眼、右眼、右嘴角这3个关键点在剪裁图中的坐标
    old_point_2d_array = np.float32([
        [point_1d_array[0], point_1d_array[5]],
        [point_1d_array[1], point_1d_array[6]],
        [point_1d_array[2], point_1d_array[7]],
        [point_1d_array[3], point_1d_array[8]],
        [point_1d_array[4], point_1d_array[9]]
    ])

    tform = trans.SimilarityTransform()
    tform.estimate(old_point_2d_array, src)
    M = tform.params[0:2, :]
    affine_matrix = M
    # 做仿射变换，并缩小像素至112 * 112
    new_size = (112, 112)
    affine_image_3d_array = cv2.warpAffine(
        image_3d_array, affine_matrix, new_size, borderValue=0.0)

    return affine_image_3d_array


class FaceSimilarity():
    def __init__(self, x1, y1, x2, y2):
        self.a = np.log(((1 - y2) * y1) / ((1 - y1) * y2)) / (x2 - x1)
        self.b = np.log((1 - y1) / y1) - self.a * x1

    def calc_similarity(self, x):
        return np.round(1 / (1 + np.power(np.e, self.a * x + self.b)), 4)


model_similarity = FaceSimilarity(0.21, 0.99, 2.31, 0.05)
model_face = FaceSys(model_face_detector, model_sphereface, model_similarity)
if __name__ == '__main__':
    bgr_3d_array = cv2.imread('/hostmount/errorPicture/Yaoming_1.jpg')
    # box_2d_array, point_2d_array = model_face_detector.infer(bgr_3d_array)
    # image = get_affine_image_3d_array(bgr_3d_array, point_2d_array[0])
    feat = model_face.get_feat(bgr_3d_array)

    ...
