import base64
import cv2
from PIL import Image

from model.face_sys import get_affine_image_3d_array
from model.face_detection import model_face_detector
from model.face_recognition import (model_sphereface, get_feature_2d_array)
from utils.image_utils import open_image_by_cv2


def extract_face_feats_business(uuid, image_base64):
    response_json = {}
    bgr_3d_array = open_image_by_cv2(uuid, image_base64, '')
    box_2d_array, point_2d_array = model_face_detector.infer(bgr_3d_array)
    if box_2d_array.tolist() == []:
        response_json["face_detector"] = False
        return response_json

    response_json["face_detector"] = True
    affine_image = get_affine_image_3d_array(bgr_3d_array, point_2d_array[0])
    rgb_3d_array = affine_image[..., ::-1]  # bgr2rgb
    feats = model_sphereface.infer(Image.fromarray(rgb_3d_array))
    feats = get_feature_2d_array(feats)
    response_json["face_feats"] = feats.tolist()
    response_json["face_base64"] = str(
        base64.b64encode(
            cv2.imencode(
                '.jpg',
                affine_image)[1].tobytes()),
        encoding="utf-8")
    return response_json
