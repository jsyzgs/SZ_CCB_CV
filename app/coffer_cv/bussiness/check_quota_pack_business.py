import cv2
import numpy as np
from copy import deepcopy
from PIL import Image

from utils.image_utils import open_image_by_cv2
from model.yolov5_obb import model_yolov5l_obb
from model.mobilenetv2 import model_mobilenetv2_verify
from utils.image_utils import oob_put_storage_draw_result
from config import Config
from model.ppocr_sys import model_ppocr_v3_sys


# 部分误检框置信度过高，该逻辑不通用
def get_yolov5_obb_result_with_thread(result, conf_thread=0.3):
    copy_result = deepcopy(result)
    results = []
    for i, ele in enumerate(copy_result['results']):
        if ele['score'] >= conf_thread:
            results.append(ele)
    result['results'] = results

    return result


def get_yolov5_obb_result_with_area(result):
    results = []
    area_hyper_parameter = 10000
    coin_area_hyper_parameter = 50000
    for i, element in enumerate(result['results']):
        ele = element['locations']
        if ('s_' in element['name'] or 'b_' in element['name']):
            ele_width = int(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(
                                ele[0],
                                ele[1])))))
            ele_height = int(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(
                                ele[1],
                                ele[2])))))
            if ele_width * ele_height > area_hyper_parameter:
                results.append(element)

        elif ('bundle' in element['name'] or 'coin_' in element['name']):
            ele_width = int(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(
                                ele[0],
                                ele[1])))))
            ele_height = int(
                np.sqrt(
                    np.sum(
                        np.square(
                            np.subtract(
                                ele[1],
                                ele[2])))))
            if ele_width * ele_height > coin_area_hyper_parameter:
                results.append(element)

        else:
            results.append(element)

    result['results'] = results
    return result


def get_ploy_image(bgr_3d_array, ori_points):
    '''
    points:[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    '''
    mask = np.zeros(bgr_3d_array.shape[:2], np.uint8)
    points = np.array([ori_points])
    # 在mask上将多边形区域填充为白色
    cv2.polylines(mask, points, 1, 255)  # 描绘边缘
    cv2.fillPoly(mask, points, 255)  # 填充
    # 逐位与，得到裁剪后图像，此时是黑色背景
    dst = cv2.bitwise_and(bgr_3d_array, bgr_3d_array, mask=mask)
    crop_point = ori_points
    min_x = np.min(crop_point[:, 0])
    max_x = np.max(crop_point[:, 0])
    min_y = np.min(crop_point[:, 1])
    max_y = np.max(crop_point[:, 1])
    crop_image = dst[min_y:max_y, min_x:max_x]
    return crop_image


# 修正标签券
def remedy_cash_denomation(detection_result, bgr_3d_array):
    result_list = detection_result['results']
    for i, ele in enumerate(result_list.copy()):
        if "b_" in ele["name"]:
            crop_image = get_ploy_image(bgr_3d_array, ele["locations"])
            # boxes1, recs1 = text_sys_v3(crop_image)
            boxes2, recs2 = model_ppocr_v3_sys(crop_image)
            recs = recs2
            # recs = recs1 + recs2
            cash_label = detect_cash_label(recs)
            result_list[i]["name"] = cash_label

        else:
            continue

    detection_result["results"] = result_list
    return detection_result


def detect_cash_label(recs):
    text = [ele[0] for ele in recs]
    text = ''.join(text)
    # 标签券100元。包含的字有：拾万圆整、人民币壹佰圆券
    if '拾万' in text or ('壹佰圆券' in text and '角' not in text):
        cash_label = 'b_100'
    elif '伍万' in text or '伍拾' in text:
        cash_label = 'b_50'
    elif '贰万' in text or '贰拾' in text:
        cash_label = 'b_20'
    elif '壹万' in text or '壹拾' in text:
        cash_label = 'b_10'
    elif '伍仟' in text or '伍圆' in text or '伍园' in text or '伍任' in text:
        cash_label = 'b_5'
    elif '贰仟' in text or '贰圆' in text or '贰园' in text or '贰任' in text:
        cash_label = 'b_2'
    elif '壹仟' in text or '壹圆' in text or '壹园' in text or '壹任' in text:
        cash_label = 'b_1'
    elif '伍佰' in text or '伍角' in text:
        cash_label = 'b_0.5'
    # 标签券1角。包含的字有：壹佰圆整、人民币壹角券
    # elif ('壹佰' in text and '角' in text) or '壹角' in text:
    elif '壹佰' in text or '壹角' in text or '壹伯' in text:
        cash_label = 'b_0.1'
    else:
        cash_label = ''

    return cash_label


# 识别倾倒的塑封券
def remedy_y_with_classification(detection_result, bgr_3d_array):
    result_list = detection_result['results']
    for i, ele in enumerate(result_list.copy()):
        if ele["name"] == "y":
            crop_image = get_ploy_image(bgr_3d_array, ele["locations"])
            pil_image = Image.fromarray(crop_image[..., ::-1])  # bgr to rgb
            class_name, _ = model_mobilenetv2_verify.infer(pil_image)
            result_list[i]["name"] = class_name
        else:
            continue

    detection_result["results"] = result_list
    return detection_result


# 使用分类模型区分捆装硬币 是壹元还是伍角
def remedy_coin_in_buddle_denomation_with_classification(
        detection_result, bgr_3d_array):
    result_list = detection_result['results']
    class_name = 'coin_0.5'
    for i, ele in enumerate(result_list.copy()):
        if ele["name"] == "bundle":
            # TODO: 等到出现其他捆装硬币时重新训练，目前只有伍角的
            # crop_image = get_ploy_image(bgr_3d_array,ele["locations"])
            # pil_image = Image.fromarray(crop_image[..., ::-1]) #bgr to rgb
            # class_name, _ = sorte_model.run(pil_image)
            result_list[i]["name"] = class_name
        else:
            continue
    detection_result['results'] = result_list
    return detection_result


def check_quota_pack_business(uuid, image_base64,task_code):
    response_dict = {
        "bundles": None,
        "reginzedImageBase64": None,
        "matList": None}
    bgr_3d_array = open_image_by_cv2(uuid, image_base64, imagePath='')
    result = model_yolov5l_obb.infer(bgr_3d_array)
    result = get_yolov5_obb_result_with_thread(result)
    result = get_yolov5_obb_result_with_area(result)
    result = remedy_cash_denomation(result, bgr_3d_array)
    result = remedy_y_with_classification(result, bgr_3d_array)
    result = remedy_coin_in_buddle_denomation_with_classification(
        result, bgr_3d_array)
    response_dict["bundles"] = len(result["results"])
    # Debug
    base64str = oob_put_storage_draw_result(
        bgr_3d_array, result["results"], uuid)
    response_dict["reginzedImageBase64"] = base64str
    mat_list = []
    for ind, ele in enumerate(result["results"]):
        res_dic = {"demonation": None, "isCoin": None}
        res_dic["demonation"] = Config.SECURITIES_TYPE_2[ele["name"]]

        if "coin" in ele["name"]:
            res_dic["isCoin"] = True
        else:
            res_dic["isCoin"] = False
        mat_list.append(res_dic)
    response_dict["matList"] = mat_list
    return response_dict
