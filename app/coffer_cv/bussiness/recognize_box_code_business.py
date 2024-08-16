import re
import cv2
import numpy as np

from model.ppocr_sys import model_ppocr_v2_sys
from utils.image_utils import open_image_by_cv2
from model.yolov5 import model_qrcode
from config import Config


def get_ploy_image(bgr_3d_array, ori_points):
    '''

    :param bgr_3d_array:
    :param ori_points:  4*2 ndarray
    :return:
    '''
    mask = np.zeros(bgr_3d_array.shape[:2], np.uint8)
    points = np.array([ori_points])  # 要增加维度，不然画图报错！

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


def detect_qr_code_with_opencv(bgr_3d_array):

    edges = cv2.Canny(bgr_3d_array, 100, 256)
    cnts, hiera = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    box = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        # print(area)
        if area <= 50000:  # boxNo area is 121409
            continue
        else:
            rect = cv2.minAreaRect(cnt)
            points = cv2.boxPoints(rect)
            box = np.array(points, np.int32)  # np.int0(points)
            # y0, y1 = box[0][1], box[2][1]
            # x0, x1 = box[0][0], box[2][0]
            # cv2.drawContours(bgr_3d_array, [box], 0, (255, 0, 0), -1)
            # cv2.putText(bgr_3d_array, str(area), (x0, y0),
            #             cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200),
            #             5)
    return box


def tlwh2tlbr(location_dict):
    '''

    :param location_dict: yolov5输出
    :return:
    '''
    left = location_dict['left']
    top = location_dict['top']
    bottom = location_dict['top'] + location_dict['height']
    right = location_dict['left'] + location_dict['width']
    return (left, top, right, bottom)


def crop_image(bgr_3d_array, int_box, gain=15):
    '''

    :param bgr_3d_array:
    :param int_box: (xmin,ymin,xmax,ymax)
    :param gain: 检测框扩大增益
    :return:
    '''
    height, width = bgr_3d_array.shape[:2]
    crop_xmin = max(0, int_box[0] - gain)
    crop_xmax = min(width, int_box[2] + gain)
    crop_ymin = max(0, int_box[1] - gain)
    crop_ymax = min(height, int_box[3] + gain)
    cv_image = bgr_3d_array[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    return cv_image


det = cv2.wechat_qrcode_WeChatQRCode()


def decode_qrcode(bgr_3d_array):
    result = det.detectAndDecode(bgr_3d_array)
    return result


qr = cv2.QRCodeDetector()


def detect_qr_with_cv(bgr_3d_array):

    result = qr.detectAndDecode(bgr_3d_array)
    return result[:2]


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image, pts, gap=10):
    # print(pts)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    tl[0] = tl[0] - gap
    tl[1] = tl[1] - gap

    tr[0] = tr[0] + gap
    tr[1] = tr[1] - gap

    br[0] = br[0] + gap
    br[1] = br[1] + gap

    bl[0] = bl[0] - gap
    bl[1] = bl[1] + gap

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # warped = cv2.copyMakeBorder(warped,50,50,50,50, cv2.BORDER_CONSTANT,value=[255,255,255])
    return warped


def judge_renhang_box(recs, chinese_with_currency):
    '''

    :param recs: OCR输出 [(aa,0.9),(bb,0.8)]
    :param chinese_with_currency: {"侠": "50", "俊": "100"}
    :return:
    '''
    messages = [ele[0] for ele in recs]
    messages = ''.join(messages)
    flag = False
    for i in chinese_with_currency:
        if re.findall(i, messages):
            flag = True
            break
    return flag


def get_Outline(bgr_3d_array):
    gray = cv2.cvtColor(bgr_3d_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    ret, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    edged = cv2.Canny(binary_img, 50, 200)
    return bgr_3d_array, gray, edged


def get_cnt(edged):
    cnts = cv2.findContours(
        edged.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    cnts = cnts[0]
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt


def recognize_box_code_business(uuid, image_base64):
    json_data = {
        'boxData': '',
        'boxNo': None,
        'boxType': None,  # 1代表人行箱；2代表周转箱 二维码
        'condition': None,
        'qrcodeData': None}

    bgr_3d_array = open_image_by_cv2(uuid, image_base64, '')
    result = model_qrcode.infer(bgr_3d_array)
    # ''' 挡板 result['results']= [] ！！！ '''
    # result['results'] = []
    if len(result['results']) == 0:
        # 人行箱 或 二维码漏检
        boxes, recs = model_ppocr_v2_sys(bgr_3d_array)
        if recs is None:
            # 可能是漏检的qrcode图片 使用传统方法检测
            box = detect_qr_code_with_opencv(bgr_3d_array)
            if len(box) != 0:
                ploy_image = get_ploy_image(bgr_3d_array, box)
                qr_result = decode_qrcode(ploy_image)

                if qr_result[0] == '':
                    # OCR 识别二维码下面的字符串
                    boxes, recs = model_ppocr_v2_sys(ploy_image)
                    qr_result = [ele[0] for ele in recs if len(ele[0]) == 6]
                    if len(qr_result) == 0:
                        json_data['condition'] = False
                else:
                    json_data['boxType'] = 2
                    json_data['condition'] = True
                    json_data['boxNo'] = qr_result[0]
                    json_data['qrcodeData'] = qr_result[0]
            else:
                json_data['condition'] = False

        elif judge_renhang_box(recs, Config.CHINESE_WITH_CURRENCY):  # 人行箱子
            messages = [ele[0] for ele in recs]
            messages = ''.join(messages)
            messages = re.sub('[1234567890]', '', messages)  # 去除字符串里的数字
            messages = re.sub('[△*]', '', messages)  # 去除字符串里的特殊字符
            key_in_CHINESE_WITH_CURRENCY = messages[0]
            value_in_CHINESE_WITH_CURRENCY = Config.CHINESE_WITH_CURRENCY[
                key_in_CHINESE_WITH_CURRENCY]
            kind_of_money = Config.MONEY_TO_ALPHABET[value_in_CHINESE_WITH_CURRENCY]
            json_data['boxNo'] = 'P' + kind_of_money
            json_data['boxData'] = messages
        else:
            # 漏检二维码 且 文字检测不全！
            ...

    else:
        # 业务中箱体仅有一个二维码
        json_data['boxType'] = 2

        int_box = tlwh2tlbr(result['results'][0]['locations'])
        crop_cv_image = crop_image(bgr_3d_array, int_box)
        qr_result = decode_qrcode(crop_cv_image)
        print('...')
        if len(qr_result[0]) == 0:
            # 需要获取二维码精确坐标，或直接OCR
            result = detect_qr_with_cv(crop_cv_image)
            points = result[1]
            if points == []:
                # 传统方法检测二维码 或直接OCR
                boxes, recs = model_ppocr_v2_sys(crop_cv_image)
                qr_result = [ele[0] for ele in recs if len(ele[0]) == 6]

            else:
                # 透视变换
                cv_image = four_point_transform(crop_cv_image, points)
                qr_result, _ = decode_qrcode(cv_image)

            if len(qr_result) == 0:
                boxes, recs = model_ppocr_v2_sys(crop_cv_image)
                qr_result = [ele[0] for ele in recs if len(ele[0]) == 6]
                if len(qr_result) == 0:
                    json_data['condition'] = False
        else:
            json_data['condition'] = True
            json_data['boxNo'] = qr_result[0]
            json_data['qrcodeData'] = qr_result[0]

    return json_data


if __name__ == '__main__':

    cv_img = cv2.imread('/hostmount/errorPicture/boxNo_085013.jpg')
    detect_qr_with_cv(cv_img)
    # cv_img = cv2.imread('/hostmount/errorPicture/boxBackNo_In20230612095155756115646.jpg')
    box = detect_qr_code_with_opencv(cv_img)
    get_ploy_image(cv_img, box)

    import base64
    from uuid import uuid4

    def bgr_3d_array_to_base64(bgr_3d_array):
        img = bgr_3d_array
        str_base64 = str(
            base64.b64encode(
                cv2.imencode(
                    '.jpg',
                    img)[1].tobytes()),
            encoding="utf-8")
        return str_base64

    # bgr_3d_rray = cv2.imread('/hostmount/errorPicture/boxNo_085013.jpg')
    bgr_3d_rray = cv2.imread(
        '/hostmount/errorPicture/boxBackNo_In20230612095155756115646.jpg')
    # bgr_3d_rray = cv2.imread('/hostmount/errorPicture/rh1.jpg')
    image_base64 = bgr_3d_array_to_base64(bgr_3d_rray)
    uuid_num = uuid4().hex
    result = recognize_box_code_business(uuid_num, image_base64)
    ...
