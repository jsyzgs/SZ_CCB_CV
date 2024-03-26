from PIL import Image, ImageFont, ImageDraw
import cv2
import base64
import io
from io import BytesIO
import numpy as np
import traceback

from utils.log import flask_log
from config import Config

colors = [(0, 255, 255), (240, 255, 0), (65, 105, 225), (60, 255, 0), (255, 255, 255), (150, 255, 0), (0, 255, 30),
          (0, 255, 120), (0, 255, 209), (0, 209, 255), (0, 120, 255), (0, 29, 255), (60, 0, 255), (149, 0, 255),
          (240, 0, 255), (255, 0, 180), (255, 0, 90)]


def open_image_by_cv2(uuid, imageBase64, imagePath):
    try:
        if imageBase64 == imagePath == "" or imagePath == imageBase64 is None:
            flask_log.error("uuid: %s 输入的imagePath与imageBase64均为空" % uuid)
            image = None
        elif imagePath != "" and imageBase64 == "":
            image = cv2.imread(imagePath)
        elif imagePath == "" and imageBase64 != "":
            img_data = base64.b64decode(imageBase64)
            nparr = np.fromstring(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            flask_log.warning(
                "uuid: %s imagePath和imageBase64传递一个值即可，若传递两个参数则默认选择imagePath" %
                uuid)
            image = cv2.imread(imagePath)
        return image
    except Exception:
        flask_log.error("uuid: %s 读取图片失败：%s" % (uuid, traceback.format_exc()))
        return None


def open_image_by_pil(uuid, imageBase64, imagePath):
    try:
        if imageBase64 == imagePath == '' or imagePath == imageBase64 is None:
            flask_log.error('uuid: %s 输入的imagePath与imageBase64均为空' % uuid)
            image = None
        elif imagePath != "" and imageBase64 == "":
            image = Image.open(imagePath)
        elif imagePath == '' and imageBase64 != '':
            img_data = base64.b64decode(imageBase64)
            image = io.BytesIO(img_data)
            image = Image.open(image)
        else:
            flask_log.warning(
                "uuid: %s imagePath和imageBase64传一个值即可，若传递两个参数则默认选择imagePath" %
                uuid)
            image = Image.open(imagePath)

        return image

    except Exception:
        flask_log.error(
            "uuid: {} 读取图片失败：{}".format(
                uuid, traceback.format_exc()))
        return None


def oob_put_storage_draw_result(
        bgr_3d_array,
        data_result,
        uuid,
        model_image_size=200):

    if data_result != []:
        label_list = []
        for data in data_result:
            predicted_class = data['name']
            if predicted_class not in label_list:
                label_list.append(predicted_class)
            poly = data['locations']
            cv2.drawContours(image=bgr_3d_array,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[label_list.index(predicted_class)],
                             thickness=3)

        image = Image.fromarray(bgr_3d_array[..., ::-1])  # bgr to rgb
        # thickness = image.height // model_image_size

        word_size = np.floor(2.4e-2 * image.height + 0.5).astype('int32')
        font = ImageFont.truetype(font=Config.FONT_PATH, size=word_size)
        draw = ImageDraw.Draw(image)
        for data in data_result:
            left = data['locations'][0][0]
            top = data['locations'][0][1]
            # points = data['locations'].flatten().tolist()
            predicted_class = data['name']
            # if predicted_class not in label_list:
            #     label_list.append(predicted_class)

            # for i in range(thickness): # draw.polygon没有参数调节边框粗细！
            #     draw.polygon(points,outline=colors[label_list.index(predicted_class)])
            label = '{}'.format(predicted_class)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=colors[label_list.index(predicted_class)])
            draw.text(
                text_origin, str(
                    label, 'UTF-8'), fill='black', font=font)
        del draw
    else:
        image = Image.fromarray(bgr_3d_array[..., ::-1])

    # 返回base64
    img_buffer = BytesIO()
    image.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str
