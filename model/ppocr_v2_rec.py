import cv2
import math
import numpy as np
import tritonclient.http as httpclient

from model import triton_http_client
from model.ocr_rec_postprocess import rec_postprocess

'''动态batchsize '''


class OcrRec():

    def __init__(self, model_name,
                 model_version,
                 input_name,
                 input_shape,
                 output_name,
                 input_date_type='FP32'):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_name = output_name
        self.input_date_type = input_date_type

    def preprocess(self, bgr_3d_array_list):
        resized_image_list = []
        # 动态batch input_shape [C,H,W]
        imgC, imgH, imgW = 3, self.input_shape[1], self.input_shape[2]
        max_wh_ratio = imgW / imgH
        for img in bgr_3d_array_list:
            h, w = img.shape[0:2]
            wh_ratio = w * 1.0 / h
            max_wh_ratio = max(max_wh_ratio, wh_ratio)
            padding_img = resize_norm_img(
                img,
                max_wh_ratio,
                self.input_shape[1],
                self.input_shape[2])
            resized_image_list.append(padding_img)
        return np.array(resized_image_list)

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_shape([tensor.shape[0]] + self.input_shape)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = response.as_numpy(self.output_name)
        return result

    def postprocess(self, excute_result):
        return rec_postprocess(excute_result)

    def infer(self, bgr_3d_array_list):
        tensor = self.preprocess(bgr_3d_array_list)
        result = self.excute(tensor)
        result = self.postprocess(result)
        return result


def resize_norm_img(img, max_wh_ratio, *model_info):
    imgC, imgH, imgW = 3, *model_info
    assert imgC == img.shape[2]
    # imgW = int((imgH * max_wh_ratio))
    h, w = img.shape[:2]
    ratio = w / float(h)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


model_ppocr_v2_rec = OcrRec(
    'new_ppocr_rec_v3_onnx', '1', 'x', [
        3, 48, 320], 'softmax_5.tmp_0')
model_ppocr_v2_rec_slim = OcrRec(
    'ppocr_rec_v2_slim_onnx', '1', 'x', [
        3, 32, 100], 'save_infer_model/scale_0.tmp_1')
if __name__ == '__main__':
    cv_img_list = []
    for i in range(3):
        cv_img = cv2.imread('/hostmount/errorPicture/det_4.jpg')
        cv_img_list.append(cv_img)
    result = model_ppocr_v2_rec.infer(cv_img_list)
    # tensor = model_ppocr_v2_rec.preprocess(cv_img_list)
    # model_ppocr_v2_rec.excute(tensor)
