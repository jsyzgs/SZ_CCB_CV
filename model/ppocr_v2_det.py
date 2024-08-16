import cv2
import numpy as np
import tritonclient.http as httpclient

from model.ocr_db_preprocess import db_resize_v2_server
from model.ocr_db_postprocess import db_postprocess
from model import triton_http_client


class OcrDetServer():
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

    def preprocess(self, image):
        data = {'image': image}
        data = db_resize_v2_server(data)
        data['image'] = norm_and_padding_img(
            data['image'], self.input_shape[2], self.input_shape[3])
        image_array = data['image']
        if len(image_array.shape) == 3:
            image_array = image_array[None]  # expand for batch dim
        data['image'] = image_array
        return data

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = response.as_numpy(self.output_name)
        return result

    def postprocess(self, excute_result, shape_list):
        preds = {}
        preds['maps'] = excute_result
        fixed_shape_list = np.array(
            [[float(self.input_shape[2]), float(self.input_shape[3]), 1., 1.]])
        # shape_list = np.expand_dims(shape_list, axis=0)
        post_result = db_postprocess(preds, fixed_shape_list)
        dt_boxes = post_result[0]['points']
        # dt_boxes = self.filter_tag_det_res(dt_boxes, image_shape) # 有无该处理 TODO：需要该处理，否则一个单位像素 可能 会引起bug
        # 并不影响检测框
        dt_boxes = restore_points(dt_boxes, shape_list)
        return dt_boxes

    def infer(self, image):
        data = self.preprocess(image)
        input = data['image']
        excute_result = self.excute(input)
        result = self.postprocess(excute_result, data['shape'])
        return result


def norm_and_padding_img(image, model_input_height, model_input_width):
    resized_image = np.array(image).astype(np.float32)
    std_chw_rgb_3d_array = resized_image / 255
    std_chw_rgb_3d_array = std_chw_rgb_3d_array.transpose(2, 0, 1).copy()
    #  RGB mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    std_chw_rgb_3d_array[0, :, :] = (
        std_chw_rgb_3d_array[0, :, :] - float(0.485)) / float(0.229)
    std_chw_rgb_3d_array[1, :, :] = (
        std_chw_rgb_3d_array[1, :, :] - float(0.456)) / float(0.224)
    std_chw_rgb_3d_array[2, :, :] = (
        std_chw_rgb_3d_array[2, :, :] - float(0.406)) / float(0.225)
    resized_image = np.array(std_chw_rgb_3d_array)

    padding_im = np.zeros(
        (3,
         model_input_height,
         model_input_width),
        dtype=np.float32)
    padding_im[:, 0:resized_image.shape[1],
               0:resized_image.shape[2]] = resized_image
    return padding_im


def restore_points(dt_boxes, shape_list):
    height_ratio = shape_list[2]  # 512/(round(ori_shape[0]/32)*32)#
    width_ratio = shape_list[3]  # 960/(round(ori_shape[1]/32)*32)#
    boxes = dt_boxes.reshape(-1, 2)
    boxes[:, 0] = boxes[:, 0] / width_ratio  # height_ratio
    boxes[:, 1] = boxes[:, 1] / height_ratio  # width_ratio
    new_dt_boxes = boxes.reshape(dt_boxes.shape).astype(np.float32)
    return new_dt_boxes


model_ppocr_v2_det = OcrDetServer(
    'ppocr_det_v2_server_onnx', '1', 'x', [
        1, 3, 640, 640], 'sigmoid_0.tmp_0')
model_ppocr_v2_det_slim = OcrDetServer('ppocr_det_v2_onnx', '1', 'x', [
    1, 3, 960, 960], 'save_infer_model/scale_0.tmp_1')
if __name__ == '__main__':
    bgr_3d_array = cv2.imread('/hostmount/errorPicture/nature_test.jpg')
    result = model_ppocr_v2_det.infer(bgr_3d_array)
