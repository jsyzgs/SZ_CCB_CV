import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from sklearn import preprocessing

from model import triton_http_client


class FaceRecognition():
    def __init__(self,
                 model_name,
                 model_version,
                 input_name,
                 input_shape,
                 output_name,
                 input_date_type='FP32'):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_shape = input_shape  # (1,3,112,96)
        self.output_name = output_name
        self.input_date_type = input_date_type

    def preprocess(self, pil_image):
        pil_image = pil_image.resize(
            (self.input_shape[3], self.input_shape[2]), Image.BILINEAR)
        bgr_3d_array = np.array(
            pil_image)[..., ::-1].astype('float32')  # rgb2bgr
        tensor = (bgr_3d_array - 127.5) * 0.0078125
        tensor = tensor.transpose([2, 0, 1])
        tensor = np.expand_dims(tensor, 0)
        return tensor

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = response.as_numpy(self.output_name)
        return result

    def postprocess(self, excute_result):

        return excute_result

    def infer(self, pil_image):
        tensor = self.preprocess(pil_image)
        result = self.excute(tensor)
        result = self.postprocess(result)
        return result


def get_feature_2d_array(tensor):
    feature_2d_array = preprocessing.normalize(tensor)
    return feature_2d_array


model_sphereface = FaceRecognition(
    'sphereface_onnx', '1', 'input', [
        1, 3, 112, 96], 'fc5_Gemm_Y')

if __name__ == '__main__':
    import cv2
    from cv2 import dnn
    from copy import deepcopy

    pil_image = Image.open('/hostmount/errorPicture/20190819_163759_6037.jpg')
    new_image = deepcopy(pil_image).resize(
        (96, 112), Image.BILINEAR)
    rgb_3d_array = np.array(new_image)
    bgr_3d_array = rgb_3d_array[..., ::-1] #rgb2bgr
    blob = dnn.blobFromImage(bgr_3d_array, 0.0078125, (96, 112), (127.5, 127.5, 127.5))
    net = dnn.readNetFromCaffe('/hostmount/sphereface_model/sphereface.prototxt',
                               '/hostmount/sphereface_model/sphereface.caffemodel')
    net.setInput(blob)
    result_dnn = net.forward()



    tensor = model_sphereface.preprocess(pil_image)
    result = model_sphereface.excute(tensor)
    result = model_sphereface.postprocess(result)
    ...
