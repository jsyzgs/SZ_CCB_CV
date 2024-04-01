import cv2
import numpy as np
import tritonclient.http as httpclient
from math import ceil
from itertools import product as product

from model import triton_http_client

model_input_hw = (1280, 1280)


class FaceDetector():
    def __init__(self, model_name,
                 model_version,
                 input_name,
                 input_shape,
                 *output_name,
                 input_date_type='FP32'):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_name = output_name
        self.input_date_type = input_date_type

    def preprocess(self, bgr_3d_array):
        image = np.array(bgr_3d_array, np.float32)
        im_height, im_width, _ = np.shape(image)
        self.scale = [
            np.shape(image)[1],
            np.shape(image)[0],
            np.shape(image)[1],
            np.shape(image)[0]]

        self.scale_for_landmarks = []
        for i in range(5):
            self.scale_for_landmarks.append(np.shape(image)[1])
            self.scale_for_landmarks.append(np.shape(image)[0])

        image = letterbox_image(
            image, [self.input_shape[2], self.input_shape[1]])
        image = normalize_input(image)
        tensor = np.expand_dims(image, 0)
        return tensor

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = []
        # 模型是多输出!!!
        for i, ele in enumerate(self.output_name):
            result.append(response.as_numpy(self.output_name[i]))
        return result

    def postprocess(self, excute_result, bgr_3d_array):
        im_height, im_width, _ = np.shape(bgr_3d_array)
        results = detection_out(
            excute_result,
            Anchors(
                image_size=(
                    self.input_shape[1], self.input_shape[2])).get_anchors(),
            confidence_threshold=0.5)
        if len(results) <= 0:
            return np.array([]), np.array([])

        results = np.array(results)
        results = retinaface_correct_boxes(results, np.array(
            [self.input_shape[1], self.input_shape[2]]), np.array([im_height, im_width]))
        results[:, :4] = results[:, :4] * self.scale
        results[:, 5:] = results[:, 5:] * self.scale_for_landmarks

        box_2d_list = []
        point_2d_list = []
        for b in results:
            b = list(map(int, b))
            box_2d_list.append(b[0:4])
            point_2d_list.append(b[5:])

        box_2d_array = np.array(box_2d_list)
        point_2d_array = np.array(point_2d_list)
        box_2d_array = box_2d_array.astype('int')
        point_2d_array = point_2d_array.astype('int')
        point_2d_array = np.concatenate(
            (point_2d_array[:, 0:10:2], point_2d_array[:, 1:10:2]), axis=1)
        return box_2d_array, point_2d_array

    def infer(self,bgr_3d_array):
        tensor = self.preprocess(bgr_3d_array)
        result = self.excute(tensor)
        box_2d_array, point_2d_array = self.postprocess(result,bgr_3d_array)
        return box_2d_array, point_2d_array




def letterbox_image(image, size):
    ih, iw, _ = np.shape(image)
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = cv2.resize(image, (nw, nh))
    new_image = np.ones([size[1], size[0], 3], dtype=np.float32) * 128
    new_image[(h - nh) // 2:nh + (h - nh) // 2,
              (w - nw) // 2:nw + (w - nw) // 2] = image
    return new_image


def normalize_input(bgr_3d_array):
    mean = [103.939, 116.779, 123.68]
    bgr_3d_array[..., 0] -= mean[0]
    bgr_3d_array[..., 1] -= mean[1]
    bgr_3d_array[..., 2] -= mean[2]
    return bgr_3d_array


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
        np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou


def detection_out(
        predictions,
        mbox_priorbox,
        confidence_threshold=0.4,
        nms_thresh=0.45):
    # ---------------------------------------------------#
    #   mbox_loc是回归预测结果
    # ---------------------------------------------------#
    mbox_loc = predictions[0][0]
    # ---------------------------------------------------#
    #   mbox_conf是人脸种类预测结果
    # ---------------------------------------------------#
    mbox_conf = predictions[1][0][:, 1:2]
    # ---------------------------------------------------#
    #   mbox_ldm是人脸关键点预测结果
    # ---------------------------------------------------#
    mbox_ldm = predictions[2][0]

    # --------------------------------------------------------------------------------------------#
    #   decode_bbox
    #   num_anchors, 4 + 10 (4代表预测框的左上角右下角，10代表人脸关键点的坐标)
    # --------------------------------------------------------------------------------------------#
    decode_bbox = decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox)

    # ---------------------------------------------------#
    #   conf_mask    num_anchors, 哪些先验框包含人脸
    # ---------------------------------------------------#
    conf_mask = (mbox_conf >= confidence_threshold)[:, 0]

    # ---------------------------------------------------#
    #   将预测框左上角右下角，置信度，人脸关键点堆叠起来
    # ---------------------------------------------------#
    detection = np.concatenate(
        (decode_bbox[conf_mask][:, :4], mbox_conf[conf_mask], decode_bbox[conf_mask][:, 4:]), -1)

    best_box = []
    scores = detection[:, 4]
    # ---------------------------------------------------#
    #   根据得分对该种类进行从大到小排序。
    # ---------------------------------------------------#
    arg_sort = np.argsort(scores)[::-1]
    detection = detection[arg_sort]
    while np.shape(detection)[0] > 0:
        # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
        best_box.append(detection[0])
        if len(detection) == 1:
            break
        ious = iou(best_box[-1], detection[1:])
        detection = detection[1:][ious < nms_thresh]
    return best_box


def decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox):
    # 获得先验框的宽与高
    prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
    prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
    # 获得先验框的中心点
    prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
    prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

    # 真实框距离先验框中心的xy轴偏移情况
    decode_bbox_center_x = mbox_loc[:, 0] * prior_width * 0.1
    decode_bbox_center_x += prior_center_x
    decode_bbox_center_y = mbox_loc[:, 1] * prior_height * 0.1
    decode_bbox_center_y += prior_center_y

    # 真实框的宽与高的求取
    decode_bbox_width = np.exp(mbox_loc[:, 2] * 0.2)
    decode_bbox_width *= prior_width
    decode_bbox_height = np.exp(mbox_loc[:, 3] * 0.2)
    decode_bbox_height *= prior_height

    # 获取真实框的左上角与右下角
    decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
    decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
    decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
    decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

    prior_width = np.expand_dims(prior_width, -1)
    prior_height = np.expand_dims(prior_height, -1)
    prior_center_x = np.expand_dims(prior_center_x, -1)
    prior_center_y = np.expand_dims(prior_center_y, -1)

    # 对先验框的中心进行调整获得五个人脸关键点
    mbox_ldm = mbox_ldm.reshape([-1, 5, 2])
    decode_ldm = np.zeros_like(mbox_ldm)
    decode_ldm[:, :, 0] = np.repeat(
        prior_width, 5, axis=-1) * mbox_ldm[:, :, 0] * 0.1 + np.repeat(prior_center_x, 5, axis=-1)
    decode_ldm[:, :, 1] = np.repeat(
        prior_height, 5, axis=-1) * mbox_ldm[:, :, 1] * 0.1 + np.repeat(prior_center_y, 5, axis=-1)

    # 真实框的左上角与右下角进行堆叠
    decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                  decode_bbox_ymin[:, None],
                                  decode_bbox_xmax[:, None],
                                  decode_bbox_ymax[:, None],
                                  np.reshape(decode_ldm, [-1, 10])), axis=-1)
    # 防止超出0与1
    decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
    return decode_bbox


def retinaface_correct_boxes(result, input_shape, image_shape):
    new_shape = image_shape * np.min(input_shape / image_shape)

    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    scale_for_boxs = [scale[1], scale[0], scale[1], scale[0]]
    scale_for_landmarks = []
    for i in range(5):
        scale_for_landmarks.append(scale[1])
        scale_for_landmarks.append(scale[0])

    offset_for_boxs = [offset[1], offset[0], offset[1], offset[0]]
    offset_for_landmarks = []
    for i in range(5):
        offset_for_landmarks.append(offset[1])
        offset_for_landmarks.append(offset[0])

    result[:, :4] = (result[:, :4] - np.array(offset_for_boxs)
                     ) * np.array(scale_for_boxs)
    result[:, 5:] = (result[:, 5:] - np.array(offset_for_landmarks)
                     ) * np.array(scale_for_landmarks)

    return result


class Anchors():
    def __init__(self, image_size=None):
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        # ---------------------------#
        #   图片的尺寸
        # ---------------------------#
        self.image_size = image_size
        # ---------------------------#
        #   三个有效特征层高和宽
        # ---------------------------#
        self.feature_maps = [[ceil(self.image_size[0] / step),
                              ceil(self.image_size[1] / step)] for step in self.steps]

    def get_anchors(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            # -----------------------------------------#
            #   对特征层的高和宽进行循环迭代
            # -----------------------------------------#
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1]
                                for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0]
                                for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        anchors = np.reshape(anchors, [-1, 4])

        output = np.zeros_like(anchors[:, :4])
        # -----------------------------------------#
        #   将先验框的形式转换成左上角右下角的形式
        # -----------------------------------------#
        output[:, 0] = anchors[:, 0] - anchors[:, 2] / 2
        output[:, 1] = anchors[:, 1] - anchors[:, 3] / 2
        output[:, 2] = anchors[:, 0] + anchors[:, 2] / 2
        output[:, 3] = anchors[:, 1] + anchors[:, 3] / 2

        if self.clip:
            output = np.clip(output, 0, 1)
        return output


''' 23.12 该版本 加载的onnx模型[-1,-1,-1,3]无需config.pbtxt，httpclient.InferInput()可指定[batch,height,width,channel]'''
model_face_detector = FaceDetector(
    'retinaface_onnx',
    '1',
    'input_1:0',
    [1, model_input_hw[0], model_input_hw[1], 3],
    'output_1:0',
    'output_2:0',
    'output_3:0')
if __name__ == '__main__':
    bgr_3d_array = cv2.imread('/hostmount/errorPicture/timg.jpg')
    # result = model_face_detector.preprocess(bgr_3d_array)
    # result = model_face_detector.excute(result)
    box_2d_array, point_2d_array = model_face_detector.infer(bgr_3d_array)

    ...
