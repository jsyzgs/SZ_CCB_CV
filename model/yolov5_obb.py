import cv2
import numpy as np
from shapely.geometry import Polygon
from copy import deepcopy
import tritonclient.http as httpclient

from model import triton_http_client
from config import Config


class Detection():

    def __init__(
            self,
            model_name,
            model_version,
            input_name,
            input_shape,
            output_name,
            input_date_type='FP32', classes=Config.OBB_LABELS):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_name = output_name
        self.input_date_type = input_date_type
        self.names = {i: ele for i, ele in enumerate(classes)}

    def preprocess(self, bgr_3d_array):
        im = letterbox(
            bgr_3d_array, [
                1024, 1024], stride=32, scaleFill=False, auto=False)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = im.astype(np.float32)
        im = im / 255

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        return im

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = response.as_numpy(self.output_name)
        return result

    def postprocess(self, excute_result, tensor, bgr_3d_array):
        pred = non_max_suppression_obb(
            prediction=excute_result,
            iou_thres=0.2,
            multi_label=False,
            max_det=1000)
        result_list = []
        return_dict = {}
        for i, det in enumerate(pred):
            if det.shape[0] != 0:
                det = det[None]
                pred_poly = det[:, :-3]  # rbox2poly(det[:, :5])
                if len(det):
                    pred_poly = scale_polys(tensor.shape[2:],
                                            pred_poly,
                                            bgr_3d_array.shape)
                    # x1y1x2y2x3y3x4y4 conf cls
                    det = np.concatenate((pred_poly, det[:, -2:]), axis=1)

                    for *poly, conf, cls in reversed(det):
                        c = int(cls)
                        result_list.append({'score': conf,
                                            'locations': np.array([(poly[0], poly[1]),
                                                                   (poly[2], poly[3]),
                                                                   (poly[4], poly[5]),
                                                                   (poly[6], poly[7])], np.int32),
                                            'name': self.names[c]})

        return_dict["results"] = result_list
        return return_dict

    def infer(self, bgr_3d_array):
        tensor = self.preprocess(bgr_3d_array)
        result = self.excute(tensor)
        result = self.postprocess(result, tensor, bgr_3d_array)
        return result


def rbox2poly(obboxes):
    """
        Trans rbox format to poly format.
        Args:
            rboxes (array/tensor): (num_gts, [cx cy l s θ]) θ∈[-pi/2, pi/2)

        Returns:
            polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
        """
    center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    Cos, Sin = np.cos(theta), np.sin(theta)

    vector1 = np.concatenate(
        [w / 2 * Cos, -w / 2 * Sin], axis=-1)
    vector2 = np.concatenate(
        [-h / 2 * Sin, -h / 2 * Cos], axis=-1)

    point1 = center + vector1 + vector2
    point2 = center + vector1 - vector2
    point3 = center - vector1 - vector2
    point4 = center - vector1 + vector2
    order = obboxes.shape[:-1]
    return np.concatenate(
        [point1, point2, point3, point4], axis=-1).reshape(*order, 8)


def scale_polys(img1_shape, polys, img0_shape, ratio_pad=None):
    # ratio_pad: [(h_raw, w_raw), (hw_ratios, wh_paddings)]
    # Rescale coords (xyxyxyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] /
            img0_shape[0],
            img1_shape[1] /
            img0_shape[1])  # gain  = resized / raw
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]  # h_ratios
        pad = ratio_pad[1]  # wh_paddings

    polys[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    polys[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    polys[:, :8] /= gain  # Rescale poly shape to img0_shape
    # clip_polys(polys, img0_shape)
    return polys


def letterbox(
        im,
        new_shape=(
            1280,
            1280),
    color=(
            114,
            114,
            114),
        auto=False,
        scaleFill=False,
        scaleup=True,
        stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios
    else:
        ratio = 0  # TODO:比例待定，暂时没有用到

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color)  # add border
    return im, ratio, (dw, dh)


def non_max_suppression_obb(prediction,
                            conf_thres=0.25,
                            iou_thres=0.45,
                            classes=None,
                            agnostic=False,
                            multi_label=False,
                            labels=(),
                            max_det=1500):
    # prediction = prediction[0]
    nc = prediction.shape[2] - 5 - 180  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    class_index = nc + 5

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and
    # height
    max_wh = 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()

    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    output = [np.zeros((0, 7))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]  # (n_conf_thres, [cx cy l s obj num_cls theta_cls])
        # If none remain process next image
        if not x.shape[0]:
            continue

            # Compute conf
        x[:, 5:class_index] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        theta_pred = np.argmax(x[:, class_index:], axis=1)
        theta_pred = np.expand_dims(theta_pred, axis=1)
        # [n_conf_thres, 1] θ ∈ [-pi/2, pi/2)
        theta_pred = (theta_pred - 90) / 180 * np.pi
        if multi_label:
            i, j = np.array(np.nonzero(x[:, 5:class_index] > conf_thres))
            x = np.concatenate((x[i, :4], theta_pred[i], x[i, j +
                                                           5, None], np.array(j[:, None], dtype=np.float32)), axis=1)
        else:  # best class only
            # print('Singal')
            conf_s = np.max(x[:, 5:class_index], axis=1)
            conf_singal = np.expand_dims(conf_s, axis=1)
            j = np.argmax(x[:, 5:class_index], axis=1)
            j = np.expand_dims(j, axis=1)
            x = np.concatenate((x[:, :4], theta_pred, conf_singal, j), axis=1)[
                conf_s > conf_thres]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # temp_in = x[:,5].argsort()[::-1]
            x = x[x[:, 5].argsort()[::-1]][:max_nms]

        # Batched NMS
        # c = x[:, 6:7] * (0 if agnostic else max_wh)  # classes
        rboxes = deepcopy(x[:, :5])
        scores = deepcopy(x[:, 5])  # scores 必须深拷贝 用作merge_score
        polys = rbox2poly(rboxes)
        # tensor:(x1y1x2y2x3y3x4y4 merge_score score cls)
        polys = np.concatenate(
            (polys, scores.reshape(-1, 1), x[:, -2:]), axis=1)
        # i = nms_locality(polys,0.2)
        # if i.shape[0] > max_det:  # limit detections
        #     i = i[:max_det]
        # output[xi] = x[i]
        result = nms_locality(polys, 0.2)
        output = result

    return output


def nms_locality(polys, thres=0.3):
    '''
    locality aware nms of EAST
    :param polys: a N*9 numpy array. first 8 coordinates, then prob
    :return: boxes after nms
    '''
    S = []  # 合并后的几何体集合
    p = None  # 合并后的几何体
    for g in polys:
        if p is not None and intersection(
                g, p) > thres:  # 若两个几何体的相交面积大于指定的阈值，则进行合并
            p = weighted_merge(g, p)
        else:  # 反之，则保留当前的几何体
            if p is not None:
                S.append(p)
            p = g
    if p is not None:
        S.append(p)
    if len(S) == 0:
        return np.array([])
    return standard_nms(np.array(S), thres)


'''
局部感知NMS
'''

def weighted_merge(g, p):
    # 取g,p两个几何体的加权（权重根据对应的检测得分计算得到）
    g[:8] = (g[8] * g[:8] + p[8] * p[:8]) / (g[8] + p[8])

    # 合并后的几何体的得分为两个几何体得分的总和
    g[8] = (g[8] + p[8])
    return g


def standard_nms(S, thres):
    # 标准NMS
    order = np.argsort(S[:, 8])[::-1]  # 根据score降序排列
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = np.array([intersection(S[i], S[t]) for t in order[1:]])
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]

    return S[keep]


def intersection(g, p):
    # 取g,p中的几何体信息组成多边形
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))

    # 判断g,p是否为有效的多边形几何体
    if not g.is_valid or not p.is_valid:
        return 0

    # 取两个几何体的交集和并集
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


model_yolov5l_obb = Detection(
    'yolov5l_obb_onnx', '1', 'images', [
        1, 3, 1024, 1024], 'output')

if __name__ == '__main__':
    # model = Detection(
    #     'yolov5l_obb_trt', '1', 'images', [
    #         1, 3, 1024, 1024], 'output')

    bgr_3d_array = cv2.imread(
        '/hostmount/errorPicture/orderMat_322002700100202306010072_paper3.jpg')
    result = model_yolov5l_obb.infer(bgr_3d_array)
