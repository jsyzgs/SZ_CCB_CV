import cv2
import numpy as np
import sys
import tritonclient.http as httpclient
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
from PIL import Image
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc

from model import triton_http_client
from config import Config


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


def pre_process(bgr_3d_array):
    im = letterbox(
        bgr_3d_array, [
            1280, 1280], stride=32, scaleFill=False, auto=False)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = im.astype(np.float32)
    im = im / 255

    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    return im


def convert_http_metadata_config(_metadata, _config):
    # NOTE: attrdict broken in python 3.10 and not maintained.
    # https://github.com/wallento/wavedrompy/issues/32#issuecomment-1306701776
    try:
        from attrdict import AttrDict
    except ImportError:
        # Monkey patch collections
        import collections
        import collections.abc

        for type_name in collections.abc.__all__:
            setattr(
                collections,
                type_name,
                getattr(
                    collections.abc,
                    type_name))
        from attrdict import AttrDict

    return AttrDict(_metadata), AttrDict(_config)


def parse_model(model_metadata, model_config):
    """
    Check the configuration of a model to make sure it meets the
    requirements for an image classification network (as expected by
    this client)
    """
    if len(model_metadata.inputs) != 1:
        raise Exception("expecting 1 input, got {}".format(
            len(model_metadata.inputs)))
    if len(model_metadata.outputs) != 1:
        raise Exception(
            "expecting 1 output, got {}".format(len(model_metadata.outputs))
        )

    if len(model_config.input) != 1:
        raise Exception(
            "expecting 1 input in model configuration, got {}".format(
                len(model_config.input)
            )
        )

    input_metadata = model_metadata.inputs[0]
    input_config = model_config.input[0]
    output_metadata = model_metadata.outputs[0]

    if output_metadata.datatype != "FP32":
        raise Exception(
            "expecting output datatype to be FP32, model '"
            + model_metadata.name
            + "' output type is "
            + output_metadata.datatype
        )

    # Output is expected to be a vector. But allow any number of
    # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
    # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
    # is one.
    output_batch_dim = model_config.max_batch_size > 0
    non_one_cnt = 0
    for dim in output_metadata.shape:
        if output_batch_dim:
            output_batch_dim = False
        elif dim > 1:
            non_one_cnt += 1
            if non_one_cnt > 1:
                raise Exception("expecting model output to be a vector")

    # Model input must have 3 dims, either CHW or HWC (not counting
    # the batch dimension), either CHW or HWC
    input_batch_dim = model_config.max_batch_size > 0
    expected_input_dims = 3 + (1 if input_batch_dim else 0)
    if len(input_metadata.shape) != expected_input_dims:
        raise Exception(
            "expecting input to have {} dimensions, model '{}' input has {}".format(
                expected_input_dims, model_metadata.name, len(
                    input_metadata.shape)))

    if isinstance(input_config.format, str):
        FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
        input_config.format = FORMAT_ENUM_TO_INT[input_config.format]

    if (input_config.format != mc.ModelInput.FORMAT_NCHW) and (
        input_config.format != mc.ModelInput.FORMAT_NHWC
    ):
        raise Exception(
            "unexpected input format "
            + mc.ModelInput.Format.Name(input_config.format)
            + ", expecting "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW)
            + " or "
            + mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC)
        )

    if input_config.format == mc.ModelInput.FORMAT_NHWC:
        h = input_metadata.shape[1 if input_batch_dim else 0]
        w = input_metadata.shape[2 if input_batch_dim else 1]
        c = input_metadata.shape[3 if input_batch_dim else 2]
    else:
        c = input_metadata.shape[1 if input_batch_dim else 0]
        h = input_metadata.shape[2 if input_batch_dim else 1]
        w = input_metadata.shape[3 if input_batch_dim else 2]

    return (
        model_config.max_batch_size,
        input_metadata.name,
        output_metadata.name,
        c,
        h,
        w,
        input_config.format,
        input_metadata.datatype,
    )


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where
    # xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM


def numpy_nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    # np.array (faster grouped)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] /
            img0_shape[0],
            img1_shape[1] /
            img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def xyxy2xywh(x):
    # Convert 1x4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where
    # x1y1=top,left, x2y2=bottom,right
    y = np.copy(x)
    y[0] = x[0]  # x1
    y[1] = x[1]  # y1
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


class ObjectDetection():
    def __init__(
            self,
            model_name,
            model_version,
            input_name,
            input_shape,
            output_name,
            input_date_type='FP32', classes=Config.QRCODE_LABELS):
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
                1280, 1280], stride=32, scaleFill=False, auto=False)[0]  # padded resize
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

    def postprocess(self, excute_result, tensor, bgr_3d_array,
                    conf_thres=0.1,
                    iou_thres=0.45,
                    nm=0,
                    max_det=1000,
                    agnostic=True):
        '''
                prediction -> (batch,channel,width,height)
                '''
        prediction = excute_result
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into nms()
        # redundant = True  # require redundant detections
        # merge = False
        mi = 5 + nc  # mask start index
        output = [np.zeros((0, 6 + nm))] * bs

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] =
            # 0  # width-height
            x = x[xc[xi]]  # confidence

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            # Box/Mask
            # center_x, center_y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            mask = x[:, mi:]  # zero columns if no masks
            temp = x[:, 5:mi]
            conf = temp.max(1)
            conf = np.expand_dims(conf, axis=1)
            j = np.argmax(temp, axis=1)
            j = np.expand_dims(j, axis=1)

            if len(box) == 1:
                x = np.concatenate((box, conf, j, mask), axis=1)
            else:
                x = np.concatenate((box, conf, j, mask), axis=1)[
                    conf.squeeze() > conf_thres]
            n = x.shape[0]

            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # sort by confidence
                x = x[np.lexsort(-x.T[4, None])][:max_nms]
            else:
                x = x[np.lexsort(-x.T[4, None])]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = numpy_nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            output[xi] = x[i]

        pred = output
        # Process predictions

        result_list = []
        return_dict = {}
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(
                    tensor.shape[2:], det[:, :4], bgr_3d_array.shape)
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    xywh = xyxy2xywh(xyxy)
                    result_list.append({'score': conf,
                                        'locations': {'left': int(xywh[0]),
                                                      'top': int(xywh[1]),
                                                      'width': int(xywh[2]),
                                                      'height': int(xywh[3])},
                                        'name': self.names[c]})
        return_dict["results"] = result_list
        return return_dict

    def infer(self, bgr_3d_array):
        tensor = self.preprocess(bgr_3d_array)
        excute_result = self.excute(tensor)
        result = self.postprocess(excute_result, tensor, bgr_3d_array)
        return result


model_vm = ObjectDetection(
    'yolov5m_onnx', '1', 'images', [
        1, 3, 1280, 1280], 'output0', classes=Config.LABELS)

model_qrcode = ObjectDetection('yolov5s_onnx', '1', 'images', [
    1, 3, 640, 640], 'output0', classes=Config.QRCODE_LABELS)


if __name__ == '__main__':

    bgr_3d_array = cv2.imread('/hostmount/errorPicture/boxMat_095908.jpg')
    result = model_vm.infer(bgr_3d_array)

    def resize_img(img, *model_info):
        c, h, w = 3, *model_info
        print(h, w)
    resize_img(1, 66, 666)

    triton_client = httpclient.InferenceServerClient(url='10.2.72.189:8000')
    # channel = grpc.insecure_channel('10.2.72.189:8001')
    # grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)
    # grpc_stub.ModelInfer()
    # model_metadata = triton_client.get_model_metadata(
    #     model_name='yolov5m_onnx', model_version='1'
    # )
    # model_config = triton_client.get_model_config(
    #     model_name='yolov5m_onnx', model_version='1')
    # model_metadata, model_config = convert_http_metadata_config(
    #     model_metadata, model_config
    # )

    cv2_image = cv2.imread('/hostmount/errorPicture/boxMat_092713.jpg')
    input_image = pre_process(cv2_image)
    iuput = httpclient.InferInput(
        'images', [1, 3, 1280, 1280], datatype="FP32")
    iuput.set_data_from_numpy(input_image, binary_data=True)
    response = triton_client.infer(model_name="yolov5m_onnx", inputs=[iuput])
    result = response.as_numpy('output0')
    ...
