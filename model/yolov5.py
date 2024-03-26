import cv2
import numpy as np
import sys
import tritonclient.http as httpclient
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.utils import InferenceServerException, triton_to_np_dtype
from PIL import Image
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc


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


if __name__ == '__main__':
    def resize_img(img,*model_info):
        c,h,w = 3,*model_info
        print(h,w)
    resize_img(1,66,666)


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
