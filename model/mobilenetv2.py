import numbers
import numpy as np
from PIL import Image
import tritonclient.http as httpclient

from config import Config
from model import triton_http_client


def letterbox_image(image, size, letterbox_image):
    w, h = size
    iw, ih = image.size
    if letterbox_image:
        '''resize image with unchanged aspect ratio using padding'''
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        if h == w:
            new_image = resize(image, h)
        else:
            new_image = resize(image, [h, w])
        new_image = center_crop(new_image, [h, w])
    return new_image


def resize(img, size, interpolation=Image.BILINEAR):
    r"""Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)


def center_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return crop(img, i, j, th, tw)


def crop(img, i, j, h, w):
    """Crop the given PIL Image.

    Args:
        img (PIL Image): Image to be cropped.
        i (int): i in (i,j) i.e coordinates of the upper left corner.
        j (int): j in (i,j) i.e coordinates of the upper left corner.
        h (int): Height of the cropped image.
        w (int): Width of the cropped image.

    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


def preprocess_input(x):
    x /= 255
    x -= np.array([0.485, 0.456, 0.406])
    x /= np.array([0.229, 0.224, 0.225])
    return x


def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


class Classification():
    def __init__(self, model_name,
                 model_version,
                 input_name,
                 input_shape,
                 output_name,
                 input_date_type='FP32', classes=Config.VERIFY_LABELS):
        self.model_name = model_name
        self.model_version = model_version
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_name = output_name
        self.input_date_type = input_date_type
        self.class_names = {i: ele for i, ele in enumerate(classes)}

    def preprocess(self, image):
        image_data = letterbox_image(image, [224, 224], False)
        image_data = np.array(image_data, np.float32)
        image_data = preprocess_input(image_data)
        image_data = np.expand_dims(image_data, 0)
        image_data = np.transpose(image_data, (0, 3, 1, 2))
        return image_data.copy()

    def excute(self, tensor, triton_client=triton_http_client):
        input = httpclient.InferInput(
            self.input_name, self.input_shape, datatype=self.input_date_type)
        input.set_data_from_numpy(tensor, binary_data=True)
        response = triton_client.infer(
            model_name=self.model_name, inputs=[input])
        result = response.as_numpy(self.output_name)
        return result

    def postprocess(self, excute_result):
        preds = softmax(excute_result)
        class_name = self.class_names[np.argmax(preds)]
        probability = np.max(preds)
        return class_name, probability

    def infer(self, image):
        input = self.preprocess(image)
        excute_result = self.excute(input)
        class_name, probability = self.postprocess(excute_result)
        return class_name, probability

model_mobilenetv2_verify = Classification('mobilenetv2_onnx','1','input.1',[1,3,224,224],'535')
if __name__ == '__main__':
    pil_image = Image.open(
        '/hostmount/errorPicture/orderMat_322002700100202309070069_paper5__y__0_Sequential_2_Fliplr_0.jpg')
    print(model_mobilenetv2_verify.infer(pil_image))

