import cv2
import copy
import numpy as np

from model.ppocr_v2_det import model_ppocr_v2_det
from model.ppocr_v2_cls import model_ppocr_v2_cls
from model.ppocr_v2_rec import model_ppocr_v2_rec


class OcrSys():

    def __init__(
            self,
            model_det,
            model_cls,
            model_rec,
            cls_thresh=0.7,
            drop_thresh=0.5):
        self.det_model = model_det
        self.cls_model = model_cls
        self.rec_model = model_rec
        self.cls_thresh = cls_thresh
        self.drop_thresh = drop_thresh

    def rotate_img(self, cls_results, img_list):
        for i, cls_res in enumerate(cls_results):
            label, score = cls_results[i]
            if '180' in label and score > self.cls_thresh:
                img_list[i] = cv2.rotate(
                    img_list[i], 1)
        return img_list

    def resize_image_keep_aspect(self, bgr_3d_array):
        h, w = bgr_3d_array.shape[:2]
        if len(self.det_model.input_shape) == 4:
            new_scale = min(
                self.det_model.input_shape[3] / w,
                self.det_model.input_shape[2] / h)
        elif len(self.det_model.input_shape) == 3:
            new_scale = min(
                self.det_model.input_shape[2] / w,
                self.det_model.input_shape[1] / h)
        else:
            assert len(self.det_model.input_shape) == 3 or len(
                self.det_model.input_shape == 4), 'input_shape must be NCHW or CHW'
        nw = int(w * new_scale)
        nh = int(h * new_scale)
        new_bgr_3d_array = cv2.resize(bgr_3d_array, (nw, nh))
        return new_bgr_3d_array, new_scale

    def __call__(self, img, cls=True, scale_flag=True):
        ori_im = img.copy()
        if scale_flag:
            ori_im, scale = self.resize_image_keep_aspect(ori_im)

        dt_boxes = self.det_model.infer(ori_im)
        if dt_boxes is None:
            return None, None

        img_crop_list = []

        ''' 缩放到原图'''
        if scale_flag:
            dt_boxes = recover_point(dt_boxes, scale)

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])

            ''' 基于原图进行抠图，对应于缩放到原图'''
            if scale_flag:
                img_crop = get_rotate_crop_image(img, tmp_box)  # 基于原图进行抠图
            else:
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        if cls:
            cls_result = self.cls_model.infer(img_crop_list)
            img_crop_list = self.rotate_img(cls_result, img_crop_list)

        rec_v3_result = self.rec_model.infer(img_crop_list)
        rec_result = rec_v3_result.copy()

        filter_boxes, filter_rec_res = [], []
        for box, rec_re in zip(dt_boxes, rec_result):
            text, score = rec_re
            if score >= self.drop_thresh:
                filter_boxes.append(box)
                filter_rec_res.append(rec_re)
        return filter_boxes, filter_rec_res


def recover_point(det_box, scale):
    if len(det_box) == 0:
        return np.array([])
    result = []
    for i, ele in enumerate(det_box):
        ele = ele / scale
        ele = ele.astype(np.int32)
        result.append(ele.astype(np.float32))
    return np.array(result)


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
    assert len(points) == 4, "shape of points must be 4*2"
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

model_ppocr_v3_sys = OcrSys(model_ppocr_v2_det,model_ppocr_v2_cls,model_ppocr_v2_rec)
if __name__ == '__main__':
    cv_img = cv2.imread('/hostmount/errorPicture/rh1.jpg')
    model_ppocr_v3_sys(cv_img)
