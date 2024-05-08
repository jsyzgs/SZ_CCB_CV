from utils.image_utils import open_image_by_cv2
from model.face_sys import model_face


def retrieve_face_business(uuid, image_base64):
    response_dict = {'face': [], 'face_num': None}
    bgr_3d_array = open_image_by_cv2(uuid, image_base64, '')
    box_2d_array, point_2d_array = model_face.model_det.infer(bgr_3d_array)
    if box_2d_array.shape[0] == 0:
        response_dict['face_num'] = 0

    else:
        response_dict['face_num'] = box_2d_array.shape[0]
        person, score = model_face.get_name_from_database(bgr_3d_array)
        person_dict = {'recognizerName': person, 'recognizerScore': score}
        response_dict['face'].append(person_dict)
    return response_dict
