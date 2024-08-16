from utils.image_utils import open_image_by_cv2
from model.yolov5 import model_vm

def verify_cash_bussiness(uuid,image_base64):
    json_data = {'isCoin':None,'denomation':None,'condition':None,'cashTypes':None,'bundles':None}
    bgr_3d_array = open_image_by_cv2(image_base64)


    ...