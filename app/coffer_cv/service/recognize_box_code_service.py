from utils.image_utils import open_image_by_cv2


def recognize_box_code_service(image_base64, uuid):
    image = open_image_by_cv2(
        uuid=uuid,
        imageBase64=image_base64,
        imagePath='')
    data = {
        "boxType": None,
        "condition": None,
        "boxNo": None,
        "qrcodeData": None,
        "MD5": None}
    boxNo = ""
    boxData = ""
    condition = False
