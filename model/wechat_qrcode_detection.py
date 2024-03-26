import cv2

qr_code_detector = cv2.wechat_qrcode_WeChatQRCode()


def derect_qr_code(bgr_3d_array):
    res, points = qr_code_detector.detectAndDecode(bgr_3d_array)
    return res, points


if __name__ == '__main__':
    bgr_3d_array = cv2.imread('/hostmount/errorPicture/boxNo_103015.jpg')
    res,points = derect_qr_code(bgr_3d_array)
    ...
