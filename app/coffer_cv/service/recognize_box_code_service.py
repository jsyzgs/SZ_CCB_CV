from app.coffer_cv.bussiness.recognize_box_code_business import recognize_box_code_business


def recognize_box_code_service(image_base64, uuid):
    response_data = recognize_box_code_business(uuid, image_base64)
    return response_data
