from app.coffer_cv.bussiness.retrieve_face_business import retrieve_face_business


def retrieve_face_service(uuid, image_base64):
    response_json = retrieve_face_business(uuid, image_base64)
    return response_json
