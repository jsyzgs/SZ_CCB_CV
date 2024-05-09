from app.coffer_cv.bussiness.extract_face_feats_business import extract_face_feats_business


def extract_face_feats_service(uuid, image_base64):
    response_data = extract_face_feats_business(uuid, image_base64)
    return response_data
