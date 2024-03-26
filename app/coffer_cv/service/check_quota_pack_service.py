from app.coffer_cv.bussiness.check_quota_pack_business import check_quota_pack_business


def check_quota_pack_service(uuid, image_base64, task_code):
    response_json_data = check_quota_pack_business(
        uuid, image_base64, task_code)
    return response_json_data
