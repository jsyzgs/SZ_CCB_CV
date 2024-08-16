from app.coffer_cv.bussiness.verify_cash_bussiness import verify_cash_bussiness

def verify_cash_service(uuid,image_base64):
    response_data = verify_cash_bussiness(uuid,image_base64)
    return response_data