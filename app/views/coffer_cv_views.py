import traceback
from flask import request,jsonify
from uuid import uuid4

from app.views import coffer_blueprint
from app.coffer_cv.service.check_quota_pack_service import check_quota_pack_service
from utils.response_info import RetCode,RetMsg
from utils.log import flask_log

@coffer_blueprint.route('post_route', methods=['POST'])
def send_json():
    try:
        if request.method=='POST':
            request_json = request.json
            imagePath = request_json["imagePath"]
            json_data = {'imagePath':imagePath}
            return jsonify(retCode='00',retMsg='成功',data=json_data)
    except Exception as e:
        flask_log.error(e)
        return jsonify(retCode='99', retMsg='失败')


'''配款封包钱捆暂存核验'''
@coffer_blueprint.route('check_quota_pack',methods=['POST'])
def check_quota_pack_pack():
    uuid = uuid4().hex
    try:
        if request.method=='POST':
            req_json_data = request.json
            image_base64 = req_json_data.get("imageBase64", "")
            task_code = req_json_data.get("taskCode", "")
            response_json_data = check_quota_pack_service(uuid,image_base64,task_code)
            return jsonify(retCode=RetCode.SUCCEED.value,
                           retMsg=RetMsg.SUCCEED.value, taskCode=task_code, data=response_json_data)
    except Exception:
        flask_log.error('uuid:{},{}'.format(uuid,traceback.format_exc()))
        return jsonify(retCode=RetCode.FAILED.value, retMsg=RetMsg.FAILED.value)




