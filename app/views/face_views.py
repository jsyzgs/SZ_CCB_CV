import base64
import cv2
import traceback
import numpy as np
from flask import request, make_response, jsonify
from uuid import uuid4

from app.views import face_blueprint
from app.coffer_cv.service.retrieve_face_service import retrieve_face_service
from app.coffer_cv.service.extract_face_feats_service import extract_face_feats_service
from utils.response_info import RetCode, RetMsg
from utils.log import flask_log


@face_blueprint.route('retrieve', methods=['POST'])
def face_retrieve():
    uuid = uuid4().hex
    try:
        if request.method == 'POST':
            image_base64 = request.json.get('image')
            response_json_data = retrieve_face_service(uuid, image_base64)
            return jsonify(
                retCode=RetCode.SUCCEED.value,
                retMsg=RetMsg.SUCCEED.value,
                data=response_json_data)

    except Exception:
        flask_log.error('uuid:{},{}'.format(uuid, traceback.format_exc()))
        return jsonify(
            retCode=RetCode.FAILED.value,
            retMsg=RetMsg.FAILED.value)


@face_blueprint.route('facetodb', methods=['POST'])
def extract_face_feats():
    uuid = uuid4().hex
    try:
        if request.method == 'POST':
            image_base64 = request.json.get('image')
            response_json_data = extract_face_feats_service(uuid,image_base64)
            return jsonify(
                retCode=RetCode.SUCCEED.value,
                retMsg=RetMsg.SUCCEED.value,
                data=response_json_data)

    except Exception:
        flask_log.error('uuid:{},{}'.format(uuid, traceback.format_exc()))
        return jsonify(
            retCode=RetCode.FAILED.value,
            retMsg=RetMsg.FAILED.value)


@face_blueprint.route('load_faceinfo_fromdb', methods=['POST'])
def load_faceinfo_fromdb():
    uuid = uuid4().hex
    try:
        if request.method == 'POST':
            ...
    except Exception:
        flask_log.error('uuid:{},{}'.format(uuid, traceback.format_exc()))
        return jsonify(
            retCode=RetCode.FAILED.value,
            retMsg=RetMsg.FAILED.value)
