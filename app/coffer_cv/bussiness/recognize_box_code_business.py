import re
import cv2

from model.ppocr_sys import model_ppocr_v2_sys


def sort_f(x):
    return x > '9', len(x)


def recognize_box_code_business(image, uuid):
    mes = []


def ocr(img, uuid):
    qrcode_mes = ''
    ocr_result = []
    result = model_ppocr_v2_sys(img)
    pattern1 = (r'\d{2}\D+')  # '*32 俊驭艺晾'
    pattern2 = re.compile(r'\d{2}')
    pattern3 = re.compile(r'\d{4}')
    patternc = re.compile(r'[\u4e00-\u9fa5]+')

    # 图片第二行的4位数字可能带三角形
    for e in result[1]:
        element = e[0]
        m = pattern3.findall(element)
        # assert len(m) <= 1
        if len(m) == 1:
            ocr_result.append(m[0])
        else:
            m = pattern1.findall(element)  # 防止第一个数字和后面汉字有粘连,拆成2份
            # assert len(m) <= 1
            if len(m) == 1:
                string = m[0].replace(" ", "")
                # string_delete = string_empty.replace(".", "")
                # string = string_delete.replace("。", "")
                num = pattern2.match(string)  # 匹配句首2个数字
                if num is not None:
                    ocr_result.append(num.group(0))
                chi = patternc.findall(string)
                # assert len(chi) <= 1
                if len(chi) == 1:
                    ocr_result.append(chi[0])
                else:
                    for i in chi:
                        ocr_result.append(i)
            else:
                ocr_result.append(element)

    # 删除干扰
    for i in ocr_result:
        if i == 'A':
            ocr_result.remove('A')
        elif i == '△':
            ocr_result.remove('△')
        elif i == '1':
            ocr_result.remove('1')

    ocr_result = sorted(ocr_result, key=sort_f, reverse=True)

    if len(ocr_result) != 0:
        flag = False
        # i是单个的匹配面值的字 or i是俊晾榕，包含匹配面值的字
        for full_str in ocr_result:
            for i in list(chinese_to_money_dict.keys()):
                if re.findall(i, full_str):
                    flag = True
                    break
        if flag:
            return ocr_result, ''
        else:
            # 未匹配到面值时，将图片进行二值化处理再识别一遍
            img_binary = binary_img_ocr(img)
            img_encode = cv2.imencode('.jpg', img_binary)[1]
            img_decode = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
            # 二值化图片ocr识别
            result = ocr_det_rec(img_decode)
            pattern1 = re.compile(r'\d{2}\D+')  # '*32 俊驭艺晾'
            pattern2 = re.compile(r'\d{2}')
            pattern3 = re.compile(r'\d{4}')
            patternc = re.compile(r'[\u4e00-\u9fa5]+')
            # 图片第二行的4位数字有可能带三角形
            for e in result[1]:
                element = e[0]
                m = pattern3.findall(element)
                # assert len(m) <= 1
                if len(m) == 1:
                    ocr_result.append(m[0])
                else:
                    m = pattern1.findall(element)  # 防止第一个数字和后面汉字有粘连,拆成2份
                    # assert len(m) <= 1
                    if len(m) == 1:
                        string = m[0].replace(" ", "")
                        # string_delete = string_empty.replace(".", "")
                        # string = string_delete.replace("。", "")
                        num = pattern2.match(string)  # 匹配句首2个数字
                        if num is not None:
                            ocr_result.append(num.group(0))
                        chi = patternc.findall(string)
                        # assert len(chi) <= 1
                        if len(chi) == 1:
                            ocr_result.append(chi[0])
                        else:
                            for i in chi:
                                ocr_result.append(i)
                    else:
                        ocr_result.append(element)
            # 删除干扰
            for i in ocr_result:
                if i == 'A':
                    ocr_result.remove('A')
                elif i == '△':
                    ocr_result.remove('△')
                elif i == '1':
                    ocr_result.remove('1')
            ocr_result = sorted(ocr_result, key=sort_f, reverse=True)
            if len(ocr_result) != 0:
                ocr_flag = False
                # i是单个的匹配面值的字 or i是俊晾榕，包含匹配面值的字
                for full_str in ocr_result:
                    for i in list(chinese_to_money_dict.keys()):
                        if re.findall(i, full_str):
                            ocr_flag = True
                            break
                if ocr_flag:
                    return ocr_result, ''
                else:
                    if True:
                        qr_code_data_list, rects_list = qr_code_detect(img)

                        if len(qr_code_data_list) == 0:
                            logger.info('uuid: {}二维码原图上未识别出二维码信息'.format(uuid))
                            # 检测二维码&裁剪出二维码图片
                            qr_code_img_list = []
                            detect_result = yoloqrcode_sample.run(img)
                            result_list = detect_result['results']
                            if len(result_list) != 0:
                                for ele in result_list:
                                    location_tuple = tlwh2tlbr(
                                        ele['locations'])
                                    left, top, right, bottom = location_tuple
                                    qrcode_image = img[top:bottom, left:right]
                                    qr_code_img_list.append(qrcode_image)

                            # 微信模型识别二维码
                            qrcode_list = []
                            for qr_code_img in qr_code_img_list:
                                qr_code_data_list, rects_list = qr_code_detect(
                                    qr_code_img)
                                # 把二维码抠出来的图检测结果为空，对该图做自适应二值化进行检测
                                if len(qr_code_data_list) == 0:
                                    logger.info(
                                        'uuid: {}二维码抠图上未识别出二维码信息'.format(uuid))
                                    if qr_code_img != []:
                                        qr_code_img_binary, ret = binary_img_qrcode(
                                            qr_code_img, 0)
                                        qr_code_data_list, rects_list = qr_code_detect(
                                            qr_code_img_binary)
                                        # 检测结果为空，最后采用ocr识别二维码图片中的字母、数字
                                        if len(qr_code_data_list) == 0:
                                            # logger.info('uuid: {}二维码二值化未识别出二维码信息'.format(uuid))
                                            # logger.info('uuid: {}进行ocr识别二维码......'.format(uuid))
                                            # ocr识别
                                            boxes, recs = qrcode_ocr_sample(
                                                qr_code_img)
                                            if len(recs) != 0:
                                                # 添加边界条件,如果前两位字母中包含数字8，则替换为字母B
                                                for idx, info in enumerate(
                                                        recs):
                                                    if '8' in info[0][:2]:
                                                        modified_string = recs[idx][0][:2].replace(
                                                            '8', 'B') + recs[idx][0][2:]
                                                        qr_code_data_list = (
                                                            modified_string,)
                                                        qrcode_list.append(
                                                            qr_code_data_list)
                                                    else:
                                                        # recs[0][0]='BC005'
                                                        qr_code_data_list = (
                                                            recs[0][0],)
                                                        qrcode_list.append(
                                                            qr_code_data_list)
                                        else:
                                            qrcode_list.append(
                                                qr_code_data_list)
                                else:
                                    qrcode_list.append(qr_code_data_list)

                            if len(qrcode_list) != 0:
                                qr_code_data_list = qrcode_list[0]

                        if len(qr_code_data_list) > 1:
                            logger.error('uuid: {}识别原封券箱出现了多个二维码'.format(uuid))
                            return None
                        elif len(qr_code_data_list) == 0:
                            logger.warn('uuid:{}识别原封券箱未识别出二维码'.format(uuid))
                        else:
                            qrcode_mes = qr_code_data_list[0]
                            logger.info(
                                'uuid:{}识别原封券箱识别出1个二维码,二维码信息为{}'.format(
                                    uuid, qr_code_data_list))
            elif len(ocr_result) == 0:
                if True:
                    qr_code_data_list, rects_list = qr_code_detect(img)

                    if len(qr_code_data_list) == 0:
                        # logger.info('uuid: {}二维码原图上未识别出二维码信息'.format(uuid))
                        # 检测二维码&裁剪出二维码图片
                        qr_code_img_list = []
                        detect_result = yoloqrcode_sample.run(img)
                        result_list = detect_result['results']
                        if len(result_list) != 0:
                            for ele in result_list:
                                location_tuple = tlwh2tlbr(ele['locations'])
                                left, top, right, bottom = location_tuple
                                qrcode_image = img[top:bottom, left:right]
                                qr_code_img_list.append(qrcode_image)

                        # 微信模型识别二维码
                        qrcode_list = []
                        for qr_code_img in qr_code_img_list:
                            qr_code_data_list, rects_list = qr_code_detect(
                                qr_code_img)
                            # 把二维码抠出来的图检测结果为空，对该图做自适应二值化进行检测
                            if len(qr_code_data_list) == 0:
                                logger.info(
                                    'uuid: {}二维码抠图上未识别出二维码信息'.format(uuid))
                                if qr_code_img != []:
                                    qr_code_img_binary, ret = binary_img_qrcode(
                                        qr_code_img, 0)
                                    qr_code_data_list, rects_list = qr_code_detect(
                                        qr_code_img_binary)
                                    # 检测结果为空，最后采用ocr识别二维码图片中的字母、数字
                                    if len(qr_code_data_list) == 0:
                                        logger.info(
                                            'uuid: {}二维码二值化未识别出二维码信息'.format(uuid))
                                        logger.info(
                                            'uuid: {}进行ocr识别二维码......'.format(uuid))
                                        # ocr识别
                                        boxes, recs = qrcode_ocr_sample(
                                            qr_code_img)
                                        if len(recs) != 0:
                                            # 添加边界条件,如果前两位字母中包含数字8，则替换为字母B
                                            for idx, info in enumerate(recs):
                                                if '8' in info[0][:2]:
                                                    modified_string = recs[idx][0][:2].replace(
                                                        '8', 'B') + recs[idx][0][2:]
                                                    qr_code_data_list = (
                                                        modified_string,)
                                                    qrcode_list.append(
                                                        qr_code_data_list)
                                                else:
                                                    # recs[0][0]='BC005'
                                                    qr_code_data_list = (
                                                        recs[0][0],)
                                                    qrcode_list.append(
                                                        qr_code_data_list)
                                    else:
                                        qrcode_list.append(qr_code_data_list)
                            else:
                                qrcode_list.append(qr_code_data_list)

                        if len(qrcode_list) != 0:
                            qr_code_data_list = qrcode_list[0]

                    if len(qr_code_data_list) > 1:
                        logger.error('uuid: {}识别原封券箱出现了多个二维码'.format(uuid))
                        return None
                    elif len(qr_code_data_list) == 0:
                        logger.warn('uuid:{}识别原封券箱未识别出二维码'.format(uuid))
                    else:
                        qrcode_mes = qr_code_data_list[0]
                        logger.info(
                            'uuid:{}识别原封券箱识别出1个二维码,二维码信息为{}'.format(
                                uuid, qr_code_data_list))
    elif len(ocr_result) == 0:
        if True:
            qr_code_data_list, rects_list = qr_code_detect(img)

            if len(qr_code_data_list) == 0:
                logger.info('uuid: {}二维码原图上未识别出二维码信息'.format(uuid))
                # 检测二维码&裁剪出二维码图片
                qr_code_img_list = []
                detect_result = yoloqrcode_sample.run(img)
                result_list = detect_result['results']
                if len(result_list) != 0:
                    for ele in result_list:
                        location_tuple = tlwh2tlbr(ele['locations'])
                        left, top, right, bottom = location_tuple
                        qrcode_image = img[top:bottom, left:right]
                        qr_code_img_list.append(qrcode_image)

                # 微信模型识别二维码
                qrcode_list = []
                for qr_code_img in qr_code_img_list:
                    qr_code_data_list, rects_list = qr_code_detect(qr_code_img)
                    # 把二维码抠出来的图检测结果为空，对该图做自适应二值化进行检测
                    if len(qr_code_data_list) == 0:
                        logger.info('uuid: {}二维码抠图上未识别出二维码信息'.format(uuid))
                        if qr_code_img != []:
                            qr_code_img_binary, ret = binary_img_qrcode(
                                qr_code_img, 0)
                            qr_code_data_list, rects_list = qr_code_detect(
                                qr_code_img_binary)
                            # 检测结果为空，最后采用ocr识别二维码图片中的字母、数字
                            if len(qr_code_data_list) == 0:
                                logger.info(
                                    'uuid: {}二维码二值化未识别出二维码信息'.format(uuid))
                                logger.info(
                                    'uuid: {}进行ocr识别二维码......'.format(uuid))
                                # ocr识别
                                boxes, recs = qrcode_ocr_sample(
                                    qr_code_img)  # recs=[('BC0076',1.0)]
                                if len(recs) != 0:
                                    # 添加边界条件,如果前两位字母中包含数字8，则替换为字母B
                                    for idx, info in enumerate(recs):
                                        if '8' in info[0][:2]:
                                            modified_string = recs[idx][0][:2].replace(
                                                '8', 'B') + recs[idx][0][2:]
                                            qr_code_data_list = (
                                                modified_string,)
                                            qrcode_list.append(
                                                qr_code_data_list)
                                        else:
                                            # recs[0][0]='BC005'
                                            qr_code_data_list = (recs[0][0],)
                                            qrcode_list.append(
                                                qr_code_data_list)
                            else:
                                qrcode_list.append(qr_code_data_list)
                    else:
                        qrcode_list.append(qr_code_data_list)

                if len(qrcode_list) != 0:
                    qr_code_data_list = qrcode_list[0]

            if len(qr_code_data_list) > 1:
                # logger.error('uuid: {}识别原封券箱出现了多个二维码'.format(uuid))
                return None
            elif len(qr_code_data_list) == 0:
                logger.warn('uuid:{}识别原封券箱未识别出二维码'.format(uuid))
            else:
                qrcode_mes = qr_code_data_list[0]
                # logger.info('uuid:{}识别原封券箱识别出1个二维码,二维码信息为{}'.format(uuid, qr_code_data_list))
    return ocr_result, qrcode_mes
