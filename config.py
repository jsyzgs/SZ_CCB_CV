import os


class Config:
    SERVER_URL = '10.2.72.189'
    SERVER_PORT = '8000'
    FONT_PATH = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        'utils',
        'font',
        'simhei.ttf')  # 可以展示中文

    SECURITIES_TYPE_2 = {'s_100': '100元', 's_50': '50元', 's_20': '20元', 's_10': '10元', 's_5': '5元',
                         'b_100': '100元', 'b_50': '50元', 'b_20': '20元', 'b_10': '10元', 'b_5': '5元',
                         'bundle': 'coin_0.5',  # TODO:等到出现其他捆装硬币时重新训练，目前只有伍角的
                         'coin_1': '1元', 'coin_0.5': '0.5元', 'coin_0.1': '0.1元',
                         'y': '倾倒钱捆'}

    OBB_LABELS = ['s_100', 's_50', 's_20', 's_10', 's_5',
                  'b_100', 'b_50', 'b_20', 'b_10', 'b_5',
                  'bundle',
                  'coin_1', 'coin_0.1',
                  'y']

    CLASS_LABELS = ['coin_0.1', 'coin_0.5', 'coin_1']
    VERIFY_LABELS = ['s_5', 's_10', 's_20', 's_50', 's_100']
    QRCODE_LABELS = ['qrcode']
    LABELS = ["s_100", "s_50", "s_20", "s_10", "s_5",
              "b_100", "b_50", "b_20", "b_10", "b_5",
              "y",
              "coin_1", "coin_0.5", "coin_0.1"]
    CHINESE_WITH_CURRENCY = {
        "侠": "50",
        "俊": "100",
        "伶": "20",
        "优": "10",
        "扣": "5",
        "仙": "5",
        "抚": "10"}
    MONEY_TO_ALPHABET = {
        "100": "A",
        "50": "B",
        "20": "C",
        "10": "D",
        "5": "E",
        "1": "F"}
