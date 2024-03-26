import os
from loguru import logger

log_script_file_path = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
# print(log_script_file_path)

log_dir = log_script_file_path + '/logs/'


def make_log_file(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


logger.add(os.path.join(make_log_file(log_dir), 'flask_app.log'))
flask_log = logger
