from flask import Flask


def create_app():
    app = Flask(__name__)
    # 导入蓝图模块
    from .views import coffer_blueprint
    # 注册蓝图
    app.register_blueprint(coffer_blueprint,url_prefix='/coffer_cv')

    from .views import face_blueprint
    app.register_blueprint(face_blueprint,url_prefix='/face')

    return app