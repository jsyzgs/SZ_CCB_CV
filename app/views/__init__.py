from flask import Blueprint

coffer_blueprint = Blueprint('coffer', __name__)
from .coffer_cv_views import *

face_blueprint = Blueprint('face',__name__)
from .face_views import *
