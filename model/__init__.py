import tritonclient.http as httpclient
from config import Config

triton_http_client = httpclient.InferenceServerClient(
    url=(Config.SERVER_URL + ':' + Config.SERVER_PORT))
