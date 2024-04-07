from gevent import monkey

from app import create_app

app = create_app()
if __name__ == '__main__':
    monkey.patch_all()
    app.run(host='0.0.0.0', port=5002)