from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="web", **kwargs)

def run_server():
    server_address = ('0.0.0.0', 8080)
    httpd = HTTPServer(server_address, Handler)
    print(f"Server running on http://0.0.0.0:8080")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()
