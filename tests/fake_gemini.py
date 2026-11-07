"""Server lokal HTTP që imiton Gemini API: testet përdorin modelin real të LangChain pa internet dhe pa kosto."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(payload, finish_reason="STOP"):
    """Përgjigje e suksesshme e Gemini me tekstin (ose JSON-in) e dhënë."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return 200, {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": finish_reason}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
    }


def api_error(code, status, message="failure"):
    return code, {"error": {"code": code, "message": message, "status": status}}


class FakeGemini:
    """Kthen përgjigjen e zgjedhur nga testi (`respond`) dhe ruan kërkesat e marra."""

    def __init__(self):
        self.requests = []
        self.respond = lambda: answer({})
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                fake.requests.append({"path": self.path, "body": json.loads(body)})
                status, payload = fake.respond()
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    @property
    def prompt(self):
        """Teksti që agjenti i dërgoi modelit në kërkesën e fundit."""
        return self.requests[-1]["body"]["contents"][0]["parts"][0]["text"]

    @property
    def system_prompt(self):
        return self.requests[-1]["body"]["systemInstruction"]["parts"][0]["text"]
