"""Server lokal HTTP që imiton Anthropic Messages API: testet përdorin ChatAnthropic real pa internet dhe pa kosto."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(payload, stop_reason="end_turn"):
    """Përgjigje e suksesshme e Claude: një bllok thinking (bosh, si në API) dhe teksti JSON."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    content = [{"type": "thinking", "thinking": "", "signature": "test-signature"}]
    if text:
        content.append({"type": "text", "text": text})
    return 200, {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 50},
    }


def api_error(status, error_type, message="failure"):
    return status, {"type": "error", "error": {"type": error_type, "message": message}}


def _text(content):
    if isinstance(content, str):
        return content
    return "".join(block.get("text", "") for block in content)


class FakeLLM:
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
                fake.requests.append({"path": self.path, "headers": dict(self.headers), "body": json.loads(body)})
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
        return _text(self.requests[-1]["body"]["messages"][0]["content"])

    @property
    def system_prompt(self):
        return _text(self.requests[-1]["body"]["system"])

    @property
    def schema_properties(self):
        """Fushat e skemës JSON që agjenti kërkoi në kërkesën e fundit."""
        return self.requests[-1]["body"]["output_config"]["format"]["schema"]["properties"]
