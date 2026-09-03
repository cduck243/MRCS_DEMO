#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRCS 로컬 실행 서버

하는 일 두 가지
  ① 화면(index.html, step1~5.html)을 띄웁니다
  ② /api/brief 요청을 받아 로컬 모델(Ollama)에게 문장을 시킵니다

브라우저가 Ollama를 직접 부르지 않고 이 서버를 거치는 이유
  - 브라우저의 교차 출처 제한(CORS)에 걸리지 않습니다
  - https 페이지에서 http://localhost 를 부르면 막히는 문제도 없습니다
  - 나중에 실서비스로 옮길 때 이 자리가 그대로 서버 API가 됩니다

실행
  python3 tools/serve.py              → http://localhost:8800
  python3 tools/serve.py --port 9000
  python3 tools/serve.py --model qwen3:8b

표준 라이브러리만 씁니다. 설치할 것이 없습니다.
"""
import argparse, http.server, json, os, socketserver, sys, threading, webbrowser
from urllib.error import URLError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ai_brief  # 프롬프트 · 검사기 · 폴백이 모두 여기 있습니다

STATE = {"model": "qwen3:8b", "data": None}


def load_data():
    if STATE["data"] is None:
        with open(os.path.join(ROOT, "data", "mrcs_alt_data.json"), encoding="utf-8") as f:
            STATE["data"] = json.load(f)
    return STATE["data"]


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def log_message(self, fmt, *args):
        if "/api/" in (self.path or ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/health"):
            try:
                import urllib.request
                with urllib.request.urlopen(ai_brief.OLLAMA + "/api/tags", timeout=3) as r:
                    tags = json.loads(r.read().decode())
                names = [m["name"] for m in tags.get("models", [])]
                return self._json(200, {"ok": True, "model": STATE["model"],
                                        "ready": STATE["model"] in names,
                                        "models": names})
            except Exception as e:
                return self._json(200, {"ok": False, "reason": str(e)})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/brief"):
            return self._json(404, {"error": "없는 경로입니다"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": "요청을 읽지 못했습니다"})

        sex = req.get("sex", "m")
        att = req.get("att", "balanced")
        model = req.get("model") or STATE["model"]
        D = load_data()
        combo = D["combos"].get(f"{sex}:{att}")
        if not combo:
            return self._json(400, {"error": "없는 조합입니다"})

        try:
            r = ai_brief.run(model, combo, D["meta"], D["benefitLabel"],
                             D["order"], retries=2, verbose=True)
        except URLError as e:
            return self._json(200, {"ok": False,
                                    "reason": "모델 서버에 연결하지 못했습니다 (%s)" % e.reason})
        except Exception as e:
            return self._json(200, {"ok": False, "reason": str(e)})

        if not r["ok"]:
            return self._json(200, {"ok": False,
                                    "reason": "검사를 통과하지 못했습니다"})
        return self._json(200, {"ok": True, "model": model,
                                "sec": r["sec"], "tries": r["tries"],
                                "paragraphs": r["paragraphs"]})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()
    STATE["model"] = a.model

    url = f"http://localhost:{a.port}/"
    print(f"""
  MRCS 생애위험준비지도 — 로컬 실행

    주소   {url}
    모델   {a.model}

    AI 문장 생성은 5단계 화면의 [AI로 다시 쓰기] 버튼을 누르면 동작합니다.
    모델이 없거나 답이 검사를 통과하지 못하면 기본 문장이 그대로 보입니다.

    멈추려면 Control-C
""")
    if not a.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        with Server(("127.0.0.1", a.port), Handler) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  종료했습니다.")
    except OSError as e:
        print(f"\n  {a.port} 번 포트를 쓸 수 없습니다 ({e}).")
        print(f"  다른 포트로 실행해 보십시오 —  python3 tools/serve.py --port 8900")


if __name__ == "__main__":
    main()
