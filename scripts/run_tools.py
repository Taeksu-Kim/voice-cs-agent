"""도메인 API와 도구 서버(MCP)를 각각 별도 프로세스로 띄운다 (docs/08 3절). 에이전트(웹 서버·평가)보다 먼저.

    python scripts/run_tools.py                   # 둘 다 (Ctrl+C로 함께 종료)
    python scripts/run_tools.py --only domain     # 하나만
    → 도메인 API  http://127.0.0.1:8020  (평가·개발 모드 — /_fixtures 켬)
      도구 서버   http://127.0.0.1:8021/mcp  (POST /reload: 카탈로그 다시 읽기)

포트·주소는 .env의 DOMAIN_URL·TOOL_SERVER_URL과 맞출 것.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SERVICES = {
    "domain": ("csagent.domain.app:create_dev_app", 8020),
    "tools": ("csagent.toolserver.server:create_default_app", 8021),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(SERVICES))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUNBUFFERED": "1"}
    env.setdefault("DOMAIN_URL", f"http://{args.host}:{SERVICES['domain'][1]}")
    procs = []
    for name in ([args.only] if args.only else ["domain", "tools"]):
        app, port = SERVICES[name]
        procs.append((name, subprocess.Popen(
            [sys.executable, "-m", "uvicorn", app, "--factory", "--host", args.host, "--port", str(port),
             "--log-level", "warning"], cwd=ROOT, env=env)))
        print(f"{name}: http://{args.host}:{port}", flush=True)
    try:
        while all(p.poll() is None for _, p in procs):
            time.sleep(0.5)
        dead = [n for n, p in procs if p.poll() is not None]
        print(f"!! 종료된 프로세스: {dead}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
        for _, p in procs:
            p.wait(timeout=10)


if __name__ == "__main__":
    main()
