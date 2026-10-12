"""웹 서버 실행 (모델 서버가 먼저 떠 있어야 한다: scripts/serve.sh).

    python scripts/run_server.py [--port 8080]
    → http://<WSL IP>:8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    uvicorn.run("csagent.server.app:create_default_app", factory=True,
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
