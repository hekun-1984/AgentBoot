"""本地运行糖尿病管理助手。

设置 AGENTBOOT_MODEL（例如 openai:gpt-4o-mini）后，路由和 Agent 会改走该模型。
未设置时使用离线契约实现，方便没有密钥时开发和回归。
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples"))

from diabetes import build_app


def main() -> None:
    question = " ".join(sys.argv[1:]).strip() or "我最近空腹血糖经常在 7.8 左右，饮食和运动该怎么调整？"
    result = build_app(model=os.getenv("AGENTBOOT_MODEL")).invoke(question, {"patient_id": "p001"})
    print(result["answer"])
    if result.get("errors"):
        print("---")
        for item in result["errors"]:
            print(f"{item.get('source')}: {item.get('message')}")


if __name__ == "__main__":
    main()
