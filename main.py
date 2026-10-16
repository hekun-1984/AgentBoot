"""本地运行糖尿病管理助手。

未设置模型时使用离线契约实现。要换真实模型，二选一：

  set AGENTBOOT_MODEL=openai:gpt-4o-mini

或使用案例里的模型表（密钥放在环境变量里）：

  set AGENTBOOT_MODELS=examples/diabetes/models.yaml
  set OPENAI_API_KEY=...
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
