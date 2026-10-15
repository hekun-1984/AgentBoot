"""兼容骨架中的 `from agentboot.logger import get_logger`。"""

from agentboot.utils.logger import get_logger, setup_logging

__all__ = ["get_logger", "setup_logging"]
