"""框架异常。调用方可以按层区分注册错误和执行错误。"""


class AgentBootError(Exception):
    """AgentBoot 异常基类。"""


class RegistrationError(AgentBootError):
    """组件重名、缺依赖或契约不完整。"""


class SkillError(AgentBootError):
    """SKILL.md 缺失或无法解析。"""


class ConfigurationError(AgentBootError):
    """模型配置缺失或无法解析。"""


class AgentExecutionError(AgentBootError):
    """Agent 按 abort 策略失败时抛出。"""

    def __init__(self, message: str, *, error_kind: str, agent: str) -> None:
        super().__init__(message)
        self.error_kind = error_kind
        self.agent = agent
