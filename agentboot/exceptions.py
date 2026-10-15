"""AgentBoot 自定义异常。"""


class AgentBootError(Exception):
    """框架基础异常。"""


class RegistrationError(AgentBootError):
    """组件重复注册或注册信息不合法。"""


class PlanValidationError(AgentBootError):
    """执行计划未通过依赖图校验。"""


class RoutingError(AgentBootError):
    """请求的能力不在白名单内。"""


class ExecutionError(AgentBootError):
    """步骤执行失败。"""


class ConfigurationError(AgentBootError):
    """配置缺失或无法解析。"""
