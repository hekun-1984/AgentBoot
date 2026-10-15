"""Python 依赖关系校验器。

在执行计划生成后校验依赖图：
- 检测循环依赖
- 检测不可达节点
- 校验子图和能力是否已注册
- 校验并行组内没有相互依赖
"""

from __future__ import annotations

from agentboot.exceptions import PlanValidationError
from agentboot.logger import get_logger
from agentboot.models import ExecutionPlan, Step

logger = get_logger(__name__)

_STEP_TYPES = {"subgraph", "skill", "llm", "condition"}


class PlanValidator:
    """执行计划校验器。"""

    def __init__(
        self,
        registered_subgraphs: set[str],
        registered_skills: set[str] | None = None,
    ) -> None:
        self.registered_subgraphs = registered_subgraphs
        self.registered_skills = registered_skills or set()

    def validate(self, plan: ExecutionPlan) -> bool:
        steps = plan.steps
        if not steps:
            raise PlanValidationError(f"Plan '{plan.plan_id}' has no steps.")
        step_ids = [step.id for step in steps]
        if len(set(step_ids)) != len(step_ids):
            raise PlanValidationError("Duplicate step IDs found.")
        known = set(step_ids)
        for step in steps:
            if step.type not in _STEP_TYPES:
                raise PlanValidationError(f"Unsupported step type '{step.type}' on '{step.id}'.")
            if len(step.depends_on) != len(set(step.depends_on)):
                raise PlanValidationError(f"Step '{step.id}' has duplicate dependencies.")
            for dep_id in step.depends_on:
                if dep_id not in known:
                    raise PlanValidationError(
                        f"Step '{step.id}' depends on unknown step '{dep_id}'."
                    )
            self._ensure_registered(step)
        self._check_circular_deps(steps)
        self._check_parallel_conflicts(steps)
        self._check_unreachable(steps)
        logger.info("Plan '%s' validation passed.", plan.plan_id)
        return True

    def _ensure_registered(self, step: Step) -> None:
        if step.type == "subgraph" and step.id not in self.registered_subgraphs:
            raise PlanValidationError(f"SubGraph '{step.id}' is not registered.")
        if step.type == "skill" and step.id not in self.registered_skills:
            raise PlanValidationError(f"Skill '{step.id}' is not registered.")

    def _check_circular_deps(self, steps: list[Step]) -> None:
        """检测循环依赖（DFS 着色法）。"""
        white, gray, black = 0, 1, 2
        color = {step.id: white for step in steps}
        lookup = {step.id: step for step in steps}

        def dfs(step_id: str) -> bool:
            color[step_id] = gray
            for dep_id in lookup[step_id].depends_on:
                if color.get(dep_id) == gray:
                    return True
                if color.get(dep_id) == white and dfs(dep_id):
                    return True
            color[step_id] = black
            return False

        for step in steps:
            if color[step.id] == white and dfs(step.id):
                raise PlanValidationError("Circular dependency detected.")

    def _check_parallel_conflicts(self, steps: list[Step]) -> None:
        """同一并行组内的步骤不能直接或间接互相依赖。"""
        lookup = {step.id: step for step in steps}
        groups: dict[str, set[str]] = {}
        for step in steps:
            if step.parallel_group:
                groups.setdefault(step.parallel_group, set()).add(step.id)

        def ancestors(step_id: str, seen: set[str] | None = None) -> set[str]:
            seen = seen or set()
            found: set[str] = set()
            for dep_id in lookup[step_id].depends_on:
                if dep_id in seen:
                    continue
                found.add(dep_id)
                found.update(ancestors(dep_id, seen | {dep_id}))
            return found

        for group_name, members in groups.items():
            for step_id in members:
                conflict = ancestors(step_id) & members
                if conflict:
                    raise PlanValidationError(
                        f"Parallel group '{group_name}' has conflicting deps: "
                        f"{step_id} depends on {sorted(conflict)}."
                    )

    def _check_unreachable(self, steps: list[Step]) -> None:
        """从无依赖的根节点出发，发现无法到达的步骤。"""
        dependents: dict[str, list[str]] = {step.id: [] for step in steps}
        roots: list[str] = []
        for step in steps:
            if not step.depends_on:
                roots.append(step.id)
            for dep_id in step.depends_on:
                dependents[dep_id].append(step.id)
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(dependents[current])
        unreachable = [step.id for step in steps if step.id not in seen]
        if unreachable:
            raise PlanValidationError(f"Unreachable steps: {unreachable}.")
