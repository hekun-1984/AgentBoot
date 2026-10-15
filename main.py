"""AgentBoot 一键启动入口。类似 Spring Boot 的 SpringApplication.run()。"""

from agentboot.app import AgentApp
from agents.diabetes_agent import DiabetesAgent
from agents.finance_agent import FinanceAgent


def main() -> None:
    app = AgentApp(app_name="MyAgentProject")
    app.register_agent(DiabetesAgent)
    app.register_agent(FinanceAgent)
    app.run()


if __name__ == "__main__":
    main()
