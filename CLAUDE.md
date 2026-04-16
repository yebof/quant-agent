# quant-agent

LLM multi-agent 美股量化交易系统，通过 Alpaca 执行交易（默认 paper trading）。

## 架构要点

- **8 个 LLM agent**：tech_analyst, news_analyst, macro_analyst, earnings_analyst, portfolio_manager, risk_manager, midday_reviewer, evening_analyst
- **双层风控**：硬规则引擎（仓位/暴露/日损/板块集中度）+ LLM Risk Manager 审核
- **三个时段**：morning（分析+交易）、midday（持仓检查）、evening（日报）
- **数据源**：yfinance（行情）、FRED（宏观）、RSS（新闻）、SEC EDGAR（财报）
- 支持 OpenAI 和 Anthropic 模型，按 agent 独立配置

## 开发规范

- Python 3.11+，依赖管理用 pyproject.toml
- 测试：`pytest tests/ -v`（115 tests）
- 配置：`config/settings.yaml`，API key 通过 `${ENV_VAR}` 引用 `.env`
- Agent prompts 在 `config/prompts/*.md`
- DB：SQLite，线程安全（`threading.Lock`）
- 入口：`python main.py --mode morning|midday|evening|live`
