# quant-agent

LLM multi-agent 美股量化交易系统，通过 Alpaca 执行交易（默认 paper trading）。

## 架构要点

- **8 个 LLM agent**：tech_analyst, news_analyst, macro_analyst, earnings_analyst, portfolio_manager, risk_manager, midday_reviewer, evening_analyst
- **双层风控**：硬规则引擎（仓位/暴露/日损/板块集中度）+ LLM Risk Manager 审核
- **三个时段**：morning（分析+交易）、midday（持仓检查）、evening（日报）
- **数据源**：yfinance（行情）、FRED（VIX / 收益率 / DFF / CPI / UNRATE / HY OAS）、RSS（新闻）、SEC EDGAR（财报）
- 支持 OpenAI 和 Anthropic 模型，按 agent 独立配置

## 关键约定（2026-04-17 重构后，见 `project_trading_design_decisions.md` 记忆）

- **反向 ETF**（SH/SDS/PSQ/SQQQ）用签名乘数算**净敞口**（对冲相消），abs 算单仓和行业上限
- **日 P&L** = `broker.equity - broker.last_equity`（含已实现 fill，包括 broker 触发的 OTO 止损）；熔断基准为 `last_equity`
- **SELL `allocation_pct`**：100=全卖、1-99=部分、0=跳过（不要再用 0 表示全卖）
- **DB**：SQLite WAL；midday `sync_positions` 整体替换仓位快照；evening `prune_agent_logs(30天)`
- **MacroAnalyst**：6 步 reasoning_chain (vol / curve / monetary / inflation+labor+credit / cross-signal / sector)；持久化到 `data/macro/last_state.json` 做 regime-shift 检测；读昨日 News narrative 做 `alignment_with_news` 交叉验证；输出含 `bull_triggers` / `bear_triggers`；`sector_guidance.sector` 限定为 yfinance 枚举
- **Midday trailing stop 是真实订单**：`TRAIL_STOP` 动作走 `broker.replace_stop_loss()`——取消旧 stop + 下新 stop。HOLD 是不动、REDUCE 是真卖半仓、SELL 是真全平
- **RiskManager 可以 `scale_all_buys: 0.0-1.0`** 对所有 BUY 做组合级缩放；看到 `tech_analyses` 可以审计 PM 对底层信号的忠实度
- **硬风控 + macro_exposure_deviation 软违规**：实际净敞口偏离 Macro 的 `target_invested_pct` > 15pp 时给 RM 一个 advisory（不拦单）
- **责任边界**：Macro 拥有 regime 枚举的权威；News 的 `current_regime` 只描述新闻/地缘背景，不复述枚举
- **TechAnalyst**：5 步 `reasoning_chain` (trend/momentum/volatility/volume/support_resistance) + `conviction`；`reference_target`（非硬止盈）；rating↔价格 cross-field validator（BUY stop 必须 < entry）；ATR 默认 stop = entry − 2*ATR；batch > 30 自动 chunk；预过滤阈值按 ATR 归一化
- **生产调度**：macOS launchd，Mon-Fri SGT 22:00/04:00/08:30 → morning/midday/evening（对应美东 10:00/16:00/20:30）

## 开发规范

- Python 3.11+，依赖管理用 pyproject.toml
- 测试：`pytest tests/ -v`（157 tests）
- 配置：`config/settings.yaml`，API key 通过 `${ENV_VAR}` 引用 `.env`
- Agent prompts 在 `config/prompts/*.md`
- 入口：`python main.py --mode morning|midday|evening|live`
- Scheduler (`--mode live`) 时区已 pin 到 US/Eastern；生产路径走 launchd（非此 scheduler）
