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
- **TechAnalyst**：5 步 `reasoning_chain` (trend/momentum/volatility/volume/support_resistance) + `conviction`；`reference_target`（非硬止盈）；`thesis_invalid_if`（软退出条件）；**自动计算 `risk_reward`（Python 计算，不信 LLM）**；rating↔价格 cross-field validator（BUY stop 必须 < entry）；ATR 默认 stop = entry − 2*ATR；batch > 30 自动 chunk；预过滤阈值按 ATR 归一化；**signal-age 记忆**（`data/tech/last_ratings.json`）——prompt 里塞昨日 rating context，8+ 天未兑现的 stale 信号 PM 自动减半仓
- **R/R 纪律全链路**：TA 自动算 R/R → PM 按 R/R 分档加减仓（≥3 加、<1.5 要 catalyst 或减半）→ RM 在 prompt 里独立执行否决纪律（R/R<1.5 必须 modification 或 scale_all_buys）
- **相关性集群风控**：`src/data/correlation.py` 算 120 日 pairwise 相关度，新 advisory `correlation_cluster`：BUY + 已持且 corr>0.7 的仓位合计 > 50% 簿本就触发；catchAI 主题假分散
- **回撤感知 sizing**：pipeline 每 morning 算 5d/20d 滚动 return，传 `recent_performance` 给 PM；若 5d<-3% 或 20d<-8% 标 `in_drawdown=True` → PM prompt 要求所有新 BUY 减半
- **RM reasoning_chain**：6 字段 (rr_audit / signal_fidelity / correlation_check / event_risk / sizing_sanity / overall) 强制，最后一道关有审计痕迹
- **Evening 自省**：EveningAnalyst 读昨日 `insights.tomorrow_outlook` → 今日输出 `previous_outlook_assessment` 老实打分，做长期 calibration
- **Midday/Evening Pydantic**：从裸 dict 升级到 `MiddayReview` + `EveningReport`；`MiddayAction.TRAIL_STOP` 强制 `new_stop_price > 0`；typo (TRIAL_STOP) 直接 ValidationError 拦下
- **EarningsAnalyst**：`investment_implications` 必含 5 步 `reasoning_chain` (fundamental_quality / growth_trajectory / strategic_risks / management_execution / valuation_context) —— sentiment 必须可从这 5 字段推导
- **生产侧防挂死**：所有 Alpaca SDK 调用注入 30s HTTP timeout（`_install_http_timeout`），且 launchd plist 外层用 `/opt/homebrew/bin/timeout --kill-after=30 600 ...` 10 分钟兜底——双层防护，防再次出现 13 小时 hang
- **价格 quantize**：`broker.submit_order` 提交前 `_quantize_price(price)` 按 Alpaca tick 规则归整（≥$1 用 2 位小数、<$1 用 4 位）——防 sub-penny reject
- **生产调度（时区弹性）**：launchd 每 30 分钟触发 wrapper（`scripts/run_if_et_window.sh`），wrapper 看 **ET 时间**判断是否在 session 窗口 + 看 last-run 文件去重。morning 09:30-12:00 ET、midday 15:00-16:30 ET、evening 20:00-22:00 ET、Mon-Fri ET。用户出差到任何时区都能正确时刻触发
- **ET 时间统一**：`src/util/time.py` 提供 `et_today()` / `et_now()`；daily_pnl key、insights 查询、broker.is_trading_day、news/macro 快照目录、earnings cutoff、market OHLCV 全部走 ET——跨时区 host 上数据一致

## 开发规范

- Python 3.11+，依赖管理用 pyproject.toml
- 测试：`pytest tests/ -v`（189 tests）
- 配置：`config/settings.yaml`，API key 通过 `${ENV_VAR}` 引用 `.env`
- Agent prompts 在 `config/prompts/*.md`
- 入口：`python main.py --mode morning|midday|evening|live`
- Scheduler (`--mode live`) 时区已 pin 到 US/Eastern；生产路径走 launchd（非此 scheduler）
