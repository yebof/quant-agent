# quant-agent

LLM multi-agent 美股量化交易系统，通过 Alpaca 执行交易（默认 paper trading）。

## 入口 + 测试

```bash
pytest tests/ -v                                    # 全量测试
python main.py --mode morning|midday|evening|live   # 手动跑
```

生产路径走 launchd（`~/Library/LaunchAgents/com.quant-agent.*.plist`），不走 `--mode live`。

## 架构速览

- 8 个 LLM agent：tech / news / macro / earnings / portfolio_manager / risk_manager / position_reviewer / evening_analyst
- 双层风控：硬规则引擎（仓位/暴露/日损/板块/相关性/earnings-queued） + LLM RiskManager 审核
- 三时段：morning（分析+交易）、midday（持仓检查+真 trailing stop）、evening（日报 + 次日 insights）
- 数据源：yfinance、FRED、RSS、SEC EDGAR
- 配置：`config/settings.yaml` + `.env`；按 agent 独立选 OpenAI / Anthropic 模型

详细设计见 `README.md`，agent 行为规则见 `config/prompts/*.md`。

## 不要违反的约定（这些不看代码就看不出，违反会出事）

这一节是 CLAUDE.md 的主要价值——约定背后的"为什么"在代码里不写死，所以必须记在这里。

### 金额 / 仓位语义
- **反向 ETF**（SH/SDS/PSQ/SQQQ）用**签名乘数**算净敞口（对冲相消），**abs 乘数**算单仓/板块上限。`src/risk/rules.py:_effective_multiplier` vs `_gross_multiplier`
- **日 P&L** = `broker.equity - broker.last_equity`（含已实现 fill，包括 broker 触发的 OTO 止损）；熔断基准永远是 `last_equity`，**不是**昨晚 DB 里的快照
- **SELL `allocation_pct`** 约定：`100` = 全卖，`1-99` = 部分，`0` = 跳过（**不要再用 0 表示全卖**）；pipeline 会 warning 然后 skip
- **现金 / 保证金**：`RiskConfig.allow_margin` 默认 `false`。三层防护：(1) 硬规则 `cash_only` 拦新 BUY 透支现金（filter 先汇总同 session SELL proceeds 避免误杀轮换）；(2) `_force_delever()` — session 开头 cash<-$1 时**自动按 biggest-loser-first 强卖**到 cash≥0，不依赖 LLM 判断，morning/midday 都跑；(3) PM/midday prompt 的 DE-LEVER MANDATE 段是给 LLM 看的 advisory。强卖用 `FORCE_DELEVER` 动作名（calibration + recent_sells 都识别）。允许保证金请显式改 `allow_margin: true`

### 责任边界
- Macro 拥有 regime 枚举（risk-on / risk-off / transitional / neutral）的权威；News 的 `current_regime` 只描述新闻/地缘背景，不重复 Macro 的枚举
- 所有 SELL 面单 path（morning SELL、midday SELL/REDUCE、emergency sell）提交后都走 `_order_accepted()` 校验，broker 返 error/rejected 不写 trades 表，别再绕开这层

### 时区
- ET 统一走 `src/trading_calendar.py`（`et_today` / `et_now` / `session_date_key` / `in_session_window` / `SESSION_WINDOWS`）。`src/util/time.py` 是向后兼容 shim，新代码直接 import `src.trading_calendar`。`daily_pnl` 主键、`insights` 查询、`broker.is_trading_day`、news/macro 快照目录、earnings cutoff、market OHLCV 全部 ET——任何 host TZ 都要出同样数据
- launchd 每 30 分钟触发 `scripts/run_if_et_window.sh`，wrapper 看 **ET 时间**判断窗口 + last-run 文件去重。窗口对应 `SESSION_WINDOWS`（Python 权威源，bash 表由 `test_trading_calendar.py` 锁定一致）：earnings_preprocess 08:00-09:15、morning 09:30-12:00、intra_check 12:00-13:30、midday 13:00-14:30、close 15:30-15:55、evening 20:00-22:00（Mon-Fri ET）。用户经常出差，不同时区必须都正确。**midday + close 都跑 position_reviewer（同一 agent，`session_type` 分流：midday = patient，close = act-on-trigger；都只卖不买，核心原则"好股长持"）**

### 生产侧防挂死 / 防拒单（都有血泪）
- 所有 Alpaca SDK 调用通过 `_install_http_timeout()` 注入 30s HTTP timeout；launchd plist 外层 `/opt/homebrew/bin/timeout --kill-after=30 600 ...` 10 分钟兜底——**双层**，防再次出现 13 小时 hang（2026-04-17 事故）
- `broker.submit_order` 提交前 `_quantize_price()` 按 Alpaca tick 归整（≥$1 → 0.01、<$1 → 0.0001）——防 sub-penny reject（2026-04-17 UPS 事故）
- 预处理 LLM 分析失败调 `record_failure()`；连 3 次失败就 abandon + 标 `abandoned=True`，不再重分析——防 LLM 失败循环烧 token
- **macOS Sequoia launchd + `com.apple.provenance`**：plist 的 `ProgramArguments` **必须**以 `/bin/bash` 开头，后面把 wrapper 作为参数传（不能直接 exec wrapper）。launchd 只 exec 系统 binary `/bin/bash`，provenance 不会挡；bash 读 wrapper 当文本源走，provenance 也不管。配套 `scripts/install_plists.sh` 会一键重写 + reload，编辑 wrapper 后务必重跑（2026-04-17 周五事故，launchd 5 个 job 全 exit 126 + 没跑）

## 开发规范

- Python 3.11+、依赖在 pyproject.toml
- LLM agent 改动后：改 `config/prompts/*.md` 的 rule + 对应 `src/agents/*.py` 的 build_user_message，然后加 test（在 `tests/test_*.py`）
- 任何进 trades / positions 表的写入必须先过 `_order_accepted()`
- **记忆**：我的长期偏好 / 决策背景见 `~/.claude/projects/-Users-yebof-Documents-Claude-workspace-quant-agent/memory/`
