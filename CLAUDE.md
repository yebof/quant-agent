# quant-agent

LLM multi-agent 美股量化交易系统，通过 Alpaca 执行交易（默认 paper trading）。

## 入口 + 测试

```bash
pytest tests/ -v                                    # 全量测试
python main.py --mode morning|midday|evening|live   # 手动跑
```

生产路径走 launchd（`~/Library/LaunchAgents/com.quant-agent.*.plist`），不走 `--mode live`。

## 架构速览

- 8 个日常 LLM agent：tech / news / macro / earnings / portfolio_manager / risk_manager / position_reviewer / evening_analyst。额外一个 **meta_reflector** 每季度末跑一次，负责对 6 个可编辑 agent（tech / news / macro / earnings / PM / evening）**自我画像 + prompt 审计 + 自动修订**。risk_manager 和 position_reviewer 被 **schema + prompt_editor 双层保护**，不允许被 auto-evolved 改动（硬纪律不能被稀释）
- 双层风控：硬规则引擎（cash_only / 仓位 / 暴露 / 日损 / 板块 / 相关性 / earnings-queued） + LLM RiskManager 审核 + `_force_delever()` 硬兜底
- **6 个 session**（ET Mon-Fri）：earnings_preprocess 08:00-09:15（唯一跑 earnings LLM）、morning 09:30-12:00（full team）、intra_check 09:30-16:00 每 30min tick（熔断器，零 LLM）、midday 13:00-14:30（position_reviewer patient）、close 15:30-16:00（position_reviewer act-on-trigger；窗口 ≥ launchd 30min tick 保证任何 phase 都能打中）、evening 20:00-22:00（report + outlook）。**季度末额外一次 meta**：`--mode meta` / `run_quarterly_meta_reflection`，跑 `quarterly_digest` 聚合 90 天事实 + `meta_reflector` LLM 7 步 CoT + `prompt_editor` 4 道保险 apply
- 数据源：yfinance、FRED、RSS、SEC EDGAR
- 配置：`config/settings.yaml` + `.env`；按 agent 独立选 OpenAI / Anthropic 模型

### Agent CoT 结构（schema-enforced 必填字段数；违反 → ValidationError）
| Agent | CoT 步数 | 备注 |
|---|---|---|
| tech_analyst | 5 | trend / momentum / volatility / volume / S&R |
| news_analyst | ❌ 无 | 故意；结构化输出 `state_changes` + `stock_news` 本身就是它的思维结构 |
| macro_analyst | 6 | vol / yield curve / monetary / inflation+labor+credit / cross-signal / sector |
| earnings_analyst | 5 | 嵌在 `investment_implications.reasoning_chain` |
| portfolio_manager | 7 | 8 层 memory (L1-L8) 喂进来 |
| risk_manager | 6 | rr_audit / signal_fidelity / correlation / event_risk / sizing / overall |
| position_reviewer | 6 | midday + close 共用；`session_type` 切换 disposition |
| meta_reflector | 7 | **facts→synthesis→diagnosis→prompt-audit→proposal** 顺序：performance / theme / loss (事实) → **self_portrait_synthesis**（5 维自画像：conviction_calibration / theme_breadth / loss_discipline / execution_style / agent_balance）→ **portrait_gap_diagnosis**（top 2-3 leverage gap + 归因到 agent）→ **existing_prompt_audit**（读 `agent_prompts_snapshot`——每个 target agent 的 persona + 关键 section + `## Learnings (system-evolved)`——检查 gap 对应的规则是否已存在 / 已失效 / 有冲突）→ prompt_edit_reasoning。**"先看自己是谁、再看哪里差、再看现有 prompt 有没有、最后才提修改"**——防止 LLM 凭记忆重复已有规则或和现有 invariant 冲突 |
| evening_analyst | 7 | performance / retrospection / decision_quality / calibration_meta / regime / **thesis_health_review** / tomorrow_prep。**结构化 `sell_grades` / `buy_grades` 持久化到 `insights.sell_grades_json` / `buy_grades_json`（JSON 列），`_build_trade_grade_summary(lookback=14)` 聚合成 counts + repeat-offender 列表，传入 position_reviewer 形成 SELL 纪律闭环**。同时 `outlook_calibration` 元循环（自己看自己 bias hit rate）自学。**thesis_health_review 每个持仓都读 8 周 tech 轨迹 + 新闻 + 估值 + 最新 10-Q/10-K 的完整 reasoning_chain（src/data/earnings_deep_dive.py 从 `data/earnings/{SYMBOL}/analysis_*.md` 解析 JSON 块、truncate 到 500c/300c），输出 `thesis_trajectory: strengthening/intact/weakening/broken` + `loss_root_cause`——让 evening 能判断"亏损是估值贵买贵了还是基本面坏了"这种 value 投资核心问题** |

详细设计见 `README.md`，agent 行为规则见 `config/prompts/*.md`。

## 不要违反的约定（这些不看代码就看不出，违反会出事）

这一节是 CLAUDE.md 的主要价值——约定背后的"为什么"在代码里不写死，所以必须记在这里。

### 金额 / 仓位语义
- **反向 ETF**（SH/SDS/PSQ/SQQQ）用**签名乘数**算净敞口（对冲相消），**abs 乘数**算单仓/板块上限。`src/risk/rules.py:_effective_multiplier` vs `_gross_multiplier`
- **日 P&L** = `broker.equity - broker.last_equity`（含已实现 fill，包括 broker 触发的 OTO 止损）；熔断基准永远是 `last_equity`，**不是**昨晚 DB 里的快照
- **SELL `allocation_pct`** 约定：`100` = 全卖，`1-99` = 部分，`0` = 跳过（**不要再用 0 表示全卖**）；pipeline 会 warning 然后 skip。**两个地方都要守住**：(a) `ExecutionStage` 执行时 skip；(b) `_filter_hard_risk_decisions` pre-sum SELL proceeds 时也 skip（曾经 filter 错把 alloc=0 当满卖预扣 phantom cash，让 BUY 偷借保证金，2026-04-19 修）
- **现金 / 保证金**：`RiskConfig.allow_margin` 默认 `false`。三层防护：(1) 硬规则 `cash_only` 拦新 BUY 透支现金（filter 先汇总同 session SELL proceeds 避免误杀轮换）；(2) `_force_delever()` — session 开头 cash<-$1 时**自动按 biggest-loser-first 强卖**到 cash≥0，不依赖 LLM 判断，morning/midday 都跑；(3) PM/midday prompt 的 DE-LEVER MANDATE 段是给 LLM 看的 advisory。强卖用 `FORCE_DELEVER` 动作名（calibration + recent_sells 都识别）。阈值 `$1` 单一常量 `src/risk/constants.py:MARGIN_DEFICIT_FLOOR_USD`，三处 import（force_delever / PM prompt / position_reviewer prompt）——别在任何一处重建私有常量。允许保证金请显式改 `allow_margin: true`

### 责任边界
- Macro 拥有 regime 枚举（risk-on / risk-off / transitional / neutral）的权威；News 的 `current_regime` 只描述新闻/地缘背景，不重复 Macro 的枚举
- 所有 SELL 面单 path（morning SELL、midday SELL/REDUCE、emergency sell）提交后都走 `_order_accepted()` 校验，broker 返 error/rejected 不写 trades 表，别再绕开这层
- 同上 SELL path **提交前必须先调 `broker.cancel_protective_stops(symbol)`**——morning BUY 的 OTO stop-loss leg 和 midday/close 的 TRAIL_STOP 都会让 Alpaca 把 shares 标 `held_for_orders=qty`，再下 SELL 必被 `insufficient qty available` 拒（2026-04-25 AMZN 实例）。helper 返回 `(ok, stop_specs)` 元组：`ok=False` 时**直接 skip 这个 symbol**；`ok=True` 时拿到 specs，**SELL 提交后必须**：(a) submit 被 broker 拒 → 调 `broker._restore_stop_orders(symbol, stop_specs)` 恢复保护，(b) **partial exit**（TAKE_PROFIT/REDUCE/PARTIAL_SELL，即 `qty < position.qty`）→ 调 `pipeline._reprotect_residual_after_partial_sell(symbol, residual_qty, stop_specs)` 在残仓上挂新 stop（取 specs 中最高 stop_price 最保护）。这条对 SELL/REDUCE/EMERGENCY_SELL/FORCE_DELEVER/TAKE_PROFIT 5 个 action 全适用——**新增 SELL path 时这三步**（cancel / restore-on-reject / reprotect-on-partial）**全要写齐**，少一个就会留下 residual 裸奔的窗口

### 时区
- ET 统一走 `src/trading_calendar.py`（`et_today` / `et_now` / `session_date_key` / `in_session_window` / `SESSION_WINDOWS`）。`src/util/time.py` 是向后兼容 shim，新代码直接 import `src.trading_calendar`。`daily_pnl` 主键、`insights` 查询、`broker.is_trading_day`、news/macro 快照目录、earnings cutoff、market OHLCV 全部 ET——任何 host TZ 都要出同样数据
- launchd 每 30 分钟触发 `scripts/run_if_et_window.sh`，wrapper 看 **ET 时间**判断窗口 + last-run 文件去重。窗口对应 `SESSION_WINDOWS`（Python 权威源，bash 表由 `test_trading_calendar.py` 锁定一致）：earnings_preprocess 08:00-09:15、morning 09:30-12:00、**intra_check 09:30-16:00（全交易时段每 30 分钟 tick 都跑，stateless 熔断器；last-run 去重 + cross-mode session lock 对它都不生效）**、midday 13:00-14:30、close 15:30-16:00、evening 20:00-22:00（Mon-Fri ET）。**每个窗口宽度必须 ≥ launchd StartInterval (1800s = 30min)**，否则 tick 相位不凑巧会整天错过窗口（2026-04-23/24 close 连续两天因原 25 min 窗口而 miss 的教训）。用户经常出差，不同时区必须都正确。**midday + close 都跑 position_reviewer（同一 agent，`session_type` 分流：midday = patient，close = act-on-trigger；都只卖不买，核心原则"好股长持"）**
- **Cross-mode session lock**（`scripts/run_if_et_window.sh` 的 `acquire_session_lock`）—— 5 个重 LLM session（earnings_preprocess / morning / midday / close / evening）通过 `~/.cache/quant-agent/active-session.lock/` mkdir-as-mutex 串行化：同一时刻只允许一个 python trading session 跑，防长 morning 拖到 midday 时撞车。Stale-lock 1800s 自愈（与外层 1200s kill 配合，30min 后必定干净）。**intra_check 是被显式豁免的**——和 last-run guard 的豁免一条逻辑：stateless 熔断器必须每 tick 都跑，否则 morning 跑久时 flash-crash 防护就哑火，这是熔断器存在的全部意义被否定。修改 lock 逻辑时**永远先确认 intra_check 仍能穿过**（test：`test_run_if_et_window_intra_check_bypasses_session_lock`）

### 生产侧防挂死 / 防拒单（都有血泪）
- 所有 Alpaca SDK 调用通过 `_install_http_timeout()` 注入 30s HTTP timeout；launchd wrapper 外层 `/opt/homebrew/bin/timeout --kill-after=30 1200 ...` 20 分钟兜底（`scripts/run_if_et_window.sh`）——**双层**，防再次出现 13 小时 hang（2026-04-17 事故）。**20 分钟不是随意选的**：morning 正常路径 = tech_analyst 3 chunks × 140-180s + news/macro 并行 + PM 50s + RM + exec，慢 OpenAI 日可达 10-11 min（2026-04-24 Fri 实测），600s 原值连续三次 tick 撞死 status 124。20min 是"1.5-2x 最坏耗时"的 ceiling，既能保护真挂死场景，又不把正常跑击杀
- **LLM client 也要钉 HTTP timeout**：`src/agents/base.py:_LLM_HTTP_TIMEOUT = 300.0` 传给 `OpenAI()` / `Anthropic()` 构造器。SDK 默认 600s 会让单次 stalled SSE 流吃掉整个 session 窗口。**300s 不是随便选的**：tech_analyst `max_tokens=128000` + 25 symbols/chunk 历史正常耗时就是 60-180s，2026-04-24 OpenAI 慢到单 chunk >180s；太紧（60s 初版）会砍掉本来能成功返回的调用 → 触发 retry spiral 撞穿 launchd 600s outer kill。300s 覆盖了"慢的一天"的正常耗时尾部 + buffer，还比 SDK 默认低一半。和 `_BROKER_HTTP_TIMEOUT` 是一条纪律（2026-04-23 morning DNS 事故 + 2026-04-24 morning OpenAI 慢响应事故的合订本——timeout 必须 > slowest legit latency）
- `broker.submit_order` 提交前 `_quantize_price()` 按 Alpaca tick 归整（≥$1 → 0.01、<$1 → 0.0001）——防 sub-penny reject（2026-04-17 UPS 事故）
- 预处理 LLM 分析失败调 `record_failure()`；连 3 次失败就 abandon + 标 `abandoned=True`，不再重分析——防 LLM 失败循环烧 token
- **macOS Sequoia launchd + `com.apple.provenance`**：plist 的 `ProgramArguments` **必须**以 `/bin/bash` 开头，后面把 wrapper 作为参数传（不能直接 exec wrapper）。launchd 只 exec 系统 binary `/bin/bash`，provenance 不会挡；bash 读 wrapper 当文本源走，provenance 也不管。配套 `scripts/install_plists.sh` 会一键重写 + reload，编辑 wrapper 后务必重跑（2026-04-17 周五事故，launchd 5 个 job 全 exit 126 + 没跑）
- **macOS TCC / Full Disk Access**：launchd 派生的 bash 默认没权限读 `~/Documents/` 下的文件。wrapper 通过 `install_plists.sh` 装到 `~/Library/Application Support/quant-agent/` 绕开第一层；但 wrapper 仍要 `source ${PROJECT_ROOT}/.env` 和执行 `~/Documents/.../main.py` 及其 src/ config/ data/ 读取——这些都在 Documents 保护目录里。解法：**System Settings → Privacy & Security → Full Disk Access → `+` → ⌘+Shift+G → `/bin/bash` → 开启**。不加这个，launchd 日志会反复看到 `Operation not permitted`（errno 8）并且 session 全部哑火（2026-04-20 周一事故）
- **Mac hibernate 丢 session**：电池低时 macOS 进 hibernate，launchd 的 StartInterval 任务不触发。若交易日晚上笔记本掉电 hibernate 到 evening 窗口之后，当晚 evening 不跑 → 没生成 insights → 次日 PM 缺衔接信号。交易日建议插电 + `sudo pmset -c sleep 0`，或手工 `python main.py --mode evening` 补跑（evening 窗外也能跑，但记得它不经 wrapper 所以不更新 `~/.cache/quant-agent/last-evening`）（2026-04-21 周二事故）

## 开发规范

- Python 3.11+、依赖在 pyproject.toml
- LLM agent 改动后：改 `config/prompts/*.md` 的 rule + 对应 `src/agents/*.py` 的 build_user_message，然后加 test（在 `tests/test_*.py`）
- 任何进 trades / positions 表的写入必须先过 `_order_accepted()`
- **Env vars（可选调节）**：
  - `QUANT_AGENT_MAX_RETRIES` — base agent LLM 调用重试次数（默认 5，总退避 1+2+4+8+16=31s）。2026-04-23 DNS 抖动事件之后从 3 升到 5，想更严格或测试用可以覆盖
  - `.env` 的必需项：`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `FRED_API_KEY`
- **长期反思（quarterly auto-evolution）**：`evolution.enabled=true` 时，每季末 `--mode meta` 会让 `PromptEditor` 按 `reflection.json` 的提案真实写入 6 个 editable agent 的 prompt 文件的 `## Learnings (system-evolved)` 段。`risk_manager` + `position_reviewer` 被 `MetaReflectionAgentName` schema literal 硬挡，`evolution.enabled=true` 也改不了。4 层护栏：FIFO cap / Jaccard dedup / prohibited-words regex / git auto-commit（`git revert <sha>` 一条命令整季回滚）
- **记忆**：我的长期偏好 / 决策背景见 `~/.claude/projects/-Users-yebof-Documents-Claude-workspace-quant-agent/memory/`
