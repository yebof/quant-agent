# quant-agent

LLM multi-agent 美股量化交易系统，通过 Alpaca 执行交易（默认 paper trading）。

## 入口 + 测试

```bash
pytest tests/ -v                                    # 全量测试
python main.py --mode morning|midday|evening|live   # 手动跑
```

生产路径走 OS-level timer，不走 `--mode live`：
- **Linux（当前部署，2026-05-11 起）**：systemd 用户 timer `~/.config/systemd/user/quant-agent@.timer` + service template，`Linger=yes` 让 user 退出登录后 timer 仍触发。状态：`systemctl --user list-timers 'quant-agent@*'`、日志：`journalctl --user -u 'quant-agent@*.service'`
- **macOS（legacy）**：launchd `~/Library/LaunchAgents/com.quant-agent.*.plist`，配套 `scripts/install_plists.sh` 安装/重载。**2026-05-11 起 Mac 路径已停用**（迁到 Linux server）；plist 文件保留作 fallback，详细注意事项见下面 "macOS Sequoia" 段

## 架构速览

- 8 个日常 LLM agent：tech / news / macro / earnings / portfolio_manager / risk_manager / position_reviewer / evening_analyst。额外一个 **meta_reflector** 每季度末跑一次，负责对 6 个可编辑 agent（tech / news / macro / earnings / PM / evening）**自我画像 + prompt 审计 + 自动修订**。risk_manager 和 position_reviewer 被 **schema + prompt_editor 双层保护**，不允许被 auto-evolved 改动（硬纪律不能被稀释）
- 双层风控：硬规则引擎（cash_only / 仓位 / 暴露 / 日损 / 板块 / 相关性 / earnings-queued） + LLM RiskManager 审核 + `_force_delever()` 硬兜底
- **6 个 session**（ET Mon-Fri）：earnings_preprocess 08:00-09:15（唯一跑 earnings LLM）、morning 09:30-12:00（full team）、intra_check 09:30-16:00 每 30min tick（熔断器，零 LLM）、midday 13:00-14:30（position_reviewer patient）、close 15:30-16:00（position_reviewer act-on-trigger；窗口 ≥ launchd 30min tick 保证任何 phase 都能打中）、evening 20:00-22:00（report + outlook）。**季度末额外一次 meta**：`--mode meta` / `run_quarterly_meta_reflection`，跑 `quarterly_digest` 聚合 90 天事实 + `meta_reflector` LLM 7 步 CoT + `prompt_editor` 4 道保险 apply
- 数据源：yfinance、FRED、RSS、SEC EDGAR
- 配置：`config/settings.yaml` + `.env`；按 agent 独立选 OpenAI / Anthropic / DeepSeek 模型。**2026-06-04 起所有 9 个 agent 用 OpenAI `gpt-5.5`**（5-11 曾切 claude-opus-4-7 应对 OpenAI quota，6-04 又切回 OpenAI 并升到 5.5；切 provider 一条 `sed` 命令）。Provider 路由按 model name 前缀判断：`deepseek-` 走 DeepSeek，`gpt-` / `o1-` / `o3-` / `o4-` 走 OpenAI，其它走 Anthropic（`src/agents/base.py` 的 `_DEEPSEEK_PREFIXES` / `_OPENAI_PREFIXES`）
- **DeepSeek（OpenAI-compatible，2026-06-05 加）**：走 openai SDK + `base_url=https://api.deepseek.com` + DeepSeek key（`_call_deepseek`）。三个坑都已处理（研究自 api-docs.deepseek.com）：(1) DeepSeek 只认 **`max_tokens`** 不认 OpenAI 的 `max_completion_tokens`（发错会被静默丢弃 → 回落 ~4096 默认截断）；(2) DeepSeek **拒绝**（不裁剪）超 ceiling 的 max_tokens，所以按 `_DEEPSEEK_MAX_OUTPUT` per-model 客户端 clamp（v4-flash/pro/chat/reasoner=384K，未知 deepseek-* 保守 8192）；(3) 402「Insufficient Balance」= 不可重试 → 触发 failover，`insufficient_system_resource` finish_reason 记为 truncated。**`deepseek-chat`/`deepseek-reasoner` 2026-07-24 弃用**（现已 alias `deepseek-v4-flash`），新配置直接用 `deepseek-v4-flash`。cost 用**官方** $0.14/$0.28（LiteLLM 的 $0.28/$0.42 是 V4 前旧值，已 **pin** 在 `cost_table._PRICING_PINNED` 防 cache 刷新覆盖）
- **跨 provider 自动 failover**：当**非-Anthropic 主**（OpenAI 或 DeepSeek）调用重试耗尽 / 非可重试错误(quota、DeepSeek 402、死 key、宕机)后，`base.py:run()` 会**自动用 Anthropic 的 `_FALLBACK_MODEL`(=`claude-opus-4-7`)单发一次**(无重试,避免吃穿 session 窗口),成功就用它的结果继续(`AgentResult.model` 记实际用的模型,cost 按实际模型算)、失败就抛出原始错误。只在「主=OpenAI/DeepSeek 且 `.env` 有 ANTHROPIC_API_KEY」时触发;主已是 Claude 则 no-op(同 provider 无意义,且构造时不会因此报错)。截断(max_tokens)不触发 failover。pipeline 给 9 个 agent 都传 `fallback_api_key=config.api_keys.anthropic`。`src/config.py` 的 LLMConfig 默认(settings.yaml 漏配时的兜底)也已从过时的 `*-4-6` 更到 `claude-opus-4-7`
- **Telegram 推送**：开/关由 `.env` 控制，缺 `TELEGRAM_BOT_TOKEN` 或 `TELEGRAM_CHAT_ID` 时 notifier 静默 no-op，trading 不受影响。每个 session 在 `main.py` finally 块里调一次 `notifier.send(format_session_result(...))`；噪声策略：morning/midday/close/evening 总推；earnings_preprocess 只在真分析了 filing 时推；intra_check 只在 emergency 触发时推（14 次/天 OK tick 静默）；meta 只在真季末跑时推；**任何 session 抛异常都强制推**绕过噪声策略。文档见 `src/notifier.py` docstring 和 README "Optional env vars" 段

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
- 同上 SELL path **提交前必须先调 `broker.cancel_protective_stops(symbol)`**——morning BUY 的 OTO stop-loss leg 和 midday/close 的 TRAIL_STOP 都会让 Alpaca 把 shares 标 `held_for_orders=qty`，再下 SELL 必被 `insufficient qty available` 拒（2026-04-25 AMZN 实例）。helper 返回 `(ok, stop_specs)` 元组。**完整三步纪律**：
  1. **`ok=False`** → 直接 skip 这个 symbol（不要盲目下单）
  2. **`_order_accepted=False`**（broker 当场拒单）→ 立刻调 `broker._restore_stop_orders(symbol, stop_specs)` 恢复保护
  3. **submit 接受后必须 stash `(order_id, position_qty, specs)` 到 `pending_protections`，等 SELL 跑完一轮后调 `wait_for_order_terminal` + `pipeline._finalize_protection_after_sell(...)`** —— **finalize 必须基于实际 fill_qty，不能在 submit 接受时就 reprotect residual**：accepted 的 limit 后续可能 cancel/expire 不成交（PR I 的洞，2026-04-26 codex r4 修复 PR J）。finalize 的三种分支：填 0 → 用 specs restore 原仓覆盖；填 < position → reprotect 实际 residual = `position - fill_qty`；全部填满 → 完整退出无残仓。**还有一种特殊路径**：`wait_for_order_terminal` 15s 超时仍非终态时（broker 还在 hold 这单），finalize 不能直接走——会和 broker 抢，restore stop 时撞 held_for_orders。这种情况要 **先 `cancel_order_by_id` 强制收敛到终态再 finalize**（PR K 修复，codex r5）。**最后一种边界**：cancel 调用本身失败 OR cancel 接受了但 5s 内仍没收敛到终态（pending_cancel/new）OR restore 抛异常 OR restore 部分成功（1/N，剩下的 spec 失败）OR reprotect submit 抛异常——这些"finalize 知道自己没补好保护"的场景，统一走 **`_persist_orphaned_protection_restore(...)` 写到 `pending_protection_restores` 表**，下次 session 入口的 `_drain_pending_protection_restores()` 会重读 broker 状态、终态后用持久化的 specs 调 finalize 把保护补回来（PR O 修复 codex r7 #3，PR R 让 finalize 返回 bool 让 drain 不误删 row，PR S 修 codex r9——partial restore 持久化的是 **失败的那部分 spec**，不是全部 originals，避免和已经活在 broker 的 stop 重复）。`_restore_stop_orders` 现在返回 `(restored_count, failed_specs)`，`finalize` 在非-drain 路径下任何失败分支都自动 persist。drain 调 finalize 时传 `from_drain=True` 跳过 persist 防止 race 时重复写
  这条对 SELL/REDUCE/EMERGENCY_SELL/FORCE_DELEVER/TAKE_PROFIT 5 个 action 全适用——**新增 SELL path 必须把 cancel / restore-on-reject / wait+finalize-on-actual-fill 三步全写齐**，少一个就会有残仓裸奔窗口（partial 没成交时尤其危险）

### 时区
- ET 统一走 `src/trading_calendar.py`（`et_today` / `et_now` / `session_date_key` / `in_session_window` / `SESSION_WINDOWS`）。`src/util/time.py` 是向后兼容 shim，新代码直接 import `src.trading_calendar`。`daily_pnl` 主键、`insights` 查询、`broker.is_trading_day`、news/macro 快照目录、earnings cutoff、market OHLCV 全部 ET——任何 host TZ 都要出同样数据
- OS-level timer（Linux systemd / macOS launchd）每 30 分钟触发 `scripts/run_if_et_window.sh`，wrapper 看 **ET 时间**判断窗口 + last-run 文件去重。窗口对应 `SESSION_WINDOWS`（Python 权威源，bash 表由 `test_trading_calendar.py` 锁定一致）：earnings_preprocess 08:00-09:15、morning 09:30-12:00、**intra_check 09:30-16:00（全交易时段每 30 分钟 tick 都跑，stateless 熔断器；last-run 去重 + cross-mode session lock 对它都不生效）**、midday 13:00-14:30、close 15:30-16:00、evening 20:00-22:00（Mon-Fri ET）。**每个窗口宽度必须 ≥ OS timer 30min tick interval**（systemd `OnCalendar=*:00,30:00` / launchd `StartInterval=1800`），否则 tick 相位不凑巧会整天错过窗口（2026-04-23/24 close 连续两天因原 25 min 窗口而 miss 的教训）。用户经常出差，不同时区必须都正确。**midday + close 都跑 position_reviewer（同一 agent，`session_type` 分流：midday = patient，close = act-on-trigger；都只卖不买，核心原则"好股长持"）**
- **Cross-mode session lock**（`scripts/run_if_et_window.sh` 的 `acquire_session_lock`）—— 5 个重 LLM session（earnings_preprocess / morning / midday / close / evening）通过 `~/.cache/quant-agent/active-session.lock/` mkdir-as-mutex 串行化：同一时刻只允许一个 python trading session 跑，防长 morning 拖到 midday 时撞车。Stale-lock 1800s 自愈（与外层 1200s kill 配合，30min 后必定干净）。**intra_check 是被显式豁免的**——和 last-run guard 的豁免一条逻辑：stateless 熔断器必须每 tick 都跑，否则 morning 跑久时 flash-crash 防护就哑火，这是熔断器存在的全部意义被否定。修改 lock 逻辑时**永远先确认 intra_check 仍能穿过**（test：`test_run_if_et_window_intra_check_bypasses_session_lock`）

### 生产侧防挂死 / 防拒单（都有血泪）
- 所有 Alpaca SDK 调用通过 `_install_http_timeout()` 注入 30s HTTP timeout；wrapper 外层 `timeout --kill-after=30 1200 ...` 20 分钟兜底（Linux：`/usr/bin/timeout`；macOS：`/opt/homebrew/bin/timeout` via brew coreutils；wrapper 用 `TIMEOUT_OVERRIDE` env 选）——**双层**，防再次出现 13 小时 hang（2026-04-17 事故）。systemd `quant-agent@.service` 还有第三层 `TimeoutStartSec=1500` 安全网，让 systemd 在 wrapper 自己的 1200+30s kill 之外再多 270s 兜底。**20 分钟不是随意选的**：morning 正常路径 = tech_analyst 3 chunks × 140-180s + news/macro 并行 + PM 50s + RM + exec，慢 OpenAI 日可达 10-11 min（2026-04-24 Fri 实测），600s 原值连续三次 tick 撞死 status 124。20min 是"1.5-2x 最坏耗时"的 ceiling，既能保护真挂死场景，又不把正常跑击杀
- **LLM client 也要钉 HTTP timeout**：`src/agents/base.py:_LLM_HTTP_TIMEOUT = 300.0` 传给 `OpenAI()` / `Anthropic()` 构造器。SDK 默认 600s 会让单次 stalled SSE 流吃掉整个 session 窗口。**300s 不是随便选的**：tech_analyst `max_tokens=128000` + 25 symbols/chunk 历史正常耗时就是 60-180s，2026-04-24 OpenAI 慢到单 chunk >180s；太紧（60s 初版）会砍掉本来能成功返回的调用 → 触发 retry spiral 撞穿 launchd 600s outer kill。300s 覆盖了"慢的一天"的正常耗时尾部 + buffer，还比 SDK 默认低一半。和 `_BROKER_HTTP_TIMEOUT` 是一条纪律（2026-04-23 morning DNS 事故 + 2026-04-24 morning OpenAI 慢响应事故的合订本——timeout 必须 > slowest legit latency）
- `broker.submit_order` 提交前 `_quantize_price()` 按 Alpaca tick 归整（≥$1 → 0.01、<$1 → 0.0001）——防 sub-penny reject（2026-04-17 UPS 事故）
- 预处理 LLM 分析失败调 `record_failure()`；连 3 次失败就 abandon + 标 `abandoned=True`，不再重分析——防 LLM 失败循环烧 token
- **macOS Sequoia launchd + `com.apple.provenance`**：plist 的 `ProgramArguments` **必须**以 `/bin/bash` 开头，后面把 wrapper 作为参数传（不能直接 exec wrapper）。launchd 只 exec 系统 binary `/bin/bash`，provenance 不会挡；bash 读 wrapper 当文本源走，provenance 也不管。配套 `scripts/install_plists.sh` 会一键重写 + reload，编辑 wrapper 后务必重跑（2026-04-17 周五事故，launchd 5 个 job 全 exit 126 + 没跑）
- **macOS TCC / Full Disk Access**：launchd 派生的 bash 默认没权限读 `~/Documents/` 下的文件。wrapper 通过 `install_plists.sh` 装到 `~/Library/Application Support/quant-agent/` 绕开第一层；但 wrapper 仍要 `source ${PROJECT_ROOT}/.env` 和执行 `~/Documents/.../main.py` 及其 src/ config/ data/ 读取——这些都在 Documents 保护目录里。解法：**System Settings → Privacy & Security → Full Disk Access → `+` → ⌘+Shift+G → `/bin/bash` → 开启**。不加这个，launchd 日志会反复看到 `Operation not permitted`（errno 8）并且 session 全部哑火（2026-04-20 周一事故）
- **Mac hibernate 丢 session**：电池低时 macOS 进 hibernate，launchd 的 StartInterval 任务不触发。若交易日晚上笔记本掉电 hibernate 到 evening 窗口之后，当晚 evening 不跑 → 没生成 insights → 次日 PM 缺衔接信号。交易日建议插电 + `sudo pmset -c sleep 0`，或手工 `python main.py --mode evening` 补跑（evening 窗外也能跑，但记得它不经 wrapper 所以不更新 `~/.cache/quant-agent/last-evening`）（2026-04-21 周二事故）。**注意**：迁到 Linux server 后这条不适用——`Linger=yes` + 服务器常开电源就没这个问题，留作 Mac fallback 时的参考。

### Telegram 推送 / 可观测性
- `src/notifier.py:TelegramNotifier`，在 `main.py` finally 块里调用 `format_session_result(mode, result, elapsed, error=...)` 推送。**Hook 必须在 finally 里**，不能在 try 内部——否则 session 抛异常时收不到 FAILED 推送（这是日志之外操作员唯一的实时信号）
- 错误必须 swallow：`notifier.send()` 内部 `except Exception` 兜住所有 HTTP / 网络 / Telegram-端报错；main.py finally 块再包一层 try/except 防 notifier 自己挂。**`notify` 失败永远不能让 session 失败**，这条比"得到通知"重要
- 噪声策略由 `format_session_result` 返回 `None` 实现（caller 看 `None` 就 skip send）。policy 见模块 docstring；改这条策略前先想清楚"这个 silence 是不是把真信号也吞了"——典型反例：earnings_preprocess 当时把 `analysis_error` 也 silence 过，结果 OpenAI quota 耗尽那天 13 个 filing 全 retry 烧 token 但没人收到通知（2026-05-11 修，`analysis_error` 现在推）
- **确定性升级告警**：evening 推送的 🚨 banner 不只看 LLM 的 `risk_rating`——`_append_evening_body` 还独立算"今日亏损 ≥ 80% 日损熔断线"（用 `result["max_daily_loss_pct"]`）触发 `DETERMINISTIC ALERT`，与 LLM 判断 OR。理由：自评风险时 LLM 最容易**低估**，而这正是最该被抓住的情况——镜像交易路径"硬规则 + LLM"两层哲学。`suggested_actions` 已挪到 P&L 历史表**之前**渲染，避免 4000 字尾部截断在高风险日吃掉最该读的行
- **内部 dead-man's check**：evening（已 gated 在交易日）调 `_expected_sessions_missing_today()`，查 `agent_logs` 今日 ET 是否有 morning(`run-`)/midday/close 的 run_id 前缀；缺了就在推送顶部 🔴(morning)/⚠️(midday·close)。catch "某 session 静默没跑"（timer 挂 / lock 卡 / 半日盘窗口算错）——push-on-completion 观测唯一看不见的失败模式。**不覆盖主机宕机 / evening 本身没跑**——那需要外部 dead-man's switch（healthchecks.io 式，wrapper 成功就 ping、缺席就外部报警），建议补上
- Telegram bot token 等同密码：写 `.env` 用 `chmod 600`，**不要**贴 git / issue / 公开 chat。token 万一外泄（推送的截图 / 误贴 ssh log）马上去 BotFather 发 `/revoke` 生新的

## 开发规范

- Python 3.11+、依赖在 pyproject.toml
- LLM agent 改动后：改 `config/prompts/*.md` 的 rule + 对应 `src/agents/*.py` 的 build_user_message，然后加 test（在 `tests/test_*.py`）
- 任何进 trades / positions 表的写入必须先过 `_order_accepted()`
- **Env vars（可选调节）**：
  - `QUANT_AGENT_MAX_RETRIES` — base agent LLM 调用重试次数（**默认 7，带 jitter**）。退避用 `_retry_backoff_seconds()`：每次睡 `[2^attempt, 2*2^attempt)` 秒（exponential floor + full positive jitter），6 次 sleep 总最坏 ~126s，加 7 次 fast-fail call 延迟 ~14s，~140s 窗口。**演化**：3→5 (2026-04-23 DNS 抖动) → 7+jitter (2026-04-28+29 RM 阶段连续两天 30s 网络中断把 5 retry 全吞)。jitter 是关键 — 没 jitter 时所有 retry 时机确定，30s outage 必吞所有 5 次 retry；有 jitter 后 retry 时机随机散开，至少有概率落在 outage 之后。**只对 transient 错误重试**：`base.py:_is_retryable()` 对 connection/timeout/429/5xx 重试，对 4xx（401 dead key / 400 bad-request / context-length）**快速失败不重试**——避免一个永不可能成功的错误烧满 ~140s × 4-5 agent 把 session 推向 1200s outer kill，也让操作员能区分"网络抖动 vs key 死了"（2026-05-11 quota 耗尽的教训）
  - **LLM 截断检测**：`AgentResult.truncated` 在 `stop_reason=max_tokens`（Anthropic）/ `finish_reason=length`（OpenAI）时为 True 并 `logger.warning`——一个被腰斩的 PM 决策 parse 成 None 后和"真的不交易"长得一样，这个 flag 让它们能区分
  - **Anthropic prompt caching**：`_call_anthropic` 把静态 system prompt 作为 `cache_control: ephemeral` 块发送，跨调用（尤其 tech_analyst 多 chunk）复用前缀，省钱 + 降延迟（延迟下来也缓解 300s timeout 压力）。prompt 不够长时自动 no-op，无副作用
  - `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` — 可选；都给上才启用 Telegram 推送，缺一个就静默 no-op。`TELEGRAM_DISABLED=1` 是 kill switch（不删 token 直接关）
  - `.env` 的必需项：`OPENAI_API_KEY` (当前主 provider，gpt-5.5) / `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` / `FRED_API_KEY`。**`ANTHROPIC_API_KEY` 现在是 failover key**——OpenAI/DeepSeek 调用彻底失败时自动切到 claude-opus-4-7(见上面 failover 段),所以生产环境应保持它有值;缺它则 failover 静默禁用(主调用失败直接抛错)。`DEEPSEEK_API_KEY` 可选——只在有 agent 配了 `deepseek-*` model 时才需要(否则 config 校验会报 `DEEPSEEK_API_KEY is required for selected DeepSeek models`)
- **长期反思（quarterly auto-evolution）**：每季末 `--mode meta` 让 `PromptEditor` 处理 `reflection.json` 的提案。**生效模式 = `enabled` × `dry_run` 两个开关的乘积，`PromptEditor.apply_reflection` 启动时会 `logger.warning` 大声打出当前模式**：`enabled:false` → OFF（只观察，不 stage）；`enabled:true dry_run:true` → **STAGE-ONLY**（提案写到 `data/evolution/{period}/proposed_edits.json` 供人工审，**不碰任何 prompt 文件**）；`enabled:true dry_run:false` → LIVE-APPLY（真实写入 6 个 editable agent 的 `## Learnings (system-evolved)` 段 + git commit）。**当前 `settings.yaml` 是 `enabled:true dry_run:true` = STAGE-ONLY**——这个循环目前是"季度自审报告"而非自动改 prompt 的工具（4 道护栏全是语法层的，分不出"对的 learning"和"措辞漂亮但错的 learning"，所以 live apply 暂不开）。要真应用：人工看 `proposed_edits.json`，那一次 run 临时 `dry_run:false` 再 `--mode meta --force`。`risk_manager` + `position_reviewer` 被 `MetaReflectionAgentName` schema literal 硬挡，任何模式都改不了。4 层护栏：FIFO cap / Jaccard dedup / prohibited-words regex / git auto-commit（`git revert <sha>` 一条命令整季回滚）
- **记忆**：操作员长期偏好 / 决策背景由 Claude Code 在本机 `~/.claude/projects/<project-hash>/memory/` 维护（每个 fork 独立）。Claude Code 文档：<https://docs.claude.com/en/docs/claude-code/memory>
