# Security Policy

This project executes trades against a brokerage API and handles credentials for multiple LLM providers (Anthropic, OpenAI), a brokerage (Alpaca), and a data provider (FRED). Vulnerabilities here can have direct financial consequences for anyone running the system, so I'd appreciate responsible disclosure.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** Instead, email **fengyebo@gmail.com** with:

- A description of the issue and its potential impact
- Steps to reproduce, or a proof-of-concept
- The repository commit / version where you observed it
- Whether you'd like to be credited in the fix's release notes

I'll acknowledge receipt within ~7 days and aim to ship a fix or a public advisory within 30 days, depending on severity.

## In-scope vulnerabilities

Things especially worth flagging:

- **API-key handling** — anything that leaks `.env` contents to logs, tracebacks, agent prompts, or external services
- **Broker order paths** — silent qty rounding, allocation_pct edge cases, or path through `execution/broker.py` that submits unintended orders
- **Stop-protection bypass** — sequences where a SELL leaves a position naked of its protective stop (the five-step `cancel_protective_stops` → `submit` → `_order_accepted` → `wait_for_order_terminal` → `_finalize_protection_after_sell` chain has been the source of repeated bugs)
- **Hard-rule bypass** — ways to make `cash_only`, daily-loss circuit breaker, or sector concentration caps fail open
- **Prompt-injection vectors** — content sources (news headlines, SEC filings, web fetches) that could carry instructions the LLM might act on
- **Schema-validation evasion** — outputs that bypass per-entry isolation or hard caps like `target_weight_pct ≤ 25`

## Out of scope

- Markdown rendering quirks in the README
- Theoretical timing attacks against rate-limited public APIs (Alpaca/FRED/yfinance)
- Issues that require local file-system or shell access already at the user's privilege level
- Vulnerabilities in third-party packages — please report those upstream

## Known operational caveats (not bugs, by design)

- The `evolution` flag, when enabled, lets the meta-reflector append text to six agent prompt files. The 10-belt safety system in `src/evolution/prompt_editor.py` (allow-list / FIFO / Jaccard / prohibited-words / git auto-commit / etc.) is the security boundary; finding a way **past** it is in scope.
- Default config points at Alpaca paper trading. Switching to live trading is the operator's choice; no warranty applies (see `LICENSE`).
