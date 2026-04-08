#!/bin/bash
# Usage: ./run.sh morning|midday|evening
set -e

cd "$(dirname "$0")"

# Load API keys directly (avoid zshrc compatibility issues with bash)
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}"
export FRED_API_KEY="${FRED_API_KEY}"
export ALPACA_API_KEY="${ALPACA_API_KEY}"
export ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY}"

# If keys not in env, try sourcing from a dedicated env file
if [ -z "$ANTHROPIC_API_KEY" ] && [ -f .env ]; then
    source .env
fi

source .venv/bin/activate

MODE="${1:-morning}"
echo "$(date): Running $MODE..."
python main.py --mode "$MODE" >> "logs/${MODE}.log" 2>&1
echo "$(date): $MODE complete."
