#!/bin/bash
# Usage: ./run.sh morning|midday|evening

cd "$(dirname "$0")"

# Load API keys from .env
if [ -f .env ]; then
    source .env
fi

source .venv/bin/activate

MODE="${1:-morning}"
DATE=$(date +%Y-%m-%d)
LOGFILE="logs/${MODE}_${DATE}.log"

echo "$(date): === $MODE run started ===" >> "$LOGFILE"
python main.py --mode "$MODE" >> "$LOGFILE" 2>&1
EXIT_CODE=$?
echo "$(date): === $MODE run finished (exit=$EXIT_CODE) ===" >> "$LOGFILE"
