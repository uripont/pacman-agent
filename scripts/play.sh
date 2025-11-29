#!/bin/bash
source .venv/bin/activate

# Track game IDs automatically
COUNTER_FILE=".game_counter"

# Get the next game ID
if [ -f "$COUNTER_FILE" ]; then
    GAME_ID=$(cat "$COUNTER_FILE")
else
    GAME_ID=0
fi

# Increment and save for next time
NEXT_ID=$((GAME_ID + 1))
echo "$NEXT_ID" > "$COUNTER_FILE"

echo "Running game with ID: $GAME_ID"
python pacman-contest/runner.py -b my_team.py 2>/dev/null

deactivate
