#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="chronos-exp"
OUT_DIR="${1:-chronos_multimodel_results}"
EXPERIMENTS="${2:-confirmation budget barrier consistency}"
MODEL="${3:-gpt-5.6-terra}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo " ChronosAttack Experiment Runner (tmux)"
echo " Session:     $SESSION_NAME"
echo " Out Dir:     $OUT_DIR"
echo " Experiments: $EXPERIMENTS"
echo " Model:       $MODEL"
echo " Trials:      paper defaults"
echo "=================================================="

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists."
    echo "To view:  tmux attach -t $SESSION_NAME"
    echo "To kill:  tmux kill-session -t $SESSION_NAME"
    exit 0
fi

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd -- "$OUT_DIR" && pwd)"

# The single-quoted command is evaluated by the inner Bash process.
# shellcheck disable=SC2016
tmux new-session -d -s "$SESSION_NAME" -c "$SCRIPT_DIR" \
    bash -o pipefail -c '
        out_dir=$1
        read -r -a experiments <<< "$2"
        model=$3
        resume=()
        if [[ -f $out_dir/run_manifest.json ]]; then
            resume=(--resume)
        else
            shopt -s nullglob dotglob
            entries=("$out_dir"/*)
            if (( ${#entries[@]} )); then
                echo "Output directory is non-empty and has no run manifest."
                exit 2
            fi
        fi

        python3 chronos_multimodel_full.py \
            --providers codex \
            --codex-model "$model" \
            --experiments "${experiments[@]}" \
            --out "$out_dir" \
            "${resume[@]}" 2>&1 | tee "$out_dir/runner.log"
        statuses=("${PIPESTATUS[@]}")
        status=${statuses[0]}
        if (( status == 0 && statuses[1] != 0 )); then
            status=${statuses[1]}
        fi

        if (( status == 0 )); then
            echo "Experiments completed."
        else
            echo "Experiments failed with exit code $status."
        fi
        exit "$status"
    ' _ "$OUT_DIR" "$EXPERIMENTS" "$MODEL"

echo "Started experiment suite in detached tmux session '$SESSION_NAME'."
echo "• Monitor logs: tail -f $OUT_DIR/runner.log"
echo "• Attach session: tmux attach -t $SESSION_NAME"
