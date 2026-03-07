#!/bin/bash
# Generic ablation study launcher (multimodal + classification)
#
# Usage:
#   ./sanity_checks/ablate.sh num-layers "2 4 6" "0 1 2"
#   ./sanity_checks/ablate.sh lr "1e-5 5e-5 1e-4" "0 1 2"
#   ./sanity_checks/ablate.sh hidden-dim "256 512 1024" "0 1 2"
#
# Default config: multimodal model, classification mode, seq-len 512

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <param-name> \"<values>\" \"<gpus>\" [suffix]"
    echo ""
    echo "Examples:"
    echo "  $0 num-layers \"2 4 6\" \"0 1 2\""
    echo "  $0 seq-len \"256 512 1024\" \"0 1 2\" multimodal"
    echo "  $0 lr \"1e-5 5e-5 1e-4\" \"0 1 2\" ce_loss"
    exit 1
fi

PARAM_NAME=$1
VALUES=($2)
GPUS=($3)
SUFFIX=${4:-}  # Optional 4th arg for custom suffix (e.g., "multimodal")

if [ -n "$SUFFIX" ]; then
    OUTPUT_DIR="checkpoints/ablation_${PARAM_NAME//[^a-zA-Z0-9]/_}_${SUFFIX}"
else
    OUTPUT_DIR="checkpoints/ablation_${PARAM_NAME//[^a-zA-Z0-9]/_}"
fi
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "ABLATION: $PARAM_NAME"
echo "Values: ${VALUES[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Launch all runs in parallel
pids=()
for i in "${!VALUES[@]}"; do
    VALUE=${VALUES[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    VALUE_SAFE=$(echo "$VALUE" | tr -c 'a-zA-Z0-9.-' '_')
    CKPT="$OUTPUT_DIR/${PARAM_NAME}_${VALUE_SAFE}"

    echo "[GPU $GPU] Starting --$PARAM_NAME $VALUE -> $CKPT"

    python -m training.walk_forward_training \
        --device cuda:$GPU \
        --no-preload \
        --data all_complete_dataset.h5 \
        --prices actual_prices_clean.h5 \
        --model-type multimodal \
        --pred-mode classification \
        --seq-len 512 \
        --seed 4 \
        --lr 1e-4 \
        --hidden-dim 256 \
        --num-layers 4 \
        --gradient-accumulation-steps 3 \
        --batch-size 128 \
        --num-workers 0 \
        --no-compile \
        --"$PARAM_NAME" $VALUE \
        --checkpoint-dir "$CKPT" \
        > "$CKPT.log" 2>&1 &

    pids+=($!)
done

echo ""
echo "Launched ${#pids[@]} training runs. Waiting..."
echo "Logs: $OUTPUT_DIR/*.log"
echo "Monitor: tail -f $OUTPUT_DIR/*.log"
echo ""

# Wait for all to complete
failed=0
for pid in "${pids[@]}"; do
    if ! wait $pid; then
        echo "Process $pid failed"
        ((failed++))
    fi
done

echo ""
echo "========================================"
echo "ABLATION COMPLETE ($failed failed)"
echo "========================================"

# Extract and compare results
echo ""
echo "RESULTS SUMMARY"
echo "------------------------------------------------------------"
printf "| %15s | %10s | %10s | %10s |\n" "$PARAM_NAME" "Val Loss" "Mean IC" "Mean IR"
echo "------------------------------------------------------------"

for VALUE in "${VALUES[@]}"; do
    VALUE_SAFE=$(echo "$VALUE" | tr -c 'a-zA-Z0-9.-' '_')
    CKPT="$OUTPUT_DIR/${PARAM_NAME}_${VALUE_SAFE}"
    EVAL_JSON="$CKPT/principled_evaluation.json"

    if [ -f "$EVAL_JSON" ]; then
        METRICS=$(python3 -c "
import json
with open('$EVAL_JSON') as f:
    data = json.load(f)
folds = data.get('fold_results', [])
if folds:
    ics = [f['mean_ic'] for f in folds if f.get('mean_ic') is not None]
    irs = [f['ir'] for f in folds if f.get('ir') is not None]
    mean_ic = sum(ics)/len(ics) if ics else 0
    mean_ir = sum(irs)/len(irs) if irs else 0
    print(f'{mean_ic:+.4f} {mean_ir:+.3f}')
else:
    print('N/A N/A')
" 2>/dev/null || echo "N/A N/A")

        IC=$(echo $METRICS | cut -d' ' -f1)
        IR=$(echo $METRICS | cut -d' ' -f2)
        VAL_LOSS=$(grep -oP 'Val Loss:\s*\K[\d.]+' "$CKPT/stats.log" 2>/dev/null | tail -1 || echo "N/A")

        printf "| %15s | %10s | %10s | %10s |\n" "$VALUE" "$VAL_LOSS" "$IC" "$IR"
    else
        printf "| %15s | %10s | %10s | %10s |\n" "$VALUE" "FAILED" "-" "-"
    fi
done

echo "------------------------------------------------------------"
echo ""
echo "Detailed logs: $OUTPUT_DIR/"
