#!/bin/bash
# Ablation study: num_layers (multimodal + classification)
# Launches parallel training runs on different GPUs

set -e

# Configuration
DATA="all_complete_dataset.h5"
PRICES="actual_prices_clean.h5"
NUM_FOLDS=3
EPOCHS=5
BATCH_SIZE=128
SEQ_LEN=512
MAX_EVAL_DATES=30
SEED=42
MODEL_TYPE="multimodal"
PRED_MODE="classification"

# Layers to test and corresponding GPUs
LAYERS=(2 4 6 8)
GPUS=(0 1 2 3)

OUTPUT_DIR="checkpoints/ablation_layers"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "ABLATION: num_layers = ${LAYERS[*]}"
echo "GPUs: ${GPUS[*]}"
echo "Output: $OUTPUT_DIR"
echo "========================================"

# Launch all runs in parallel
pids=()
for i in "${!LAYERS[@]}"; do
    NLAYERS=${LAYERS[$i]}
    GPU=${GPUS[$((i % ${#GPUS[@]}))]}
    CKPT="$OUTPUT_DIR/layers_${NLAYERS}"

    echo "[GPU $GPU] Starting num_layers=$NLAYERS -> $CKPT"

    python -m training.walk_forward_training \
        --device cuda:$GPU \
        --no-preload \
        --data "$DATA" \
        --prices "$PRICES" \
        --model-type $MODEL_TYPE \
        --pred-mode $PRED_MODE \
        --num-folds $NUM_FOLDS \
        --epochs-per-fold $EPOCHS \
        --batch-size $BATCH_SIZE \
        --seq-len $SEQ_LEN \
        --num-layers $NLAYERS \
        --hidden-dim 256 \
        --gradient-accumulation-steps 3 \
        --max-eval-dates $MAX_EVAL_DATES \
        --seed $SEED \
        --num-workers 0 \
        --checkpoint-dir "$CKPT" \
        --no-compile \
        > "$CKPT.log" 2>&1 &

    pids+=($!)
done

echo ""
echo "Launched ${#pids[@]} training runs. Waiting..."
echo "Logs: $OUTPUT_DIR/*.log"
echo ""

# Wait for all to complete
for pid in "${pids[@]}"; do
    wait $pid || echo "Process $pid failed"
done

echo ""
echo "========================================"
echo "ABLATION COMPLETE"
echo "========================================"

# Extract and compare results
echo ""
echo "RESULTS SUMMARY"
echo "----------------------------------------"
printf "| %8s | %10s | %10s | %10s |\n" "Layers" "Val Loss" "Mean IC" "Mean IR"
echo "----------------------------------------"

for NLAYERS in "${LAYERS[@]}"; do
    CKPT="$OUTPUT_DIR/layers_${NLAYERS}"
    EVAL_JSON="$CKPT/principled_evaluation.json"

    if [ -f "$EVAL_JSON" ]; then
        # Extract metrics with python one-liner
        METRICS=$(python3 -c "
import json
with open('$EVAL_JSON') as f:
    data = json.load(f)
folds = data.get('fold_results', [])
if folds:
    ics = [f['mean_ic'] for f in folds if f.get('mean_ic')]
    irs = [f['ir'] for f in folds if f.get('ir')]
    mean_ic = sum(ics)/len(ics) if ics else 0
    mean_ir = sum(irs)/len(irs) if irs else 0
    print(f'{mean_ic:+.4f} {mean_ir:+.3f}')
else:
    print('N/A N/A')
" 2>/dev/null || echo "N/A N/A")

        IC=$(echo $METRICS | cut -d' ' -f1)
        IR=$(echo $METRICS | cut -d' ' -f2)

        # Get val loss from stats.log
        VAL_LOSS=$(grep -oP 'Val Loss:\s*\K[\d.]+' "$CKPT/stats.log" 2>/dev/null | tail -1 || echo "N/A")

        printf "| %8s | %10s | %10s | %10s |\n" "$NLAYERS" "$VAL_LOSS" "$IC" "$IR"
    else
        printf "| %8s | %10s | %10s | %10s |\n" "$NLAYERS" "FAILED" "-" "-"
    fi
done

echo "----------------------------------------"
