#!/bin/bash
set -e

# Run directory for Matrix A (Real EEG, decode_unit targets)
RUN_DIR="outputs/runs/20260525_005934_zuna_real_A_real"

if [ ! -d "$RUN_DIR" ]; then
    echo "Error: Directory $RUN_DIR does not exist."
    exit 1
fi

for k in 1 5 10 25; do
    for t in 0.03 0.05 0.1; do
        OUT="outputs/clip_native_eval_k${k}_t${t}.png"
        echo "Running evaluation for k=$k, temperature=$t..."
        python scripts/evaluate_clip_native_decoder.py \
            --run-dir "$RUN_DIR" \
            --k "$k" \
            --temperature "$t" \
            --output "$OUT"
    done
done

echo "Sweep complete! Results are in outputs/clip_native_eval_k*_t*.png"
