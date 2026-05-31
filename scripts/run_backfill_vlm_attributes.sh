#!/usr/bin/env bash
# Backfill 11 Phase 11A calibration VLM attributes into the multi-subject JSON bank.
# Does not re-annotate Tier-1 keys when they already exist (--tier calibration --merge).
#
# See docs/VLM_ATTRIBUTES.md for full procedure and probe retrain notes.
#
# Run on RunPod:
#   bash scripts/run_backfill_vlm_attributes.sh

set -e
cd /workspace/mindeye
source venv/bin/activate
export PYTHONPATH=src
export HF_HOME=/workspace/hf_cache
export PYTHONUNBUFFERED=1

METADATA="data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub02_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub03_runs01_40/all_runs_metadata.csv,data/processed/semantic_epochs/zuna_tight1s_sub04_runs01_40/all_runs_metadata.csv"
IMAGE_DIR="data/raw/nod/stimuli/ImageNet"
VLM_JSON="outputs/common_probe/vlm_attributes_runs01_40.json"

mkdir -p outputs/common_probe outputs/vlm_audit

echo "=== Pre-backfill audit ==="
python3 scripts/analyze_vlm_attributes.py \
  --vlm-json "$VLM_JSON" \
  --metadata "$METADATA" \
  --output-dir outputs/vlm_audit/pre_backfill

if [ ! -f "$VLM_JSON" ]; then
  echo ""
  echo "=== No existing JSON — running full Tier-1 + calibration (tier=all) ==="
  python3 scripts/generate_vlm_attributes.py \
    --tier all \
    --metadata "$METADATA" \
    --image-dir "$IMAGE_DIR" \
    --output "$VLM_JSON" \
    --batch-size 4
else
  echo ""
  echo "=== Backfill calibration tier only (merge) ==="
  python3 scripts/generate_vlm_attributes.py \
    --tier calibration \
    --merge \
    --metadata "$METADATA" \
    --image-dir "$IMAGE_DIR" \
    --output "$VLM_JSON" \
    --batch-size 4
fi

echo ""
echo "=== Post-backfill audit ==="
python3 scripts/analyze_vlm_attributes.py \
  --vlm-json "$VLM_JSON" \
  --metadata "$METADATA" \
  --output-dir outputs/vlm_audit/post_backfill

echo ""
echo "=== Backfill complete ==="
echo "  JSON:   $VLM_JSON"
echo "  Audit:  outputs/vlm_audit/post_backfill/vlm_audit_report.json"
echo ""
echo "Optional: retrain RAE-code probe with updated labels:"
echo "  python3 scripts/pretrain_common_probe.py \\"
echo "    --metadata \"$METADATA\" \\"
echo "    --common-embeddings data/processed/rae_embeddings/rae_bottleneck_codes_4x4.pt \\"
echo "    --vlm-attributes $VLM_JSON \\"
echo "    --target-key rae_code --spatial-pool \\"
echo "    --output-dir outputs/rae_code_probe_4x4_v2 --device cuda"
