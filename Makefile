.PHONY: setup pipeline matrix simulate

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

pipeline:
	. venv/bin/activate && bash scripts/execute_recovery_v2.sh

# Run just the baseline matrix assuming prior steps (ZUNA, crop, embeddings) are completed
matrix:
	. venv/bin/activate && python -u scripts/run_baseline_matrix.py \
		--subjects sub-01,sub-02,sub-03,sub-04 \
		--run-range 01_40 \
		--common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
		--val-runs 8 \
		--window-mode tight1s \
		--model temporal_attn_small \
		--epochs 50 \
		--batch-size 64 \
		--device cuda \
		--slug common_space_sprint2 \
		--add-event-marker \
		--augment-eeg \
		--common-probe outputs/common_probe/common_probe.pt \
		--conditions zuna_real zuna_shuffled zuna_random

# Sprint 3: simulate EPOC-14 low-channel conditions for all runs
simulate:
	. venv/bin/activate && \
		for run in 1 2 3 4 5 6 7 8; do \
			python scripts/simulate_low_channel_zuna.py \
				--subject sub-01 --run $$run; \
		done
