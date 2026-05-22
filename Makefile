.PHONY: setup pipeline matrix simulate fold_replication

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

# Ablation: no-probe vs probe=0.03 on zuna_real + controls (fastest key comparison ~2-3h)
ablation:
	. venv/bin/activate && python -u scripts/run_baseline_matrix.py \
		--subjects sub-01 \
		--run-range 01_40 \
		--common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
		--val-runs 8 \
		--window-mode tight1s \
		--model temporal_attn_small \
		--epochs 50 \
		--batch-size 64 \
		--device cuda \
		--slug ablation_probe \
		--add-event-marker \
		--augment-eeg \
		--common-probe outputs/common_probe/common_probe.pt \
		--conditions zuna_real zuna_real_noprobe zuna_shuffled zuna_random

# Probe weight sweep: 0 / 0.01 / 0.03 / 0.05 / 0.10 on zuna_real (~5 runs, ~4-5h)
probe_sweep:
	. venv/bin/activate && python -u scripts/run_baseline_matrix.py \
		--subjects sub-01 \
		--run-range 01_40 \
		--common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
		--val-runs 8 \
		--window-mode tight1s \
		--model temporal_attn_small \
		--epochs 50 \
		--batch-size 64 \
		--device cuda \
		--slug probe_sweep \
		--add-event-marker \
		--augment-eeg \
		--common-probe outputs/common_probe/common_probe.pt \
		--probe-weights 0,0.01,0.03,0.05,0.10 \
		--conditions zuna_real

# Sprint 3: simulate EPOC-14 low-channel conditions for all runs
simulate:
	. venv/bin/activate && \
		for run in 1 2 3 4 5 6 7 8; do \
			python scripts/simulate_low_channel_zuna.py \
				--subject sub-01 --run $$run; \
		done

# Run fold replication sweep (5 folds x 3 probe weights)
fold_replication:
	. venv/bin/activate && python -u scripts/run_fold_replication.py \
		--metadata data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv \
		--epochs-dir data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40 \
		--common-embeddings data/processed/clip_embeddings/common_embeddings.pt \
		--common-probe outputs/common_probe/common_probe.pt \
		--epochs 50 \
		--batch-size 64 \
		--device cuda
