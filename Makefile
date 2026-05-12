.PHONY: setup smoke nod zuna crop crop-raw crop-resample clip train grid audit matrix simulate

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

smoke:
	. venv/bin/activate && python scripts/test_pipeline.py

nod:
	. venv/bin/activate && python scripts/download_nod.py --runs 1-5

zuna:
	. venv/bin/activate && python scripts/run_zuna_batch.py --runs 1-5

crop:
	. venv/bin/activate && python scripts/run_cropper.py --mode zuna --runs 1 2 3 4 5

crop-raw:
	. venv/bin/activate && python scripts/run_cropper.py --mode raw --runs 1 2 3 4 5

crop-resample:
	. venv/bin/activate && python scripts/run_cropper.py --mode resample --runs 1 2 3 4 5

clip:
	. venv/bin/activate && python scripts/generate_clip_embeddings.py

audit:
	. venv/bin/activate && python scripts/audit_zuna_timing.py

# Sprint 1: standard contrastive training (ZUNA crops, run-held-out, 80 epochs)
train:
	. venv/bin/activate && python scripts/train_eeg_clip.py \
		--loss contrastive \
		--center-clip \
		--split-mode run \
		--val-runs 5 \
		--epochs 80 \
		--input-domain zuna \
		--target-mode real

grid:
	. venv/bin/activate && python scripts/make_retrieval_grid.py

# Sprint 2: run all 6 baseline-matrix conditions
# raw_runheldout and resample_runheldout require crop-raw / crop-resample to have been run first.
matrix:
	. venv/bin/activate && python scripts/run_baseline_matrix.py \
		--epochs 30 \
		--epochs-dir-raw data/processed/semantic_epochs/raw_sub-01_runs0102030405 \
		--epochs-dir-resample data/processed/semantic_epochs/resample_sub-01_runs0102030405

matrix-full5s:
	. venv/bin/activate && python scripts/run_baseline_matrix.py \
		--window-mode full5s \
		--semantic-target image_text \
		--text-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
		--model temporal_attn \
		--runs 1 2 3 4 5 6 7 8 9 10 \
		--val-runs 10 \
		--epochs 50 \
		--slug full5s_recovery

matrix-full5s-back:
	. venv/bin/activate && python scripts/run_baseline_matrix.py \
		--window-mode full5s_backaligned \
		--add-event-marker \
		--semantic-target image_text \
		--text-embeddings data/processed/clip_embeddings/imagenet_text_embeddings.pt \
		--model temporal_attn \
		--val-runs 8 \
		--epochs 50 \
		--batch-size 64 \
		--device cuda \
		--slug full5s_backaligned_recovery

# Sprint 3: simulate EPOC-14 low-channel conditions for all runs
simulate:
	. venv/bin/activate && \
		for run in 1 2 3 4 5; do \
			python scripts/simulate_low_channel_zuna.py \
				--subject sub-01 --run $$run; \
		done
