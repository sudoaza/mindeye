# VLM Attribute Annotation — Schema, Backfill, and Probe Integration

This document is the canonical reference for semantic attribute labels used by the frozen
`CommonProbeModel` auxiliary loss during EEG training (including Phase 18 RAE-code runs).

**Source of truth for class choices:** `src/mindseye/models/common_probe.py` → `ATTRIBUTE_SCHEMAS`.

---

## 1. Two separate “missing label” problems

Do not conflate these:

| Problem | Symptom | Cause | Fix |
|--------|---------|-------|-----|
| **A. Never annotated** | Keys absent from `vlm_attributes.json` for all images | Original `generate_vlm_attributes.py` prompt only listed **18 Tier-1** attrs; **11 calibration attrs** exist in `ATTRIBUTE_SCHEMAS` but were never sent to Qwen | `--tier calibration` backfill (see §4) |
| **B. Sparse / unclear** | ~25% probe “coverage” on val (Phase 18C tables) | VLM returns `"unclear"` on ~75% of images for many attrs → `IGNORE_INDEX` in training | Prompt tuning, optional re-annotate Tier-1; gate probe tasks by non-unclear rate |

**Image-level coverage** (image_id present in JSON) is separate from **task-level coverage**
(fraction of samples where label ≠ `unclear`).

---

## 2. Attribute registry (29 + class_label)

### Tier 1 — natural ImageNet semantics (18 attributes)

Used since early Phase 6; included in the original Qwen system prompt.

| Attribute | Classes |
|-----------|---------|
| `is_animate` | no, yes |
| `human_visible` | no, yes |
| `face_visible` | no, yes |
| `animal_visible` | no, yes |
| `indoor_outdoor` | indoor, outdoor, mixed |
| `natural_artificial` | natural, artificial, mixed |
| `scene_dominance` | isolated_object, object_with_background, full_scene |
| `real_world_size` | tiny, small, medium, large, huge |
| `dominant_color` | red … gray (12) |
| `main_subject_position_x` | left, center, right, full_frame |
| `subject_scale` | close_up, medium_shot, wide_shot |
| `soft_texture` | no, yes |
| `spiky_or_pointed` | no, yes |
| `furry` | no, yes |
| `metallic` | no, yes |
| `tool_like` | no, yes |
| `vehicle_like` | no, yes |
| `food_like` | no, yes |

### Phase 11A — calibration / material-shape axes (11 attributes)

Defined for the visual calibration battery; **not** in the original VLM generator prompt.
On natural ImageNet images these keys are usually **missing** or default to `unclear`.

| Attribute | Classes |
|-----------|---------|
| `warm_vs_cool` | warm, cool, neutral |
| `bright_vs_dark` | bright, dark, neutral |
| `round_or_curved` | no, yes |
| `angular_or_geometric` | no, yes |
| `symmetrical` | no, yes |
| `single_object` | no, yes |
| `glossy` | no, yes |
| `rough` | no, yes |
| `smooth` | no, yes |
| `transparent` | no, yes |
| `organic_texture` | no, yes |

### `class_label` (not from VLM)

- 1000-way ImageNet synset from metadata / embeddings bank.
- Probe may include it if it beats majority baseline on pretrain val split.
- **Not useful** for EEG code prediction at chance (~0.13% in 18C); keep low `probe_weight` (0.01).

---

## 3. How labels flow into training

```
image_id → vlm_attributes.json[attr → string]
         → CommonProbeModel.encode_label(attr, string)
         → IGNORE_INDEX (-100) if "unclear" / unknown / missing key
         → cross_entropy on frozen probe heads (probe_weight, probe_start_epoch)
```

**Phase 18 RAE:** probe input = `normalize(mean_pool(pred_code))`, shape `[B, 768]`.
Probe checkpoint: `outputs/rae_code_probe_4x4/common_probe.pt` + `task_specs.json` (active tasks only).

**Pretrain gating** (`scripts/pretrain_common_probe.py`):

- Trains all tasks; saves only heads where **val accuracy > majority baseline**.
- “11 active tasks” (Phase 18C probe) = 11 attributes passed gating, **not** “11 missing attrs”.

---

## 4. Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_vlm_attributes.py` | Qwen2-VL-2B batch annotation → JSON |
| `scripts/analyze_vlm_attributes.py` | Coverage audit: images, per-attr unclear %, missing keys |
| `scripts/run_backfill_vlm_attributes.sh` | RunPod orchestration: audit → calibration backfill → audit |
| `scripts/pretrain_common_probe.py` | Train probe; write `task_specs.json` (active tasks) |

### `generate_vlm_attributes.py` flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--tier` | `all` | `tier1` (18), `calibration` (11), or `all` (29) |
| `--merge` | on | Merge new keys into existing JSON (required for calibration backfill) |
| `--metadata` | required | Comma-separated metadata CSVs (all 4 subjects) |
| `--image-dir` | required | `data/raw/nod/stimuli/ImageNet` |
| `--output` | required | e.g. `outputs/common_probe/vlm_attributes_runs01_40.json` |

Resume behavior: re-processes images **missing any required key** for the selected tier.

### Recommended file paths

```text
outputs/common_probe/vlm_attributes_runs01_40.json   # canonical multi-subject bank
data/processed/clip_embeddings/vlm_attributes.json # legacy sub-01 only (avoid for new work)
```

Phase 18 run scripts prefer `outputs/common_probe/vlm_attributes_runs01_40.json` when present.

---

## 5. Backfill procedure (RunPod)

**Prerequisite:** Phase 18E loss fix merged; backfill does **not** block 18E rerun.

```bash
cd /workspace/mindeye && source venv/bin/activate
export PYTHONPATH=src HF_HOME=/workspace/hf_cache

# 1) Audit current JSON
python3 scripts/analyze_vlm_attributes.py \
  --vlm-json outputs/common_probe/vlm_attributes_runs01_40.json \
  --output-dir outputs/vlm_audit/pre_backfill

# 2) Backfill 11 calibration attributes only (merges into existing JSON)
bash scripts/run_backfill_vlm_attributes.sh

# 3) Post-audit
python3 scripts/analyze_vlm_attributes.py \
  --vlm-json outputs/common_probe/vlm_attributes_runs01_40.json \
  --output-dir outputs/vlm_audit/post_backfill
```

Or manually:

```bash
python3 scripts/generate_vlm_attributes.py \
  --tier calibration \
  --merge \
  --metadata "data/processed/semantic_epochs/zuna_tight1s_sub01_runs01_40/all_runs_metadata.csv,..." \
  --image-dir data/raw/nod/stimuli/ImageNet \
  --output outputs/common_probe/vlm_attributes_runs01_40.json \
  --batch-size 4
```

**GPU:** Qwen2-VL-2B; ~15.9k images × calibration tier ≈ several hours at batch 4.

---

## 6. After backfill: when to retrain probe

| Action | When |
|--------|------|
| **Keep** `rae_code_probe_4x4` | Fixed 18E rerun (probe_weight=0.01) |
| **Retrain** `pretrain_common_probe.py` on RAE codes | After backfill **and** you want new active tasks in probe loss |
| **Do not** add all 29 tasks to EEG loss at 0.01 each | Prefer high non-unclear coverage; gating already drops weak tasks |

Retrain example:

```bash
python3 scripts/pretrain_common_probe.py \
  --metadata "<4-subject CSVs>" \
  --common-embeddings data/processed/rae_embeddings/rae_bottleneck_codes_4x4.pt \
  --vlm-attributes outputs/common_probe/vlm_attributes_runs01_40.json \
  --target-key rae_code \
  --spatial-pool \
  --output-dir outputs/rae_code_probe_4x4_v2 \
  --epochs 30 --batch-size 128 --lr 1e-4 --device cuda
```

Then point Phase 18+ scripts at `outputs/rae_code_probe_4x4_v2/common_probe.pt`.

---

## 7. Probe task policy (what to add / remove)

### Do not remove from `ATTRIBUTE_SCHEMAS`

Keeps encode/decode consistent; missing keys → `unclear` → ignored.

### For RAE / EEG probe **active** set (via gating + manual review)

**Prefer keeping** (showed signal in 18C on predicted codes):

- `animal_visible`, `dominant_color`, `face_visible`, `furry`, `food_like`

**Deprioritize / expect drop from active set:**

- `class_label` — 1000-way, EEG at chance; oracle diagnostic only
- Any attribute with **< 30%** non-unclear rate in `analyze_vlm_attributes.py` report
- Calibration tier on natural ImageNet until backfill + audit confirms usable balance

### Do not enable all 29 at high weight

Incomplete labels should not dominate; `probe_weight=0.01`, `probe_start_epoch=5` remains the default for RAE phases.

---

## 8. Success criteria for backfill

| Check | Target |
|-------|--------|
| Image coverage | ≥ 99% of metadata `image_id`s present in JSON |
| Calibration keys | All 11 keys present for ≥ 99% of annotated images |
| Per-attr non-unclear | Report in audit; aspirational **> 40%** for attrs kept in active probe |
| Active probe tasks | ≥ 15 attribute tasks beat baseline after retrain (aspirational) |

---

## 9. Relation to roadmap

```text
Now:     Phase 18E rerun (expander loss fix) — use existing VLM + rae_code_probe_4x4
After:   Phase 18E gate
Then:    VLM calibration backfill (this doc) + optional probe v2
Later:   Phase 18F+ capacity (5×5 / full tokens) only if 18E gate fails
```

See also `docs/PLAN.md` § Phase 18F — VLM Backfill.
