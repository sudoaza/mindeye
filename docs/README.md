# MindEye Documentation Index

> **Start here.** This is the single source of truth that links every MindEye document.
> If a doc contradicts this index or [`HANDOVER.md`](HANDOVER.md), those two win.

## Current architecture in one line

**ZUNA** (frozen EEG foundation embedding) → **QFormer** (learned bridge) → **RAE / DINOv2** (reconstruction target). All GPU work runs on a **RunPod** pod via the **runpod MCP**; data and big weights live on a **network volume** (see infra doc).

## Where things run

- **Local dev machine**: no GPU. Editing code, docs, git.
- **RunPod pod**: all training, ZUNA inference, embedding builds, evaluation.
- **RunPod network volume**: persistent `data/`, `outputs/`, model cache — moved between pods (see [`INFRA.md`](INFRA.md)).

## Document map

| Doc | Purpose | Status |
|---|---|---|
| [`HANDOVER.md`](HANDOVER.md) | Current phase, architecture, resume-from-cold-start steps. **Authoritative on current state.** | ✅ Current |
| [`PLAN.md`](PLAN.md) | Phased roadmap and the way of work. History + live plan. | ✅ Current |
| [`INFRA.md`](INFRA.md) | RunPod + network-volume strategy: pod sizing, what lives on the volume, detach/reattach workflow. | ✅ Current |
| [`RunPod_SKILL.md`](RunPod_SKILL.md) | runpod MCP tool reference (list/get/start/stop/create/delete pod). | ✅ Current |
| [`CHEAT.md`](CHEAT.md) | Dev cheat sheet: pipeline commands, background execution, troubleshooting/pitfalls. | ✅ Current |
| [`VLM_ATTRIBUTES.md`](VLM_ATTRIBUTES.md) | VLM attribute schema + backfill (Qwen2-VL semantic labels). | ✅ Current |
| [`../scripts/README.md`](../scripts/README.md) | Script-by-script reference and canonical execution order. | ✅ Current |
| [`../README.md`](../README.md) | Project thesis + top-level overview. | ✅ Current |
| [`SPRINT2_ZUNA_TIGHT1S_RECOVERY_ANALYSIS.md`](SPRINT2_ZUNA_TIGHT1S_RECOVERY_ANALYSIS.md) | Historical analysis: tight1s recovery. | 📎 Archive |
| [`PHASE3_5_BACKALIGNED_SUMMARY.md`](PHASE3_5_BACKALIGNED_SUMMARY.md) | Historical analysis: back-aligned windows. | 📎 Archive |
| [`PHASE3_5_FULL5S_SUMMARY.md`](PHASE3_5_FULL5S_SUMMARY.md) | Historical analysis: full-5s windows. | 📎 Archive |

> There is intentionally **one** `CHEAT.md` (this `docs/CHEAT.md`). The former root-level `/CHEAT.md` (pod sizing) was folded into [`INFRA.md`](INFRA.md).

## Agent skills

Repeatable pod workflow is captured as a Cursor skill at [`../.cursor/skills/mindeye-pod/SKILL.md`](../.cursor/skills/mindeye-pod/SKILL.md) — "provision/start pod → attach volume → run → pull outputs → stop".

## The non-negotiables (see PLAN §2 and HANDOVER §7 for full list)

1. Full-set retrieval + paired bootstrap vs `real/shuffled/random` is the only honest gate.
2. ZUNA and RAE are frozen; only the QFormer bridge trains.
3. No RAE decoder / diffusion until the retrieval gate is consistently beaten.
4. `PYTHONPATH=src` before any `python scripts/` call on the pod.
5. Never `rsync` local→remote without pulling `outputs/` first.
