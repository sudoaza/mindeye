import os
import sys
import json
import argparse
import pandas as pd
import mne

def audit_timing(subject, run, raw_path, zuna_path, events_path, crops_metadata_path):
    """
    Audits the timing and integrity of the ZUNA pipeline.
    """
    audit = {
        "subject": subject,
        "run": run,
        "status": "fail"
    }

    try:
        # 1. Load Raw and ZUNA FIF info
        raw = mne.io.read_raw_fif(raw_path, preload=False, verbose=False)
        zuna = mne.io.read_raw_fif(zuna_path, preload=False, verbose=False)

        audit["raw_sfreq"] = raw.info['sfreq']
        audit["zuna_sfreq"] = zuna.info['sfreq']
        audit["raw_duration_s"] = raw.times[-1]
        audit["zuna_duration_s"] = zuna.times[-1]
        audit["duration_delta_s"] = abs(audit["raw_duration_s"] - audit["zuna_duration_s"])

        # 2. Load Metadata
        events_df = pd.read_csv(events_path) if os.path.exists(events_path) else pd.DataFrame()
        if not events_df.empty and 'run' in events_df and 'session' in events_df:
            events_df = events_df[(events_df['run'] == run) & (events_df['session'] == 'ImageNet01')]
        crops_df = pd.read_csv(crops_metadata_path) if os.path.exists(crops_metadata_path) else pd.DataFrame()

        audit["n_stim_events_raw"] = len(events_df[events_df['trial_type'] == 'stimulus']) if 'trial_type' in events_df else len(events_df)
        audit["n_metadata_rows"] = len(events_df)
        audit["n_crops_written"] = len(crops_df)

        if not events_df.empty and 'onset' in events_df:
            audit["first_stim_onset_s"] = float(events_df['onset'].min())
            audit["last_stim_onset_s"] = float(events_df['onset'].max())
        else:
            audit["first_stim_onset_s"] = -1
            audit["last_stim_onset_s"] = -1

        if not crops_df.empty and 'start_s' in crops_df and 'end_s' in crops_df:
            audit["first_crop_start_s"] = float(crops_df['start_s'].min())
            audit["last_crop_end_s"] = float(crops_df['end_s'].max())
            audit["n_failed_boundary_crops"] = len(crops_df[crops_df['end_s'] > audit["zuna_duration_s"]])
        else:
            audit["first_crop_start_s"] = -1
            audit["last_crop_end_s"] = -1
            audit["n_failed_boundary_crops"] = 0

        # Hard failure checks
        failures = []
        if audit["duration_delta_s"] > 5.0:
            failures.append(f"Duration mismatch > 5.0s (raw: {audit['raw_duration_s']:.2f}, zuna: {audit['zuna_duration_s']:.2f}, delta: {audit['duration_delta_s']:.2f})")
        if audit["n_stim_events_raw"] != audit["n_metadata_rows"] and audit["n_stim_events_raw"] > 0:
            failures.append("Stim events != metadata rows")
        if audit["n_crops_written"] < 0.98 * audit["n_metadata_rows"] and audit["n_metadata_rows"] > 0:
            failures.append(f"Missing crops (wrote {audit['n_crops_written']}, expected {audit['n_metadata_rows']})")
        if audit["n_failed_boundary_crops"] > 0:
            failures.append("Crops out of bounds")
        if audit["zuna_sfreq"] != 256.0:
            failures.append(f"ZUNA sfreq is {audit['zuna_sfreq']}, expected 256")

        if failures:
            audit["failures"] = failures
        else:
            audit["status"] = "pass"

    except Exception as e:
        audit["failures"] = [str(e)]

    return audit

def main():
    parser = argparse.ArgumentParser(description="Audit ZUNA timing alignment.")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--runs", type=str, default="1,2,3,4,5")
    parser.add_argument("--out-dir", type=str, default="outputs/audits")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = [int(r) for r in args.runs.split(",")]
    
    all_audits = []
    failed = False
    
    for run in runs:
        # Placeholder paths based on typical project structure
        run_str = f"run-{run:02d}"
        raw_path = f"data/raw/nod/derivatives/preprocessed/raw/{args.subject}_ses-ImageNet01_task-ImageNet_{run_str}_eeg_clean.fif"
        zuna_path = f"data/processed/zuna_real_{args.subject.replace('-', '')}_runs01_05/{args.subject}_{run_str}_zuna.fif"
        events_path = f"data/raw/nod/derivatives/detailed_events/{args.subject}_events.csv"
        crops_path = f"data/processed/semantic_epochs/zuna_real_{args.subject.replace('-', '')}_runs01_05/{args.subject}_ses-ImageNet01_{run_str}_metadata.csv"
        
        # If actual data doesn't exist yet, we just log failure due to missing files.
        # This acts as a strict gate.
        print(f"Auditing {args.subject} {run_str}...")
        audit_res = audit_timing(args.subject, run, raw_path, zuna_path, events_path, crops_path)
        all_audits.append(audit_res)
        
        if audit_res["status"] != "pass":
            failed = True
            print(f"  ❌ FAILED: {audit_res.get('failures', [])}")
        else:
            print("  ✅ PASSED")

    summary_path = os.path.join(args.out_dir, f"zuna_timing_{args.subject.replace('-', '')}_runs{min(runs):02d}_{max(runs):02d}.json")
    with open(summary_path, "w") as f:
        json.dump(all_audits, f, indent=2)
        
    df = pd.DataFrame(all_audits)
    df.to_csv(summary_path.replace('.json', '.csv'), index=False)
    
    with open(summary_path.replace('.json', '_summary.md'), "w") as f:
        f.write(f"# ZUNA Timing Audit Summary\n\n")
        f.write(f"Subject: {args.subject}\n")
        f.write(f"Runs: {args.runs}\n")
        f.write(f"Status: {'FAIL ❌' if failed else 'PASS ✅'}\n\n")
        f.write(df.to_markdown())
        
    print(f"\nAudit completed. Results saved to {summary_path}")
    
    if failed:
        sys.exit(1)
        
if __name__ == "__main__":
    main()
