"""
Offline ZUNA batch processing pipeline.
Wraps zuna.preprocessing() + zuna.inference() + zuna.pt_to_fif() for NOD-EEG .fif files.
"""
import os
import glob


def run_zuna_offline(
    input_fif_dir,
    working_dir,
    target_channels=None,
    bad_channels=None,
    gpu_device=0,
    diffusion_steps=50,
    data_norm=10.0,
):
    """
    Run the full ZUNA pipeline on all .fif files in input_fif_dir.

    Args:
        input_fif_dir: directory containing preprocessed .fif files
        working_dir: output directory for ZUNA pipeline stages
        target_channels: list of channel names to add/reconstruct (from 10-05 montage)
        bad_channels: list of known bad channels to zero out
        gpu_device: GPU ID or "" for CPU
        diffusion_steps: number of diffusion sampling steps (50=quality, 10-20=fast)
        data_norm: normalization denominator (ZUNA expects std~0.1)
    """
    from zuna import preprocessing, inference, pt_to_fif

    fif_output_dir = os.path.join(working_dir, "4_fif_output")
    os.makedirs(fif_output_dir, exist_ok=True)

    fif_files = glob.glob(os.path.join(input_fif_dir, "*.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files in {input_fif_dir}")
    print(f"Found {len(fif_files)} .fif files to process")

    import shutil
    import tempfile

    for fif_path in fif_files:
        base_name = os.path.basename(fif_path)
        expected_out = os.path.join(fif_output_dir, base_name)
        if os.path.exists(expected_out):
            print(f"\n--- Skipping {base_name} (already processed) ---")
            continue
            
        print(f"\n--- Processing {base_name} ---")
        
        with tempfile.TemporaryDirectory(dir=working_dir) as tmpdir:
            tmp_in = os.path.join(tmpdir, "in")
            tmp_1 = os.path.join(tmpdir, "1")
            tmp_2 = os.path.join(tmpdir, "2")
            tmp_3 = os.path.join(tmpdir, "3")
            tmp_4 = os.path.join(tmpdir, "4")
            for d in [tmp_in, tmp_1, tmp_2, tmp_3, tmp_4]:
                os.makedirs(d)
            
            shutil.copy(fif_path, tmp_in)
            
            preprocess_kwargs = dict(
                input_dir=tmp_in,
                output_dir=tmp_2,
                apply_notch_filter=False,
                apply_highpass_filter=True,
                apply_average_reference=True,
                preprocessed_fif_dir=tmp_1,
            )
            if target_channels:
                preprocess_kwargs["target_channel_count"] = target_channels
            if bad_channels:
                preprocess_kwargs["bad_channels"] = bad_channels

            print("=== Preprocessing ===")
            preprocessing(**preprocess_kwargs)

            print("=== Inference ===")
            inference(
                input_dir=tmp_2,
                output_dir=tmp_3,
                gpu_device=gpu_device,
                data_norm=data_norm,
                diffusion_sample_steps=diffusion_steps,
            )

            print("=== Reconstruction ===")
            pt_to_fif(
                input_dir=tmp_3,
                output_dir=tmp_4,
            )
            
            # Copy result to final output dir
            out_files = glob.glob(os.path.join(tmp_4, "*.fif"))
            for out_f in out_files:
                shutil.copy(out_f, fif_output_dir)

    output_files = glob.glob(os.path.join(fif_output_dir, "*.fif"))
    print(f"\n Done. {len(output_files)} output .fif files in {fif_output_dir}")
    return fif_output_dir

if __name__ == "__main__":
    import sys
    import yaml
    
    if len(sys.argv) < 2:
        print("Usage: python offline_pipeline.py <config.yaml>")
        sys.exit(1)
        
    config_path = sys.argv[1]
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    z_conf = config.get("zuna", {})
    run_zuna_offline(
        input_fif_dir=z_conf.get("input_dir", "data/raw/nod/derivatives/preprocessed/raw"),
        working_dir=z_conf.get("working_dir", "data/processed/zuna_output"),
        target_channels=z_conf.get("target_channels", None),
        bad_channels=z_conf.get("bad_channels", None),
        gpu_device=z_conf.get("gpu_device", 0),
        diffusion_steps=z_conf.get("diffusion_steps", 50),
        data_norm=z_conf.get("data_norm", 10.0),
    )
