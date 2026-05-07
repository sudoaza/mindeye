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

    # Stage dirs
    fif_filter_dir = os.path.join(working_dir, "1_fif_filter")
    pt_input_dir = os.path.join(working_dir, "2_pt_input")
    pt_output_dir = os.path.join(working_dir, "3_pt_output")
    fif_output_dir = os.path.join(working_dir, "4_fif_output")

    for d in [fif_filter_dir, pt_input_dir, pt_output_dir, fif_output_dir]:
        os.makedirs(d, exist_ok=True)

    fif_files = glob.glob(os.path.join(input_fif_dir, "*.fif"))
    if not fif_files:
        raise FileNotFoundError(f"No .fif files in {input_fif_dir}")
    print(f"Found {len(fif_files)} .fif files to process")

    # Step 1: Preprocess (resample 256Hz, filter, epoch 5s, normalize)
    print("\n=== ZUNA Preprocessing ===")
    preprocess_kwargs = dict(
        input_dir=input_fif_dir,
        output_dir=pt_input_dir,
        apply_notch_filter=False,
        apply_highpass_filter=True,
        apply_average_reference=True,
        preprocessed_fif_dir=fif_filter_dir,
    )
    if target_channels:
        preprocess_kwargs["target_channel_count"] = target_channels
    if bad_channels:
        preprocess_kwargs["bad_channels"] = bad_channels

    preprocessing(**preprocess_kwargs)

    # Step 2: Inference (denoise / reconstruct / upsample)
    print("\n=== ZUNA Inference ===")
    inference(
        input_dir=pt_input_dir,
        output_dir=pt_output_dir,
        gpu_device=gpu_device,
        data_norm=data_norm,
        diffusion_sample_steps=diffusion_steps,
    )

    # Step 3: Convert back to .fif
    print("\n=== ZUNA Reconstruction -> .fif ===")
    pt_to_fif(
        input_dir=pt_output_dir,
        output_dir=fif_output_dir,
    )

    output_files = glob.glob(os.path.join(fif_output_dir, "*.fif"))
    print(f"\n Done. {len(output_files)} output .fif files in {fif_output_dir}")
    return fif_output_dir
