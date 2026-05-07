"""
Smoke test for the ZUNA-first NOD-EEG pipeline.
Run after downloading NOD data: venv/bin/python scripts/test_pipeline.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mindseye.datasets.nod import NODLoader

NOD_ROOT = "data/raw/nod"


def test_nod_loader():
    print("=" * 60)
    print("1. NOD-EEG data availability")
    print("=" * 60)

    if not os.path.exists(NOD_ROOT):
        print(f"  Data dir {NOD_ROOT} does not exist yet.")
        print("  Run: venv/bin/python scripts/download_nod.py")
        return False

    loader = NODLoader(NOD_ROOT, subject="sub-01")
    loader.summary()
    return True


def test_epochs():
    print("\n" + "=" * 60)
    print("2. Epoch file inspection")
    print("=" * 60)

    loader = NODLoader(NOD_ROOT, subject="sub-01")
    try:
        epochs = loader.load_epochs()
        print(f"  Epochs loaded successfully!")
        print(f"  Shape: {epochs.get_data().shape} (trials, channels, samples)")
        print(f"  Sample rate: {epochs.info['sfreq']} Hz")
        print(f"  Channels: {epochs.info['nchan']}")
        print(f"  Channel names: {epochs.ch_names[:10]}...")
        print(f"  tmin={epochs.tmin}s  tmax={epochs.tmax}s")
        print(f"  Duration per epoch: {epochs.tmax - epochs.tmin:.3f}s")

        # Check montage
        montage = epochs.get_montage()
        print(f"  Montage: {'SET' if montage else 'MISSING'}")

        # Check metadata
        if epochs.metadata is not None:
            print(f"  Metadata columns: {list(epochs.metadata.columns)}")
            print(f"  Metadata rows: {len(epochs.metadata)}")
        else:
            print(f"  Metadata: None (use events CSV instead)")

        return True
    except FileNotFoundError as e:
        print(f"  {e}")
        return False
    except Exception as e:
        print(f"  Error loading epochs: {e}")
        return False


def test_events():
    print("\n" + "=" * 60)
    print("3. Events CSV")
    print("=" * 60)

    loader = NODLoader(NOD_ROOT, subject="sub-01")
    try:
        events = loader.load_events()
        print(f"  Rows: {len(events)}")
        print(f"  Columns: {list(events.columns)}")
        print(f"  Sessions: {sorted(events['session'].unique())}")
        print(f"  Runs: {sorted(events['run'].unique())}")
        print(f"  Unique images: {events['image_id'].nunique()}")
        print(f"  Unique classes: {events['class'].nunique()}")
        print(f"  Super classes: {sorted(events['super_class'].unique())}")
        print(f"\n  Head:\n{events.head()}")
        return True
    except FileNotFoundError as e:
        print(f"  {e}")
        return False


def test_stimulus_metadata():
    print("\n" + "=" * 60)
    print("4. Stimulus metadata")
    print("=" * 60)

    import pandas as pd
    class_info_path = os.path.join(NOD_ROOT, "stimuli", "metadata", "class_info.tsv")
    if os.path.exists(class_info_path):
        df = pd.read_csv(class_info_path, sep="\t")
        print(f"  class_info.tsv: {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        print(f"\n  Head:\n{df.head()}")
    else:
        print(f"  class_info.tsv: MISSING")


def test_zuna_compatibility():
    print("\n" + "=" * 60)
    print("5. ZUNA compatibility check")
    print("=" * 60)

    loader = NODLoader(NOD_ROOT, subject="sub-01")
    try:
        epochs = loader.load_epochs()
    except Exception:
        print("  Cannot load epochs — skipping")
        return

    montage = epochs.get_montage()
    sfreq = epochs.info["sfreq"]
    n_ch = epochs.info["nchan"]
    duration = epochs.tmax - epochs.tmin

    print(f"  Has montage with 3D positions: {'YES' if montage else 'NO — NEEDS FIX'}")
    print(f"  Sample rate: {sfreq} Hz (ZUNA will resample to 256)")
    print(f"  Channels: {n_ch}")
    print(f"  Epoch duration: {duration:.3f}s")
    print(f"  Note: ZUNA needs 5s continuous windows, not short epochs.")
    
    # Now let's check the continuous runs
    print("\n  Checking continuous raw runs for ZUNA...")
    runs = loader.list_runs()
    if not runs:
        print("  No continuous runs downloaded yet.")
    else:
        for run_path in runs:
            raw = loader.load_raw(run_path, preload=False)
            run_sfreq = raw.info['sfreq']
            run_duration = raw.times[-1]
            status = "✓ Ready for ZUNA" if run_duration > 5.0 else "✗ Too short"
            print(f"  Run: {os.path.basename(run_path)}")
            print(f"    Duration: {run_duration:.1f}s ({run_duration/5:.1f} ZUNA windows) -> {status}")
            print(f"    Sample rate: {run_sfreq} Hz")
            
    print(f"\n  Plan: download raw .fif runs -> ZUNA on continuous data -> crop around events")


if __name__ == "__main__":
    has_data = test_nod_loader()
    if has_data:
        test_epochs()
        test_events()
        test_stimulus_metadata()
        test_zuna_compatibility()
    print("\n✓ Pipeline test complete.")
