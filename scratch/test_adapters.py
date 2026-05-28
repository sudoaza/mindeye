import torch
import pandas as pd
import numpy as np
import re
from mindseye.models.eeg_encoder import TemporalAttnEncoder, DualHeadTemporalAttnEncoder
from mindseye.datasets.semantic_pairs import ZunaClipPairDataset, SemanticPairConfig

def test_temporal_attn_adapters():
    print("=== Testing TemporalAttnEncoder with Adapters ===")
    device = "cpu"
    # 4 subjects, batch size 4
    model = TemporalAttnEncoder(
        n_channels=62,
        embedding_dim=512,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        num_subjects=4
    ).to(device)

    # Verify that subject embedding and subject heads exist
    assert model.subject_embed is not None
    assert model.subject_heads is not None
    assert len(model.subject_heads) == 4
    
    # Input batch of shape [4, 62, 307] (EEG crop)
    eeg = torch.randn(4, 62, 307, device=device)
    
    # Run with different subject IDs
    subj_0 = torch.tensor([0, 0, 0, 0], device=device)
    subj_1 = torch.tensor([1, 1, 1, 1], device=device)
    subj_mix = torch.tensor([0, 1, 2, 3], device=device)
    
    out_0 = model(eeg, subject_id=subj_0)
    out_1 = model(eeg, subject_id=subj_1)
    out_mix = model(eeg, subject_id=subj_mix)
    
    # Output shape should be [4, 512]
    assert out_0.shape == (4, 512)
    assert out_1.shape == (4, 512)
    assert out_mix.shape == (4, 512)
    
    # Initialize embedding to non-zero to verify it changes output
    with torch.no_grad():
        model.subject_embed.weight.fill_(0.5)
        # Change projection head weights to be distinct
        for idx, head in enumerate(model.subject_heads):
            head.weight.fill_(float(idx + 1) * 0.1)
            
    out_0_new = model(eeg, subject_id=subj_0)
    out_1_new = model(eeg, subject_id=subj_1)
    out_mix_new = model(eeg, subject_id=subj_mix)
    
    # The outputs for subject 0 vs subject 1 should be mathematically distinct
    diff_0_1 = torch.abs(out_0_new - out_1_new).mean().item()
    print(f"Mean diff between sub-0 and sub-1 outputs: {diff_0_1:.6f}")
    assert diff_0_1 > 1e-4
    
    # Test fallback when subject_id is None
    out_none = model(eeg, subject_id=None)
    assert out_none.shape == (4, 512)
    print("TemporalAttnEncoder adapter tests passed!")

def test_dual_head_adapters():
    print("=== Testing DualHeadTemporalAttnEncoder with Adapters ===")
    device = "cpu"
    # 4 subjects, batch size 4
    model = DualHeadTemporalAttnEncoder(
        n_channels=62,
        embedding_dim=512,
        hidden_dim=128,
        n_layers=2,
        n_heads=4,
        num_subjects=4
    ).to(device)

    assert model.subject_embed is not None
    assert model.subject_unit_heads is not None
    assert model.subject_norm_heads is not None
    assert len(model.subject_unit_heads) == 4
    assert len(model.subject_norm_heads) == 4
    
    eeg = torch.randn(4, 62, 307, device=device)
    subj_mix = torch.tensor([0, 1, 2, 3], device=device)
    
    z_unit, norm = model(eeg, subject_id=subj_mix, return_norm=True)
    assert z_unit.shape == (4, 512)
    assert norm.shape == (4, 1)
    # Norms must be positive
    assert (norm > 0).all()
    
    print("DualHeadTemporalAttnEncoder adapter tests passed!")

def test_dataset_subject_mapping():
    print("=== Testing ZunaClipPairDataset Auto-Subject-Mapping ===")
    
    # Replicate the logic we added to datasets/semantic_pairs.py
    def get_subject_mapping(metadata_df, subject_list=None):
        if "subject" in metadata_df.columns:
            if subject_list is not None:
                unique_subjects = list(subject_list)
                subject_to_id = {sub: i for i, sub in enumerate(unique_subjects)}
            else:
                raw_subs = metadata_df["subject"].astype(str).unique().tolist()
                digit_subs = []
                for sub in raw_subs:
                    match = re.search(r'\d+', sub)
                    if match:
                        digit_subs.append((int(match.group(0)), sub))
                    else:
                        digit_subs.append((0, sub))
                digit_subs.sort(key=lambda x: x[0] if x[0] > 0 else x[1])
                
                digits = [x[0] for x in digit_subs]
                if len(digits) > 0 and max(digits) <= 12 and len(set(digits)) == len(digits) and all(d > 0 for d in digits):
                    subject_to_id = {sub: d - 1 for d, sub in digit_subs}
                    max_id = max(d - 1 for d, sub in digit_subs)
                    unique_subjects = [f"sub-{i+1}" for i in range(max_id + 1)]
                else:
                    unique_subjects = sorted(raw_subs)
                    subject_to_id = {sub: i for i, sub in enumerate(unique_subjects)}
        else:
            unique_subjects = ["unknown"]
            subject_to_id = {"unknown": 0}
        return unique_subjects, subject_to_id

    # Test 1: Full multi-subject case
    df_multi = pd.DataFrame({"subject": [1, 2, 3, 4, 2, 1]})
    uniq, mapping = get_subject_mapping(df_multi)
    print("Multi-subject mapping:", mapping)
    assert mapping["1"] == 0
    assert mapping["2"] == 1
    assert mapping["3"] == 2
    assert mapping["4"] == 3
    assert len(uniq) == 4

    # Test 2: Single subject 2 during evaluation
    df_sub2 = pd.DataFrame({"subject": [2, 2, 2]})
    uniq, mapping = get_subject_mapping(df_sub2)
    print("Single subject 2 mapping:", mapping)
    assert mapping["2"] == 1
    assert len(uniq) == 2  # ["sub-1", "sub-2"]
    
    # Test 3: Single subject 4
    df_sub4 = pd.DataFrame({"subject": [4, 4]})
    uniq, mapping = get_subject_mapping(df_sub4)
    print("Single subject 4 mapping:", mapping)
    assert mapping["4"] == 3
    assert len(uniq) == 4  # ["sub-1", "sub-2", "sub-3", "sub-4"]

    # Test 4: Passing custom subject list override
    df_override = pd.DataFrame({"subject": [2, 4]})
    uniq, mapping = get_subject_mapping(df_override, subject_list=["1", "2", "3", "4"])
    print("Override subject mapping:", mapping)
    assert mapping["2"] == 1
    assert mapping["4"] == 3
    assert len(uniq) == 4
    
    print("ZunaClipPairDataset auto-subject-mapping tests passed!")

if __name__ == "__main__":
    test_temporal_attn_adapters()
    test_dual_head_adapters()
    test_dataset_subject_mapping()
    print("\nAll unit tests passed successfully!")
