import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Ensure sys.path includes src/
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from diffusers import QwenImageEditPipeline
from mindseye.models.common_to_qwen_adapter import CommonToQwenAdapter

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load pipeline
    print("Loading QwenImageEditPipeline...")
    pipe = QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    
    # Create a dummy image for testing
    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    
    # 2. Check A: Extract embedding and check shapes
    print("\n--- Check A: Extraction smoke test ---")
    # preprocess image for the pipeline's expected format (tensor/list)
    prompt_embeds, attn_mask = pipe._get_qwen_prompt_embeds(
        prompt="",
        image=dummy_img,
        device=device,
        dtype=torch.float16
    )
    print(f"Extracted prompt_embeds shape: {prompt_embeds.shape}")
    print(f"Extracted attention mask shape: {attn_mask.shape}")
    print(f"Dtype: {prompt_embeds.dtype}")
    
    seq_len = prompt_embeds.shape[1]
    hidden_dim = prompt_embeds.shape[2]
    
    # Verify that pipeline accepts random embeddings
    print("\n--- Check A (cont.): Random embedding generation test ---")
    random_embeds = torch.randn((1, seq_len, hidden_dim), dtype=torch.float16, device=device)
    random_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
    
    try:
        # Generate with random embeds
        output_images = pipe(
            prompt_embeds=random_embeds,
            prompt_attention_mask=random_mask,
            num_inference_steps=5, # fast steps for smoke test
            height=256,
            width=256,
            output_type="pil"
        ).images
        print("Success: Generated image from random embeddings.")
        # Check if the generated image is not solid black
        img_np = np.array(output_images[0])
        if img_np.max() == 0:
            print("Warning: Generated image is solid black!")
        else:
            print(f"Generated image max pixel value: {img_np.max()}")
    except Exception as e:
        print(f"Failed to generate from random embeddings: {e}")
        
    # 3. Check B: Oracle target embeds generation test
    print("\n--- Check B: Oracle target embeds generation test ---")
    try:
        oracle_images = pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=attn_mask,
            num_inference_steps=15,
            height=512,
            width=512,
            output_type="pil"
        ).images
        print("Success: Generated image from oracle target embeddings.")
        oracle_images[0].save("oracle_verification_output.png")
        print("Saved oracle generation to 'oracle_verification_output.png'")
    except Exception as e:
        print(f"Failed to generate from oracle embeddings: {e}")
        
    # 4. Check C: Overfit one image
    print("\n--- Check C: Overfit one image ---")
    # Mock a z_common embedding
    z_common = torch.randn((1, 512), dtype=torch.float32, device=device)
    
    # We set up an adapter with correct dimensions
    adapter = CommonToQwenAdapter(
        common_dim=512,
        adapter_dim=2048,
        num_tokens=seq_len,
        qwen_hidden_dim=hidden_dim,
        dropout=0.0
    ).to(device).to(torch.float16)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    
    # We want to overfit z_common to match target prompt_embeds
    print("Starting overfit training loop (100 steps)...")
    for step in range(101):
        adapter.train()
        pred_embeds = adapter(z_common)
        
        # Loss: MSE between predicted and target embeds
        loss = F.mse_loss(pred_embeds, prompt_embeds)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(f"Step {step:3d} | Loss: {loss.item():.6f}")
            
    print("Overfit complete. Generating from overfitted adapter embeddings...")
    adapter.eval()
    with torch.no_grad():
        fitted_embeds = adapter(z_common)
        
    try:
        fitted_images = pipe(
            prompt_embeds=fitted_embeds,
            prompt_attention_mask=attn_mask,
            num_inference_steps=15,
            height=512,
            width=512,
            output_type="pil"
        ).images
        fitted_images[0].save("overfit_verification_output.png")
        print("Saved overfit generation to 'overfit_verification_output.png'")
    except Exception as e:
        print(f"Failed to generate from overfitted embeds: {e}")

if __name__ == "__main__":
    main()
