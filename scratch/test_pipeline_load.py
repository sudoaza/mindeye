import os
import torch
import gc
import numpy as np
from PIL import Image
from diffusers import QwenImageEditPipeline

def print_vram(step_name):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print_vram("Start")
    
    print("Loading pipeline components on CPU first...")
    try:
        pipe = QwenImageEditPipeline.from_pretrained(
            "Qwen/Qwen-Image",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=None
        )
        print("Pipeline loaded successfully on CPU!")
        print("Enabling model CPU offloading...")
        pipe.enable_model_cpu_offload()
        print("Model CPU offloading enabled successfully!")
    except Exception as e:
        print(f"Pipeline loading failed with: {e}")
        return
        
    print_vram("After load & offload")
    
    # Create a dummy image for testing
    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    
    # Check A: Extract embedding and check shapes
    print("\n--- Check A: Extraction smoke test ---")
    try:
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
        
        output_images = pipe(
            prompt_embeds=random_embeds,
            prompt_attention_mask=random_mask,
            num_inference_steps=5, # fast steps for smoke test
            height=256,
            width=256,
            output_type="pil"
        ).images
        print("Success: Generated image from random embeddings.")
        img_np = np.array(output_images[0])
        print(f"Generated image max pixel value: {img_np.max()}")
    except Exception as e:
        print(f"Check A failed: {e}")

if __name__ == "__main__":
    main()
