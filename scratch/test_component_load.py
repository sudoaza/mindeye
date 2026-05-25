import os
import torch
import gc
import numpy as np
from PIL import Image

from diffusers import QwenImagePipeline, QwenImageTransformer2DModel, AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor, BitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig

def print_vram(step_name):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[{step_name}] VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print_vram("Start")
    
    from transformers import Qwen2VLImageProcessor, Qwen2VLVideoProcessor
    
    # 1. Load CPU/Config components
    print("\nLoading tokenizer, processor, scheduler...")
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen-Image", subfolder="tokenizer")
    image_processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    video_processor = Qwen2VLVideoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    processor = Qwen2VLProcessor(
        image_processor=image_processor,
        video_processor=video_processor,
        tokenizer=tokenizer
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained("Qwen/Qwen-Image", subfolder="scheduler")
    print_vram("After config components")
    
    # 2. Load VAE in float16 directly to GPU
    print("\nLoading VAE...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        "Qwen/Qwen-Image", 
        subfolder="vae", 
        torch_dtype=torch.float16
    ).to("cuda")
    print_vram("After VAE")
    
    # 3. Load text_encoder entirely on CPU in bfloat16 to save all GPU VRAM
    print("\nLoading Text Encoder (Qwen2.5-VL) on CPU...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    print_vram("After Text Encoder")
    
    # Force clean cache
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After Text Encoder GC")
    
    # 4. Load transformer in 8-bit using bitsandbytes quantizer config dict
    print("\nLoading Transformer (QwenImageTransformer2DModel) in 8-bit...")
    quantization_config = {
        "quant_method": "bitsandbytes",
        "load_in_8bit": True,
        "llm_int8_enable_fp32_cpu_offload": True
    }
    
    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="transformer",
        quantization_config=quantization_config,
        device_map="balanced",
        max_memory={0: "6GiB", "cpu": "30GiB"},
        torch_dtype=torch.float16,
        offload_folder="/workspace/offload"
    )
    print_vram("After Transformer")
    
    # Force clean cache
    gc.collect()
    torch.cuda.empty_cache()
    print_vram("After Transformer GC")
    
    # 5. Assemble pipeline
    print("\nAssembling QwenImagePipeline...")
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer
    )
    print("Pipeline assembled successfully!")
    print_vram("After Assembly")
    
    # 6. Execute smoke tests
    print("\nRunning extraction smoke test (using VL processor)...")
    dummy_img = Image.fromarray(np.uint8(np.random.rand(512, 512, 3) * 255))
    try:
        # Run extraction using the raw VLM/processor logic on CPU
        template = "describe the image"
        txt = [template.format("")]
        model_inputs = processor(
            text=txt,
            images=dummy_img,
            padding=True,
            return_tensors="pt"
        ).to("cpu")
        
        with torch.no_grad():
            outputs = text_encoder(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                pixel_values=model_inputs.get("pixel_values").to(dtype=torch.bfloat16),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            # Process outputs similarly to _get_qwen_prompt_embeds
            prompt_embeds = hidden_states
            attn_mask = model_inputs["attention_mask"]
            
        print(f"Extracted prompt_embeds shape: {prompt_embeds.shape}")
        print(f"Extracted attention mask shape: {attn_mask.shape}")
        
        # Generation smoke test on GPU
        print("\nRunning generation smoke test (5 steps)...")
        # Move inputs to cuda
        prompt_embeds = prompt_embeds.to("cuda", dtype=torch.float16)
        attn_mask = attn_mask.to("cuda")
        
        output_images = pipe(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=attn_mask,
            num_inference_steps=5,
            height=256,
            width=256,
            output_type="pil"
        ).images
        print("Success: Generated image!")
        img_np = np.array(output_images[0])
        print(f"Generated image max pixel value: {img_np.max()}")

if __name__ == "__main__":
    main()
