import torch
import torch.nn as nn
from diffusers import QwenImagePipeline
from PIL import Image

class QwenBackend:
    """
    Wrapper for Qwen-Image generation pipeline using continuous prompt_embeds.
    """
    def __init__(self, model_id: str = "Qwen/Qwen-Image", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        
        # Load the pipeline
        # QwenImage requires transformer, text_encoder, etc. We load it in float16.
        self.pipe = QwenImagePipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            safety_checker=None
        ).to(device)
        
        # Freeze all components
        self.pipe.set_progress_bar_config(disable=True)
        for param in self.pipe.transformer.parameters():
            param.requires_grad = False
        if hasattr(self.pipe, "text_encoder") and self.pipe.text_encoder is not None:
            for param in self.pipe.text_encoder.parameters():
                param.requires_grad = False
        if hasattr(self.pipe, "vae") and self.pipe.vae is not None:
            for param in self.pipe.vae.parameters():
                param.requires_grad = False
                
        # Discover qwen_hidden_dim dynamically from text_encoder or transformer config
        if hasattr(self.pipe.transformer.config, "cross_attention_dim"):
            self.qwen_hidden_dim = self.pipe.transformer.config.cross_attention_dim
        elif hasattr(self.pipe.transformer.config, "joint_attention_dim"):
            self.qwen_hidden_dim = self.pipe.transformer.config.joint_attention_dim
        else:
            # Fallback to a common dimension for Qwen-Image / MMDiT
            self.qwen_hidden_dim = 4096 
            
    def generate_from_embeds(
        self, 
        prompt_embeds: torch.Tensor, 
        prompt_embeds_mask: torch.Tensor = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024
    ):
        """
        Generates images directly from soft conditioning tokens.
        
        Args:
            prompt_embeds: [B, num_tokens, qwen_hidden_dim]
            prompt_embeds_mask: [B, num_tokens]
        """
        # Ensure mask is provided if the pipeline expects it
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                (prompt_embeds.shape[0], prompt_embeds.shape[1]),
                dtype=torch.int32,
                device=prompt_embeds.device
            )
            
        # Diffusers pipelines often expect specific kwargs for embeds
        # The exact kwargs depend on the QwenImagePipeline implementation.
        # Usually it's `prompt_embeds` and `prompt_attention_mask`.
        
        # We wrap in a generic call, inspecting the signature or just passing them.
        outputs = self.pipe(
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_embeds_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            output_type="pil"
        )
        
        return outputs.images
