import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF
from diffusers import AutoencoderRAE
import math

def monkeypatch_dinov2_meta_init():
    """Patches Dinov2WithRegistersModel._init_weights to support meta-device initialization."""
    try:
        import transformers.models.dinov2_with_registers.modeling_dinov2_with_registers as dinov2_mod
    except ImportError:
        print("[RAEBackend] Dinov2WithRegistersModel not found in transformers library. Skipping patch.")
        return

    # Check if already patched to avoid recursive patching
    if getattr(dinov2_mod.Dinov2WithRegistersModel, "_is_patched_meta_init", False):
        return

    original_init = dinov2_mod.Dinov2WithRegistersModel._init_weights

    def patched_init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if module.weight is not None:
                w = module.weight.data
                if w.device.type == "meta":
                    cpu_tensor = torch.empty(w.shape, dtype=torch.float32)
                    nn.init.trunc_normal_(cpu_tensor, mean=0.0, std=self.config.initializer_range)
                    module.weight.data = cpu_tensor.to(device=w.device, dtype=w.dtype)
                else:
                    nn.init.trunc_normal_(w, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)
        elif isinstance(module, dinov2_mod.Dinov2WithRegistersEmbeddings):
            pe = module.position_embeddings.data
            if pe.device.type == "meta":
                cpu_tensor = torch.empty(pe.shape, dtype=torch.float32)
                nn.init.trunc_normal_(cpu_tensor, mean=0.0, std=self.config.initializer_range)
                module.position_embeddings.data = cpu_tensor.to(device=pe.device, dtype=pe.dtype)
            else:
                nn.init.trunc_normal_(pe, mean=0.0, std=self.config.initializer_range)

    dinov2_mod.Dinov2WithRegistersModel._init_weights = patched_init_weights
    dinov2_mod.Dinov2WithRegistersModel._is_patched_meta_init = True
    print("[RAEBackend] Successfully applied Dinov2WithRegistersModel._init_weights patch for meta-device initialization compatibility.")

class RaeDecoderBackend:
    def __init__(self, model_id: str = "nyu-visionx/RAE-dinov2-wReg-base-ViTXL-n08", device: str = "cuda", apply_patch: bool = True):
        self.device = device
        self.model_id = model_id
        self.apply_patch = apply_patch
        self.model = None

    def load(self):
        if self.model is not None:
            return self
            
        if self.apply_patch:
            monkeypatch_dinov2_meta_init()
            
        print(f"Loading AutoencoderRAE from {self.model_id}...")
        self.model = AutoencoderRAE.from_pretrained(
            self.model_id
        ).to(self.device)
        self.model.eval()
        
        # Disable gradients on RAE parameters since it is used as a frozen target/decoder
        for p in self.model.parameters():
            p.requires_grad = False
            
        print("AutoencoderRAE loaded successfully.")
        return self

    @property
    def dtype(self):
        self.load()
        return next(self.model.parameters()).dtype

    @torch.inference_mode()
    def extract_rae_latent(self, images: list[Image.Image] | Image.Image | torch.Tensor):
        """
        Extract RAE spatial latents and global embeddings.
        
        Args:
            images: PIL Image, list of PIL Images, or torch.Tensor in range [0, 1]
            
        Returns:
            dict containing:
                "tokens": Spatial latents of shape [B, 768, 16, 16]
                "global": Mean-pooled embedding of shape [B, 768]
                "unit": L2-normalized mean-pooled embedding of shape [B, 768]
                "norm": Vector norm of mean-pooled embedding of shape [B]
        """
        self.load()
        if isinstance(images, Image.Image):
            images = [images]
            
        if isinstance(images, list):
            # Preprocess PIL images to torch tensors
            tensors = [TF.to_tensor(img.convert("RGB")) for img in images]
            x = torch.stack(tensors).to(self.device, dtype=self.dtype)
        else:
            x = images.to(self.device, dtype=self.dtype)
            
        # Get raw spatial latents (B, C, H, W)
        tokens = self.model.encode(x).latent
        
        # Mean pool over spatial dimensions
        global_embed = tokens.mean(dim=[-2, -1])  # [B, C]
        
        # L2-normalize to get unit embedding
        unit_embed = F.normalize(global_embed, dim=-1)
        
        # Vector norm of the raw global embedding
        norm = torch.linalg.vector_norm(global_embed, dim=-1)
        
        return {
            "tokens": tokens,
            "global": global_embed,
            "unit": unit_embed,
            "norm": norm
        }

    def decode_differentiable(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode RAE spatial latents to an image tensor while preserving gradients.

        Unlike ``generate_from_embeds`` (which is wrapped in ``inference_mode`` and
        returns PIL images), this keeps the autograd graph intact so a loss on the
        decoded image can backprop into the ``latents`` (and therefore the QFormer).
        The RAE itself stays frozen (its parameters have ``requires_grad=False``);
        gradients only flow through it to the input tokens.

        Args:
            latents: RAE spatial latents of shape [B, 768, 16, 16], requires_grad.

        Returns:
            image tensor of shape [B, 3, H, W] in [0, 1] (float, differentiable).
        """
        self.load()
        latents = latents.to(self.device, dtype=self.dtype)
        sample = self.model.decode(latents).sample
        return sample.clamp(0.0, 1.0)

    @torch.inference_mode()
    def generate_from_embeds(self, latents: torch.Tensor) -> list[Image.Image]:
        """
        Generate images from RAE spatial latents.
        
        Args:
            latents: RAE spatial latents of shape [B, 768, 16, 16]
            
        Returns:
            images: list of PIL Images (each 256x256)
        """
        self.load()
        latents = latents.to(self.device, dtype=self.dtype)
        outputs = self.model.decode(latents)
        sample = outputs.sample
        
        # Postprocess: clamp to [0, 1], scale to 0-255, and convert to PIL images
        x_rec = sample.clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)
        
        images = []
        for img_arr in x_rec:
            img_uint = (img_arr * 255.0).round().astype("uint8")
            images.append(Image.fromarray(img_uint))
            
        return images
