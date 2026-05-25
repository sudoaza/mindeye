import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableUnCLIPImg2ImgPipeline
import gc

class ClipNativeDecoderBackend:
    def __init__(self, model_id: str = "sd2-community/stable-diffusion-2-1-unclip", device: str = "cuda"):
        self.device = device
        self.model_id = model_id
        
        print(f"Loading StableUnCLIPImg2ImgPipeline from {model_id}...")
        self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        ).to(device)
        self.pipe.set_progress_bar_config(disable=False)
        
        print("Model loaded successfully.")

    @torch.no_grad()
    def extract_teacher_embeds(
        self,
        images: list[Image.Image] | Image.Image,
        normalize: bool = False
    ):
        """
        Extract teacher image_embeds from a PIL image or list of PIL images.
        Uses the exact feature_extractor and image_encoder attached to the pipeline.
        
        Returns:
            embeds: FloatTensor [B, hidden_dim]
        """
        if isinstance(images, Image.Image):
            images = [images]
            
        inputs = self.pipe.feature_extractor(images=images, return_tensors="pt").to(self.device)
        
        # diffusers CLIPVisionModelWithProjection returns image_embeds
        outputs = self.pipe.image_encoder(**inputs)
        embeds = outputs.image_embeds
        
        if normalize:
            embeds = F.normalize(embeds, dim=-1)
            
        return embeds

    def generate_from_embeds(
        self,
        image_embeds: torch.Tensor,
        num_inference_steps: int = 20,
        noise_level: int = 0,
        watermark: bool = True,
    ):
        """
        Generate images from image embeddings.
        Enforces empty prompt to neutralize text conditioning.
        """
        batch_size = image_embeds.shape[0]
        
        outputs = self.pipe(
            prompt=[""] * batch_size,
            image_embeds=image_embeds.to(self.device, dtype=self.pipe.dtype),
            noise_level=noise_level,
            guidance_scale=1.0, # Neutralize classifier-free guidance on the prompt
            num_inference_steps=num_inference_steps,
            output_type="pil",
        )
        
        images = outputs.images
        if not watermark:
            return images

        watermarked = []
        for img in images:
            img = img.copy()
            draw = ImageDraw.Draw(img)
            w, h = img.size
            bar_h = 30
            draw.rectangle([(0, h - bar_h), (w, h)], fill="black")
            draw.text(
                (10, h - bar_h + 8),
                "MindEye CLIP-native demo — not validated reconstruction",
                fill="white",
                font=ImageFont.load_default(),
            )
            watermarked.append(img)
        return watermarked
