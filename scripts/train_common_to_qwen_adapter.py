import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
from diffusers import QwenImagePipeline
from transformers import AutoProcessor

# Add src to python path if needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from mindseye.models.common_to_qwen_adapter import CommonToQwenAdapter

class EmbeddingDistillationDataset(Dataset):
    def __init__(self, metadata_path, stimuli_root, common_embeddings_path, processor):
        self.metadata = pd.read_csv(metadata_path)
        # Filter for unique images if it's trial-based metadata
        self.metadata = self.metadata.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
        self.stimuli_root = stimuli_root
        self.common_embeddings = torch.load(common_embeddings_path, map_location="cpu")
        self.processor = processor
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        # Depending on dataset, image_path might need construction
        if 'image_path' in row:
            image_path = os.path.join(self.stimuli_root, row['image_path'])
        else:
            image_path = os.path.join(self.stimuli_root, f"{image_id}.jpg")
            
        image = Image.open(image_path).convert("RGB")
        
        # We will resize to a fixed dimension to ensure consistent token count from Qwen2.5-VL
        image = image.resize((224, 224))
        
        # Qwen2.5-VL processor expects messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": ""} # empty text
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        
        # Remove batch dim
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.squeeze(0)
                
        # Common embedding
        z_common = self.common_embeddings[image_id]
        
        return inputs, z_common

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stimuli-root", required=True)
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image")
    parser.add_argument("--adapter-mode", default="soft_prompt")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Qwen Processor and Text Encoder
    print("Loading Qwen text encoder and processor...")
    pipe = QwenImagePipeline.from_pretrained(
        args.qwen_model, 
        text_encoder_only=True, # Custom flag if diffusers supports it, otherwise load full pipeline
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(args.device)
    
    text_encoder = pipe.text_encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    text_encoder.eval()
    
    # Load processor. Diffusers pipeline might not expose processor directly,
    # so we load it from the same model ID or the text_encoder's config.
    try:
        processor = AutoProcessor.from_pretrained(args.qwen_model)
    except:
        # Fallback to the underlying text_encoder model name if nested
        processor = AutoProcessor.from_pretrained(text_encoder.config._name_or_path)
    
    # 2. Dataset and Dataloader
    dataset = EmbeddingDistillationDataset(args.metadata, args.stimuli_root, args.common_embeddings, processor)
    
    # Use custom collate to handle variable length inputs if any
    def collate_fn(batch):
        # batch is list of (inputs_dict, z_common)
        keys = batch[0][0].keys()
        batched_inputs = {k: torch.stack([item[0][k] for item in batch]) for k in keys}
        z_commons = torch.stack([item[1] for item in batch])
        return batched_inputs, z_commons
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 3. Initialize Adapter
    # We dynamically find the qwen_hidden_dim
    qwen_hidden_dim = text_encoder.config.hidden_size
    
    # Since Qwen2.5-VL outputs a sequence of tokens, we will dynamically set num_tokens 
    # based on the first batch if args.num_tokens is a placeholder.
    # Actually, we can just use an adaptive pooling in the adapter or fix num_tokens.
    # We'll use the user's provided num-tokens.
    adapter = CommonToQwenAdapter(
        common_dim=512, 
        adapter_dim=1024, 
        num_tokens=args.num_tokens, 
        qwen_hidden_dim=qwen_hidden_dim
    ).to(args.device)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    
    # 4. Training Loop (Embedding Distillation)
    print("Starting embedding distillation training...")
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0.0
        
        for batch_inputs, z_common in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_inputs = {k: v.to(args.device) for k, v in batch_inputs.items()}
            z_common = z_common.to(args.device, dtype=torch.float32)
            
            with torch.no_grad():
                # Get target embeddings from Qwen text encoder
                # Depending on Qwen2.5-VL's API:
                outputs = text_encoder(**batch_inputs, output_hidden_states=True)
                # Usually we want the last hidden state
                target_embeds = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.last_hidden_state
                # target_embeds shape: [B, seq_len, qwen_hidden_dim]
                
            pred_embeds = adapter(z_common) # [B, num_tokens, qwen_hidden_dim]
            
            # Match tokens via truncation, interpolation, or attention
            # Since we just want simple MSE, we align sequences.
            # Easiest way: take the first N vision tokens (or pool).
            # The prompt text is empty, so most tokens are image patches.
            seq_len = target_embeds.shape[1]
            if pred_embeds.shape[1] != seq_len:
                # Interpolate pred_embeds to match target seq_len, or vice versa
                # It's better to force pred_embeds to match target_embeds shape
                # if we want to mimic the target exactly.
                pred_embeds = pred_embeds.permute(0, 2, 1) # [B, dim, num_tokens]
                pred_embeds = F.interpolate(pred_embeds, size=seq_len, mode='linear', align_corners=False)
                pred_embeds = pred_embeds.permute(0, 2, 1) # [B, seq_len, dim]
                
            loss = F.mse_loss(pred_embeds, target_embeds.to(torch.float32))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.4f}")
        
        # Save checkpoint
        torch.save(adapter.state_dict(), os.path.join(args.output_dir, f"adapter_epoch_{epoch+1}.pt"))

    print("Training complete.")

if __name__ == "__main__":
    main()
