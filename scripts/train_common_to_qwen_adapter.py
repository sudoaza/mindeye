import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
from diffusers import QwenImageEditPipeline

# Add src to python path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from mindseye.models.common_to_qwen_adapter import CommonToQwenAdapter

class QwenTeacherDataset(Dataset):
    def __init__(self, metadata_path, stimuli_root, common_embeddings_path, pipe, num_tokens=256, device="cuda"):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata.drop_duplicates(subset=["image_id"]).reset_index(drop=True)
        self.stimuli_root = stimuli_root
        self.common_embeddings = torch.load(common_embeddings_path, map_location="cpu")
        self.pipe = pipe
        self.num_tokens = num_tokens
        self.device = device
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        if 'image_path' in row:
            image_path = os.path.join(self.stimuli_root, row['image_path'])
        else:
            image_path = os.path.join(self.stimuli_root, f"{image_id}.jpg")
            
        image = Image.open(image_path).convert("RGB")
        # Resize to fixed size to ensure relatively stable embedding shapes
        image = image.resize((512, 512))
        
        # 1. Extract target embeddings through pipeline's native conditioning pathway
        with torch.no_grad():
            prompt_embeds, attn_mask = self.pipe._get_qwen_prompt_embeds(
                prompt="",
                image=image,
                device=self.device,
                dtype=torch.float16
            )
            # Remove batch dimension from extraction
            prompt_embeds = prompt_embeds.squeeze(0).cpu() # [seq_len, dim]
            attn_mask = attn_mask.squeeze(0).cpu() # [seq_len]
            
        # 2. Handle truncation/padding to match fixed num_tokens
        seq_len = prompt_embeds.shape[0]
        dim = prompt_embeds.shape[1]
        
        if seq_len > self.num_tokens:
            prompt_embeds = prompt_embeds[:self.num_tokens]
            attn_mask = attn_mask[:self.num_tokens]
        elif seq_len < self.num_tokens:
            pad_len = self.num_tokens - seq_len
            pad_embeds = torch.zeros((pad_len, dim), dtype=prompt_embeds.dtype)
            prompt_embeds = torch.cat([prompt_embeds, pad_embeds], dim=0)
            pad_mask = torch.zeros((pad_len,), dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, pad_mask], dim=0)
            
        z_common = self.common_embeddings[image_id]
        
        return z_common, prompt_embeds, attn_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--common-embeddings", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--stimuli-root", required=True)
    parser.add_argument("--qwen-model", default="Qwen/Qwen-Image")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load pipeline (Teacher model)
    print(f"Loading teacher model {args.qwen_model}...")
    pipe = QwenImageEditPipeline.from_pretrained(
        args.qwen_model, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(args.device)
    
    pipe.set_progress_bar_config(disable=True)
    
    # Freeze all teacher weights
    for param in pipe.transformer.parameters():
        param.requires_grad = False
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
            
    # 2. Setup dataset and loader
    print("Preparing dataset (extracting native teacher prompt embeds)...")
    dataset = QwenTeacherDataset(
        metadata_path=args.metadata,
        stimuli_root=args.stimuli_root,
        common_embeddings_path=args.common_embeddings,
        pipe=pipe,
        num_tokens=args.num_tokens,
        device=args.device
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Discover embedding dim
    qwen_hidden_dim = pipe.transformer.config.cross_attention_dim if hasattr(pipe.transformer.config, "cross_attention_dim") else 4096
    
    # 3. Setup student adapter
    adapter = CommonToQwenAdapter(
        common_dim=512, 
        adapter_dim=args.adapter_dim, 
        num_tokens=args.num_tokens, 
        qwen_hidden_dim=qwen_hidden_dim,
        dropout=0.1
    ).to(args.device)
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    
    # 4. Train loop (MSE + Cosine Loss)
    print("Starting distillation training...")
    for epoch in range(args.epochs):
        adapter.train()
        total_loss = 0.0
        
        for z_common, target_embeds, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            z_common = z_common.to(args.device, dtype=torch.float32)
            target_embeds = target_embeds.to(args.device, dtype=torch.float32) # match float32 for loss
            masks = masks.to(args.device, dtype=torch.float32)
            
            # Predict soft conditioning tokens
            pred_embeds = adapter(z_common) # [B, num_tokens, qwen_hidden_dim]
            
            # Masked MSE loss
            masks_expanded = masks.unsqueeze(-1).expand_as(target_embeds)
            sum_mask = masks_expanded.sum()
            
            # Avoid division by zero
            if sum_mask > 0:
                mse_loss = F.mse_loss(pred_embeds * masks_expanded, target_embeds * masks_expanded, reduction='sum') / sum_mask
                
                # Masked Cosine Similarity loss
                cos_sim = F.cosine_similarity(pred_embeds, target_embeds, dim=-1) # [B, num_tokens]
                cosine_loss = 1.0 - ((cos_sim * masks).sum() / masks.sum())
            else:
                mse_loss = F.mse_loss(pred_embeds, target_embeds)
                cosine_loss = 1.0 - F.cosine_similarity(pred_embeds, target_embeds, dim=-1).mean()
                
            loss = mse_loss + cosine_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss / len(dataloader):.6f}")
        
        # Save checkpoint
        torch.save(adapter.state_dict(), os.path.join(args.output_dir, f"adapter_epoch_{epoch+1}.pt"))

    print("Training complete.")

if __name__ == "__main__":
    main()
