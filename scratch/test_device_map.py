import torch
from diffusers import QwenImageTransformer2DModel
from accelerate import infer_auto_device_map

def main():
    print("Loading config...")
    config = QwenImageTransformer2DModel.load_config("Qwen/Qwen-Image", subfolder="transformer")
    print("Instantiating model on meta device...")
    with torch.device("meta"):
        model = QwenImageTransformer2DModel.from_config(config)
    
    print("Inferring device map...")
    device_map = infer_auto_device_map(
        model, 
        max_memory={0: "2.5GiB", "cpu": "35GiB"},
        no_split_module_classes=model._no_split_modules
    )
    print("Generated device map:")
    for k, v in device_map.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
