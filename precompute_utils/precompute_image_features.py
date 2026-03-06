import argparse
import os
from pathlib import Path
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from accelerate import Accelerator
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--order", type=str, required=True, choices=["unpaired", "paired"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)

    parser.add_argument("--dataroot", type=str, required=True, help="Path to the dataset root directory.")
    parser.add_argument("--phase", type=str, required=True, help="Phase of the dataset (e.g. train, test).")
    parser.add_argument("--category", type=str, required=True, choices=["all", "dresses", "upper_body", "lower_body"], help="Category of the dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["dresscode", "vitonhd"], help="Dataset to use.")
    parser.add_argument("--mask_type", type=str, default="bounding_box", help="Type of mask to use.")

    args = parser.parse_args()

    return args

def get_clip_image_embeds(image_encoder_large, image_encoder_bigG, image):
    image_embeds_large = image_encoder_large(image).image_embeds
    image_embeds_bigG = image_encoder_bigG(image).image_embeds
    return torch.cat([image_embeds_large, image_embeds_bigG], dim=1)

@torch.inference_mode()
def main(args):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vit_processing = CLIPImageProcessor()

    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype).to(device)
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype).to(device)
    
    outputlist = ['image', 'cloth', 'category', 'im_name', 'c_name']
    if args.category == "all":
        args.category = ["upper_body", "lower_body", "dresses"]
    else:
        args.category = [args.category]
        
    if args.dataset == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataroot,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=args.category,
            size=(args.height, args.width),
            mask_type=args.mask_type,
            )
    
    elif args.dataset == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataroot,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            size=(args.height, args.width),
            mask_type=args.mask_type,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    dataroot = test_dataloader.dataset.dataroot
    test_dataloader = accelerator.prepare(test_dataloader)

        
    print("Start extracting image embeddings")
    for idx, batch in enumerate(tqdm(test_dataloader)):
        vton_img = batch.get("image").to('cpu')
        vton_img = (vton_img+1) /2
        
        vton_img = vton_img.to(device=device)
        vton_img_vit = vit_processing(images=vton_img, return_tensors="pt").data['pixel_values']
        vton_img_vit = vton_img_vit.to(device=device)
        vton_img_embeds = get_clip_image_embeds(image_encoder_large, image_encoder_bigG, vton_img_vit)
        
        for i in range(vton_img.shape[0]):
            vton_name = batch["im_name"][i]
            vton_image_embeds = vton_img_embeds[i]
            if args.dataset == "dresscode":
                image_feat_path = Path(dataroot) / "vton_image_embeddings" / "CLIP_VIT_L_VIT_G_concat" / batch["category"][i] / f"{vton_name.split('.')[0]}.pt"
            elif args.dataset == "vitonhd":
                image_feat_path = Path(dataroot) / "vton_image_embeddings" / "CLIP_VIT_L_VIT_G_concat" / test_dataloader.dataset.phase / f"{vton_name.split('.')[0]}.pt"
            else: raise NotImplementedError("dataset not supported")
            if not image_feat_path.parent.exists():
                image_feat_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not image_feat_path.exists():
                torch.save(vton_image_embeds, image_feat_path)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)