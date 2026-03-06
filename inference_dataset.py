import argparse
import torch
from pathlib import Path
import os
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from src.transformer_vtoff import SD3Transformer2DModel
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_feature_extractor
from src.pipeline_stable_diffusion_3_tryoff_masked import StableDiffusion3TryOffPipelineMasked
from transformers import CLIPVisionModelWithProjection

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from tqdm import tqdm


def main(args):
    """
    Generate virtual try-off images using SD3 pipeline.
    It can be used to generate images from the DressCode or VITON-HD dataset.

    Args:
        args: Parsed command line arguments containing model paths,
              dataset configuration, and generation parameters.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path_sd3_tryoff, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    transformer_vton_feature_extractor = SD3Transformer2DModel_feature_extractor.from_pretrained(
        args.pretrained_model_name_or_path_sd3_tryoff, subfolder="transformer_vton", revision=args.revision, variant=args.variant
    )

    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device=device, dtype=weight_dtype)
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device=device, dtype=weight_dtype)

    pipeline = StableDiffusion3TryOffPipelineMasked(
        scheduler=noise_scheduler,
        vae=vae,
        transformer_vton_feature_extractor=transformer_vton_feature_extractor,
        transformer_garm=transformer,
        image_encoder_large=image_encoder_large,
        image_encoder_bigG=image_encoder_bigG,
        tokenizer=None,
        tokenizer_2=None,
        tokenizer_3=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
    )

    pipeline.to(device, dtype=weight_dtype)

    if not args.fine_mask:
        outputlist = ['category', 'image', 'im_name', 'vton_image_embeddings', 'clip_embeds', 't5_embeds', 'clip_pooled',
                      'inpaint_mask', 'im_mask']
    else:
        outputlist = ['category', 'image', 'im_name', 'vton_image_embeddings', 'clip_embeds', 't5_embeds', 'clip_pooled',
                      'parse_cloth', 'im_mask_fine']

    if args.dataset_name == "dresscode":
        if args.category == "all":
            args.category = ["upper_body", "lower_body", "dresses"]
        else:
            args.category = [args.category]
        dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=args.category,
            mask_type=args.mask_type,
            size=(args.height, args.width),
            coarse_caption_file=args.coarse_caption_file
        )
    elif args.dataset_name == "vitonhd":
        dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            size=(args.height, args.width),
            mask_type=args.mask_type,
            caption_file=args.coarse_caption_file
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    empty_string_pooled_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat_pooled" / "empty_string.pt", map_location="cpu").to(device, dtype=weight_dtype)
    empty_string_clip_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat" / "empty_string.pt", map_location="cpu").to(device, dtype=weight_dtype)
    empty_string_t5_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "T5_XXL" / "empty_string.pt", map_location="cpu").to(device, dtype=weight_dtype)
    negative_text_embeds = torch.cat([empty_string_clip_prompt_embeds, empty_string_t5_prompt_embeds], dim=1)
    negative_pooled_embeds = empty_string_pooled_prompt_embeds

    negative_text_embeds = negative_text_embeds.repeat(args.batch_size, 1, 1)
    negative_pooled_embeds = negative_pooled_embeds.repeat(args.batch_size, 1)

    for batch in tqdm(dataloader):
        image = batch.get("image").to(device=device)
        image_name = batch.get("im_name")
        image = (image+1)/2  #from [-1, 1] to [0, 1]
        clip_text_embeddings = batch.get("clip_embeds").to(device=device, dtype=weight_dtype)
        t5_text_embeddings = batch.get("t5_embeds").to(device=device, dtype=weight_dtype)
        clip_pooled_embeddings = batch.get("clip_pooled").to(device=device).to(dtype=weight_dtype)
        text_embeddings=torch.cat([clip_text_embeddings, t5_text_embeddings], dim=1)
        if not args.fine_mask:
            masked_vton_img = batch.get("im_mask").to(device=device)
            inpaint_mask = batch.get("inpaint_mask").to(device=device)
        else:
            masked_vton_img = batch.get("im_mask_fine").to(device=device)
            inpaint_mask = batch.get("parse_cloth").to(device=device)
        if len(inpaint_mask.shape) == 3:
            inpaint_mask = inpaint_mask.unsqueeze(1)
            
        masked_vton_img = masked_vton_img.to(dtype=weight_dtype)
        inpaint_mask = inpaint_mask.to(dtype=weight_dtype)
        
        generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
        
        images = pipeline(
            prompt_embeds = text_embeddings,
            pooled_prompt_embeds = clip_pooled_embeddings,
            negative_prompt_embeds = negative_text_embeds,
            negative_pooled_prompt_embeds = negative_pooled_embeds,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            vton_image=image,
            mask_input=inpaint_mask,
            image_input_masked=masked_vton_img,
        ).images
        
        for i in range(len(images)):
            category = batch.get("category")[i]
            os.makedirs(f"{os.path.join(args.output_dir, category)}", exist_ok=True)
            images[i].save(f"{os.path.join(args.output_dir, category)}/{image_name[i].split('.')[0]}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_sd3_tryoff",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["dresscode", "vitonhd"],
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--coarse_caption_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--order",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=42,
    )
    parser.add_argument(
        "--width",
        type=int,
        required=False,
        default=768,
    )
    parser.add_argument(
        "--height",
        type=int,
        required=False,
        default=1024,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16"],
        required=False,
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--fine_mask",
        action="store_true",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        required=False,
        default=2.0,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        required=False,
        default=28,
    )
    args = parser.parse_args()
    main(args)