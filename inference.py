import argparse
import torch
import logging
import os
import time
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from src.transformer_vtoff import SD3Transformer2DModel
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_feature_extractor
from src.pipeline_stable_diffusion_3_tryoff_masked import StableDiffusion3TryOffPipelineMasked
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from transformers import CLIPVisionModelWithProjection
from PIL import Image
from torchvision import transforms


def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one, text_encoder_two, text_encoder_three = None, None, None
    if class_one is not None and class_two is not None:
        text_encoder_one = class_one.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            variant=args.variant,
        )
        text_encoder_two = class_two.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            variant=args.variant,
        )
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)

    if class_three is not None:
        text_encoder_three = class_three.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_3",
            revision=args.revision,
            variant=args.variant,
        )
        text_encoder_three.requires_grad_(False)
    return text_encoder_one, text_encoder_two, text_encoder_three


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def main(args):
    """
    Generate virtual try-off image using SD3 pipeline.
    It can be used to generate a try-off image from a single image.

    Args:
        args: Parsed command line arguments containing model paths, and generation parameters.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load model, processor

    model_load_time = time.time()

    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        low_cpu_mem_usage=True,
    )

    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one,
        text_encoder_cls_two,
        text_encoder_cls_three,
    )

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
        args.pretrained_model_name_or_path_sd3_tryoff,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    transformer_vton_feature_extractor = SD3Transformer2DModel_feature_extractor.from_pretrained(
        args.pretrained_model_name_or_path_sd3_tryoff,
        subfolder="transformer_vton",
        revision=args.revision,
        variant=args.variant,
    )

    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(
        device=device, dtype=weight_dtype
    )
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(
        device=device, dtype=weight_dtype
    )

    pipeline = StableDiffusion3TryOffPipelineMasked(
        scheduler=noise_scheduler,
        vae=vae,
        transformer_vton_feature_extractor=transformer_vton_feature_extractor,
        transformer_garm=transformer,
        image_encoder_large=image_encoder_large,
        image_encoder_bigG=image_encoder_bigG,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        tokenizer_3=tokenizer_three,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
    )

    pipeline.to(device, dtype=weight_dtype)
    print(f"Model load time: {time.time() - model_load_time:.4f}")

    # load data
    data_load_time = time.time()

    image_path = args.example_image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image_directory = os.path.dirname(image_path)

    image = Image.open(image_path)
    image = image.resize((args.width, args.height))
    image = transforms.ToTensor()(image).unsqueeze(0)

    caption = open(os.path.join(image_directory, f"{image_name}_caption.txt"), "r").read()

    image_fine_mask = Image.open(os.path.join(image_directory, f"{image_name}_fine_mask.jpg"))
    image_fine_mask = image_fine_mask.resize((args.width, args.height))

    image_binary_mask = Image.open(os.path.join(image_directory, f"{image_name}_binary_mask.jpg"))
    image_binary_mask = image_binary_mask.resize((args.width, args.height))

    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed else None
    print(f"Data load time: {time.time() - data_load_time:.4f}")

    # Inferecen Time
    total_inference_time = time.time()
    image = pipeline(
        prompt=caption,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        vton_image=image,
        mask_input=image_binary_mask,
        image_input_masked=image_fine_mask,
    ).images[0]
    print(f"Total Inference Time: {time.time() - total_inference_time:.4f}")

    image.save(f"{args.output_dir}/{image_name}_output.png")


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
        "--seed",
        type=int,
        default=42,
        required=False,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        required=True,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        required=True,
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        required=False,
    )
    parser.add_argument(
        "--example_image",
        type=str,
        default="examples/example1.jpg",
        required=True,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=2.0,
        required=True,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        required=True,
    )
    args = parser.parse_args()
    main(args)
