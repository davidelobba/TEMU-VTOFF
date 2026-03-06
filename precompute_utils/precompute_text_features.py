import argparse
import gc
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, PretrainedConfig, T5EncoderModel, T5TokenizerFast
from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from accelerate import Accelerator
from tqdm import tqdm


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        required=True,
        choices=["dresscode", "vitonhd"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        required=True,
        help="Path to the dataset root.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        required=True,
        choices=["train", "test"],
        help="Phase of the dataset to use.",
    )
    parser.add_argument(
        "--order",
        type=str,
        required=True,
        choices=["unpaired", "paired"],
        help="Test order of the dataset to use.",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=False,
        choices=["all", "dresses", "upper_body", "lower_body"],
        help="Category of the dataset."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--captions_type",
        type=str,
        required=False,
        choices=["text_embeddings", "qwen_text_embeddings", "struct_text_embeddings"],
        default="text_embeddings",
        help="caption type, for example original (text embeddings), qwen_text_embeddings, struct_text_embeddings",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Height of the input images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="Width of the input images.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["fp16", "bf16", "no"],
        help="Mixed precision.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument("--text_encoders", type=str, nargs='+', default="CLIP")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def load_text_encoders(class_one, class_two, class_three, pretrained_model_name_or_path, revision, variant):
    text_encoder_one, text_encoder_two, text_encoder_three = None, None, None
    if class_one is not None and class_two is not None:
        text_encoder_one = class_one.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder", revision=revision, variant=variant,
        )
        text_encoder_two = class_two.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_2", revision=revision, variant=variant,
        )
        text_encoder_one.requires_grad_(False)
        text_encoder_two.requires_grad_(False)
        
    if class_three is not None:
        text_encoder_three = class_three.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder_3", revision=revision, variant=variant,
        )
        text_encoder_three.requires_grad_(False)
    return text_encoder_one, text_encoder_two, text_encoder_three

def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        if text_encoder is None:
            continue
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    if len(clip_prompt_embeds_list) > 0:
        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
    else: pooled_prompt_embeds = None

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    if len(clip_prompt_embeds_list) > 0:
        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )       
    else: clip_prompt_embeds = None

    return t5_prompt_embed, clip_prompt_embeds, pooled_prompt_embeds

def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt_no_t5(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for i, (tokenizer, text_encoder) in enumerate(zip(clip_tokenizers, clip_text_encoders)):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=None,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_zeros = torch.zeros(
        (
            clip_prompt_embeds.shape[0],  # batch size
            max_sequence_length,          # T5 sequence length
            4096                          #transformer.config.joint_attention_dim
        ),
        device=device if device is not None else clip_prompt_embeds.device,
        dtype=clip_prompt_embeds.dtype
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_zeros.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_zeros], dim=-2)

    return prompt_embeds, pooled_prompt_embeds

def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds

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


@torch.inference_mode()
def main(args):
    width, height = args.width, args.height

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
            
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        with torch.no_grad():
            if text_encoders[-1] is not None:
                t5_prompt_embeds, prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
                )
            else:
                prompt_embeds, pooled_prompt_embeds = encode_prompt_no_t5(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                t5_prompt_embeds=None
                
            if prompt_embeds is not None:
                prompt_embeds = prompt_embeds.to(accelerator.device)
            if pooled_prompt_embeds is not None:
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
            if t5_prompt_embeds is not None: t5_prompt_embeds = t5_prompt_embeds.to(accelerator.device)
            
        return t5_prompt_embeds, prompt_embeds, pooled_prompt_embeds
        
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

    text_encoder_cls_one, text_encoder_cls_two = None, None
    if "CLIP" in args.text_encoders:
        text_encoder_cls_one = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision
        )
        text_encoder_cls_two = import_model_class_from_model_name_or_path(
            args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
        )
    text_encoder_cls_three = None
    if "T5" in args.text_encoders:
        text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
        )   
    
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three,
        args.pretrained_model_name_or_path, args.revision, args.variant,
    )
    
    if "CLIP" in args.text_encoders:
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if "T5" in args.text_encoders:
        text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    outputlist = ['captions', 'c_name', 'category']
    if args.category == "all":
        args.category = ["upper_body", "lower_body", "dresses"]
    elif type(args.category) is not list:
        args.category = [args.category]
            
    if args.captions_type == "text_embeddings":
        coarse_caption_file = "coarse_captions.json"
    elif args.captions_type == "struct_text_embeddings":
        coarse_caption_file = "all_struct_captions.json"
    elif args.captions_type == "qwen_text_embeddings":
        coarse_caption_file = "qwen_captions_2_5_0_2.json"
            
            
    if args.dataset_name == "dresscode":
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=args.category,
            size=(height, width),
            coarse_caption_file=coarse_caption_file
            )
    elif args.dataset_name == "vitonhd":
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            size=(height, width),
            caption_file=coarse_caption_file
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented")
        
    dataroot = test_dataset.dataroot
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
        

    test_dataloader = accelerator.prepare(test_dataloader)
        
    # empty text
    prompts = [""]
    t5_prompt_embeds, clip_prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    pooled_feat_path = Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat_pooled" / "empty_string.pt"
    clip_feat_path = Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat" / "empty_string.pt"
    t5_feat_path = Path(args.dataset_root).parent / "empty_string_embeddings" / "T5_XXL" / "empty_string.pt"
        
    if not pooled_feat_path.parent.exists():
        pooled_feat_path.parent.mkdir(parents=True, exist_ok=True)
    if not clip_feat_path.parent.exists():
        clip_feat_path.parent.mkdir(parents=True, exist_ok=True)
    if not t5_feat_path.parent.exists():
        t5_feat_path.parent.mkdir(parents=True, exist_ok=True)
                            
    if pooled_prompt_embeds is not None:
        pooled_embeds = pooled_prompt_embeds
        torch.save(pooled_embeds, pooled_feat_path)
    if clip_prompt_embeds is not None:
        clip_embeds = clip_prompt_embeds
        torch.save(clip_embeds, clip_feat_path)
    if t5_prompt_embeds is not None:
        t5_embeds = t5_prompt_embeds
        torch.save(t5_embeds, t5_feat_path)

    # text embeddings
    for idx, batch in enumerate(tqdm(test_dataloader)):
        prompts = batch["captions"]
        t5_prompt_embeds, clip_prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, text_encoders, tokenizers
        )
        
        for i in range(len(prompts)):
            name = batch["c_name"][i]
            
            if args.dataset_name == "dresscode":
                pooled_feat_path = Path(dataroot) / args.captions_type / "CLIP_VIT_L_VIT_G_concat_pooled" / batch["category"][i] / f"{name.split('.')[0]}.pt"
                clip_feat_path = Path(dataroot) / args.captions_type / "CLIP_VIT_L_VIT_G_concat" / batch["category"][i] / f"{name.split('.')[0]}.pt"
                t5_feat_path = Path(dataroot) / args.captions_type / "T5_XXL" / batch["category"][i] / f"{name.split('.')[0]}.pt"
            elif args.dataset_name == "vitonhd":
                pooled_feat_path = Path(dataroot) / args.captions_type / "CLIP_VIT_L_VIT_G_concat_pooled" / args.phase / f"{name.split('.')[0]}.pt"
                clip_feat_path = Path(dataroot) / args.captions_type / "CLIP_VIT_L_VIT_G_concat" / args.phase / f"{name.split('.')[0]}.pt"
                t5_feat_path = Path(dataroot) / args.captions_type / "T5_XXL" / args.phase / f"{name.split('.')[0]}.pt"                    
            else: raise NotImplementedError("dataset not supported")
            
            if not pooled_feat_path.parent.exists():
                pooled_feat_path.parent.mkdir(parents=True, exist_ok=True)
            if not clip_feat_path.parent.exists():
                clip_feat_path.parent.mkdir(parents=True, exist_ok=True)
            if not t5_feat_path.parent.exists():
                t5_feat_path.parent.mkdir(parents=True, exist_ok=True)
                                
            if pooled_prompt_embeds is not None:
                pooled_embeds = pooled_prompt_embeds[i]
                torch.save(pooled_embeds, pooled_feat_path)
            if clip_prompt_embeds is not None:
                clip_embeds = clip_prompt_embeds[i]
                torch.save(clip_embeds, clip_feat_path)
            if t5_prompt_embeds is not None:
                t5_embeds = t5_prompt_embeds[i]
                torch.save(t5_embeds, t5_feat_path)
    

if __name__ == '__main__':
    args = parse_args()
    main(args)