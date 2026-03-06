import argparse
import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import torch
import transformers
from accelerate import Accelerator, DistributedType, init_empty_weights
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from torchvision import transforms
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.utils import (
    check_min_version,
    is_wandb_available,
)
from diffusers.utils.torch_utils import is_compiled_module

from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset

from src.transformer_repa import SD3Transformer2DModel as SD3Transformer2DModel_repa
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_feature_extractor
import random
import torch.nn.functional as F

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)

@torch.no_grad()
def load_encoder(enc_type, device, resolution=512):
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')

        if 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()

        return encoder

def pad_width_to_1024_tensor(image_tensor: torch.Tensor) -> torch.Tensor:

    B, C, H, W = image_tensor.shape
    assert H == 1024, "Input height must be 1024"
    assert W <= 1024, "Input width must be <= 1024"

    pad_total = 1024 - W
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    padded = F.pad(image_tensor, (pad_left, pad_right, 0, 0), value=1.0)
    return padded

def preprocess_for_dino_with_padding(image_tensor: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Full preprocessing: assumes input in [-1, 1], [B, 3, 1024, 768].
    Pads to square (1024x1024), rescales to `size`, normalizes with ImageNet stats.
    """
    image_tensor = pad_width_to_1024_tensor(image_tensor)
    image_tensor = (image_tensor + 1.0) / 2.0 # normalize to [0, 1]
    image_transforms = transforms.Compose([
                       transforms.Resize(target_size),
                       transforms.CenterCrop(target_size),
                       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image_tensor = image_transforms(image_tensor)
    return image_tensor

def preprocess_for_dino(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Full preprocessing: assumes input in [-1, 1], [B, 3, 1024, 768].
    Pads to square (1024x1024), rescales to `size`, normalizes with ImageNet stats.
    """
    image_tensor = (image_tensor + 1.0) / 2.0
    image_transforms = transforms.Compose([
                       transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    image_tensor = image_transforms(image_tensor)
    return image_tensor

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def projection_loss(z_gt, z_pred):
    """
    Compute the projection loss (Cosine Distance) between z_pred and z_gt.
    Loss = 1 - CosineSimilarity
    """
    proj_loss = 0.0
    bsz = z_gt[0].shape[0]
    num_feature_sets = len(z_gt)

    for i, (z, z_tilde) in enumerate(zip(z_gt, z_pred)):
        z_tilde = torch.nn.functional.normalize(z_tilde, dim=-1)
        z = torch.nn.functional.normalize(z, dim=-1)

        cosine_similarity = (z * z_tilde).sum(dim=-1)
        loss_per_item = mean_flat(- cosine_similarity)

        proj_loss += loss_per_item.sum() # Sum over batch

    proj_loss /= (num_feature_sets * bsz)
    return proj_loss

def memory_usage(device=None, accelerator=None, verbose=True):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(device, str):
        device = torch.device(device)

    if device.type == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(device)
        used = total - free
        mem_used_MB = used / 1024**2
        mem_used_GB = used / 1024**3
        result = {"MB": mem_used_MB, "GB": mem_used_GB}

        if verbose:
            accelerator.print(f"Memory usage on {device}: {mem_used_MB:.2f} MB ({mem_used_GB:.2f} GB)")
        return result
    elif device.type == "cpu":
        if verbose:
            accelerator.print("Memory usage check is only available for CUDA devices.")
        return {"MB": 0.0, "GB": 0.0}
    else:
        raise ValueError(f"Unsupported device type: {device}")

def get_gpu_memory_info(accelerator):
    if torch.backends.cuda.is_built():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated()
        free_memory = total_memory - allocated_memory
        total_memory_mb = total_memory / 1024**3
        allocated_memory_mb = allocated_memory / 1024**3
        free_memory_mb = free_memory / 1024**3
        
        accelerator.print(f"Total GPU Memory: {total_memory_mb:.2f} GB")
        accelerator.print(f"Used Memory: {allocated_memory_mb:.2f} GB")
        accelerator.print(f"Free Memory: {free_memory_mb:.2f} GB")
        accelerator.print(f"Memory utilization: {(allocated_memory/total_memory)*100:.2f}%")
    else:
        accelerator.print("CUDA/ROCm not available")


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
        "--pretrained_model_name_or_path_sd3",
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
        "--caption_type",
        type=str,
        required=False,
        choices=["original", "structural", "qwen"],
        help="caption type based on the model adopted for captioning. Structural is original without color and texture informations"
    )
    parser.add_argument(
        "--coarse_caption_file",
        type=str,
        required=False,
        default='coarse_captions.json'
    )
    parser.add_argument(
        "--mask_type",
        type=str,
        required=False,
        default="bounding_box",
        choices=["mask", "bounding_box"],
        help="Mask type of the dataset.",
    )
    parser.add_argument(
        "--texture_order",
        type=str,
        required=False,
        choices=["shuffled", "original"],
        help="Texture order of the dataset.",
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
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd3-vtoff",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Height of the input images.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=384,
        help="Width of the input images.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for the dataloader.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--cfg_dropout_probability",
        type=float,
        default=0.1,
        help="Probability of dropping the conditioning for classifier-free guidance training.",
    )
    parser.add_argument(
        "--cfg_text_dropout_probability",
        type=float,
        default=0.1,
        help="Probability of dropping the text conditioning for classifier-free guidance training.",
    )
    parser.add_argument(
        "--repa_loss_weight",
        type=float,
        default=1.0,
        help="Weight of the repa loss.",
    )
    parser.add_argument(
        "--probing_method",
        type=str,
        choices=["average_pool", "conv"],
        default="average_pool",
    )
    parser.add_argument(
        "--encoder_depth",
        type=int,
        default=8,
        help="Depth of the encoder.",
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def main(args):
    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    
    dinov2_encoder = load_encoder("dinov2_reg-vit-b", accelerator.device, resolution=512)
    dinov2_encoder.eval()
    dinov2_encoder.requires_grad_(False)
    dinov2_encoder.to(dtype=torch.bfloat16)
    z_dims = [dinov2_encoder.embed_dim]
        
    transformer = SD3Transformer2DModel_repa.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, 
        variant=args.variant, low_cpu_mem_usage=False, z_dims=z_dims, encoder_depth=args.encoder_depth, probing_method=args.probing_method,
    )
    transformer = transformer.to(dtype=torch.bfloat16)
    transformer.initialize_weights()
    accelerator.print(f"Transformer depth: {args.encoder_depth}")

    transformer_vton_feature_extractor = SD3Transformer2DModel_feature_extractor.from_pretrained(
        args.pretrained_model_name_or_path_sd3, subfolder="transformer_vton", revision=args.revision, variant=args.variant
    ).to(dtype=torch.bfloat16)

    transformer.requires_grad_(True)
    transformer_vton_feature_extractor.requires_grad_(False)
    vae.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer_vton_feature_extractor.to(accelerator.device, dtype=weight_dtype)

    accelerator.print("Memory usage after loading models")
    memory_usage(accelerator.device, accelerator)
    #get_gpu_memory_info(accelerator)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                # Unwrap the model to get the original Transformer model, handling DeepSpeedEngine
                unwrapped_model = unwrap_model(model) # Use the utility function

                # Check if the unwrapped model is one of the expected transformer types
                if isinstance(unwrapped_model, (SD3Transformer2DModel_repa)):
                    # Save the unwrapped model
                    unwrapped_model.save_pretrained(os.path.join(output_dir, "transformer"))
                else:
                     # If it's not one of the expected types after unwrapping, raise an error
                    raise ValueError(f"Unexpected unwrapped model type: {unwrapped_model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        transformer_ = None
        init_under_meta = False

        if not accelerator.distributed_type == DistributedType.DEEPSPEED:
            while len(models) > 0:
                model = models.pop()

                if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                    transformer_ = unwrap_model(accelerator, model)
                else:
                    raise ValueError(f"Unexpected save model: {unwrap_model(accelerator, model).__class__}")
        else:
            with init_empty_weights():
                accelerator.print("Loading repa transformer")
                load_model = SD3Transformer2DModel_repa.from_pretrained(
                    os.path.join(input_dir, "transformer")
                )
                transformer_ = SD3Transformer2DModel_repa.from_config(
                    load_model.config
                )
                init_under_meta = True

        transformer_.register_to_config(**load_model.config)
        transformer_.load_state_dict(load_model.state_dict(), assign=init_under_meta)
        del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer.parameters(), "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    if args.dataset_name == "dresscode":
        outputlist = ['image', 'cloth', 'vton_image_embeddings', 'clip_pooled', 'clip_embeds', 't5_embeds', 'parse_cloth', 'im_mask_fine']
        if args.category == "all":
            args.category = ["upper_body", "lower_body", "dresses"]
        else:
            args.category = [args.category]
        train_dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=args.category,
            size=(args.height, args.width),
            coarse_caption_file=args.coarse_caption_file
        )
        test_dataset = DressCodeDataset(
            dataroot_path=args.dataset_root,
            phase="test",
            order=args.order,
            radius=5,
            outputlist=outputlist,
            sketch_threshold_range=(20, 20),
            category=args.category,
            size=(args.height, args.width),
            coarse_caption_file=args.coarse_caption_file
        )
    elif args.dataset_name == "vitonhd":
        outputlist = ['image', 'cloth', 'vton_image_embeddings', 'clip_pooled', 'clip_embeds', 't5_embeds', 'parse_cloth', 'im_mask_fine']
        train_dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase=args.phase,
            order=args.order,
            radius=5,
            outputlist=outputlist,
            size=(args.height, args.width),
            mask_type=args.mask_type,
            texture_order=args.texture_order,
            caption_file=args.coarse_caption_file
        )
        test_dataset = VitonHDDataset(
            dataroot_path=args.dataset_root,
            phase="test",
            order=args.order,
            radius=5,
            outputlist=outputlist,
            size=(args.height, args.width),
            mask_type=args.mask_type,
            texture_order=args.texture_order,
            caption_file=args.coarse_caption_file
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True, # cannot pin 'CUDABFloat16Type' only dense CPU tensors can be pinned
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )
    
    accelerator.print("Memory usage after accelerator prepare")
    memory_usage(accelerator.device, accelerator)
    #get_gpu_memory_info(accelerator)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "sd3-vtoff"
        accelerator.init_trackers(tracker_name, config=vars(args))#, init_kwargs={"wandb": {"relogin": True}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0
    
    # DEBUG
    accelerator.print("max gpu usage after accelerator loading")
    accelerator.print(torch.cuda.max_memory_allocated() / 1024**3)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # Expected in format 1 x hs or 1 x ntokens x hs
    empty_string_pooled_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat_pooled" / "empty_string.pt", map_location="cpu").to(accelerator.device, dtype=weight_dtype)
    empty_string_clip_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "CLIP_VIT_L_VIT_G_concat" / "empty_string.pt", map_location="cpu").to(accelerator.device, dtype=weight_dtype)
    empty_string_t5_prompt_embeds = torch.load(Path(args.dataset_root).parent / "empty_string_embeddings" / "T5_XXL" / "empty_string.pt", map_location="cpu").to(accelerator.device, dtype=weight_dtype)

    print("Starting training")
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]
            with accelerator.accumulate(models_to_accumulate):
                vton_img = batch.get("image").to(device=accelerator.device)
                vton_img = vton_img.to(dtype=weight_dtype)
                vton_clip_embeddings = batch.get("vton_image_embeddings").to(device=accelerator.device).to(dtype=weight_dtype)
                clip_pooled_embeddings = batch.get("clip_pooled").to(device=accelerator.device).to(dtype=weight_dtype)
                clip_text_embeddings = batch.get("clip_embeds").to(device=accelerator.device).to(dtype=weight_dtype)
                t5_text_embeddings = batch.get("t5_embeds").to(device=accelerator.device).to(dtype=weight_dtype)
                text_embeddings=torch.cat([clip_text_embeddings, t5_text_embeddings], dim=1)
                garment_img = batch.get("cloth").to(device=accelerator.device)
                garment_img = garment_img.to(dtype=weight_dtype)
                # Dropout the conditioning for classifier-free guidance training.
                if random.random() < args.cfg_dropout_probability:
                    vton_clip_embeddings = torch.zeros_like(vton_clip_embeddings)
                if random.random() < args.cfg_text_dropout_probability:
                    text_embeddings = torch.cat([empty_string_clip_prompt_embeds, empty_string_t5_prompt_embeds], dim=1)
                if random.random() < args.cfg_text_dropout_probability:
                    clip_pooled_embeddings = empty_string_pooled_prompt_embeds

                with torch.no_grad():
                    # Convert images to latent space
                    latent_dist = vae.encode(garment_img).latent_dist
                    model_input = latent_dist.sample()
                    model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                    vton_latent_dist = vae.encode(vton_img).latent_dist
                    vton_model_input = vton_latent_dist.sample()
                    vton_model_input = (vton_model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    vton_model_input = vton_model_input.to(dtype=weight_dtype)

                masked_vton_img = batch.get("im_mask_fine").to(device=accelerator.device)
                inpaint_mask = batch.get("parse_cloth").to(device=accelerator.device)
                if len(inpaint_mask.shape) == 3:
                    inpaint_mask = inpaint_mask.unsqueeze(1)
                masked_vton_img = masked_vton_img.to(dtype=weight_dtype)
                inpaint_mask = inpaint_mask.to(dtype=weight_dtype)
                inpaint_mask = torch.nn.functional.interpolate(inpaint_mask, size=(inpaint_mask.shape[2] // 8, inpaint_mask.shape[3] // 8))
                
                with torch.no_grad():
                    masked_latent_dist = vae.encode(masked_vton_img).latent_dist
                    masked_model_input = masked_latent_dist.sample()
                    masked_model_input = (masked_model_input - vae.config.shift_factor) * vae.config.scaling_factor
                    masked_model_input = masked_model_input.to(dtype=weight_dtype)

                if random.random() < args.cfg_dropout_probability:
                    vton_model_input = torch.zeros_like(vton_model_input)
                    inpaint_mask = torch.zeros_like(inpaint_mask)
                    masked_vton_img = torch.zeros_like(masked_vton_img)
                    
                vton_model_input = torch.cat([vton_model_input, inpaint_mask, masked_model_input], dim=1)
                    
                torch.cuda.empty_cache()

                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                
                with torch.no_grad():
                    output = transformer_vton_feature_extractor(
                        hidden_states=vton_model_input, # vton_model_input is the latent image WITHOUT noise
                        timestep=timesteps * 0,
                        encoder_hidden_states=None,
                        pooled_projections=vton_clip_embeddings,
                        return_dict=False,
                    )

                    ref_key = output[1]
                    ref_value = output[2]
                
                output = transformer(
                        hidden_states=noisy_model_input, # noisy model input is the latent image WITH noise
                        timestep=timesteps,
                        encoder_hidden_states=text_embeddings,
                        pooled_projections=clip_pooled_embeddings,
                        return_dict=False,
                        ref_key=ref_key,
                        ref_value=ref_value,
                    )
                model_pred = output[0]

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                zs = []
                zs_tilde = output[3]

                if not isinstance(zs_tilde, list):
                    zs_tilde = [zs_tilde]
                garment_img_for_dino = preprocess_for_dino_with_padding(garment_img, target_size=448)
                garment_img_for_dino = garment_img_for_dino.to(device=accelerator.device, dtype=weight_dtype)
                with torch.no_grad():
                    z = dinov2_encoder.forward_features(garment_img_for_dino)
                    z = z['x_norm_patchtokens']
                zs.append(z.detach())
                proj_loss_val = projection_loss(zs, zs_tilde)
                loss = loss + args.repa_loss_weight * proj_loss_val

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (transformer.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % args.checkpointing_steps == 0:
                        # before saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        
                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]
                        
                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                        
                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            logs["repa_loss"] = args.repa_loss_weight * proj_loss_val.detach().item()
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        os.makedirs(os.path.join(args.output_dir, "final", "transformer"), exist_ok=True)
        transformer.save_pretrained(
                            os.path.join(args.output_dir, "final", "transformer"),
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(transformer),
        )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)