<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center" style="margin: 0 0 px 0;">TEMU-VTOFF</h1>
  <h3 align="center" style="margin: 0;">Text-Enhanced MUlti-category Virtual Try-Off</h3>
  <h3 align="center" style="margin: 6px 0 15px 0;">ICLR 2026</h3>
</div>

<div align="center">
  <picture>
    <source srcset="assets/teaser.png" media="(prefers-color-scheme: dark)">
      <img src="assets/teaser.png" width="75%" alt="MiniMax">
    </source>
  </picture>
</div>

<div align="center">

> **Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals**  
> [Davide Lobba](https://scholar.google.com/citations?user=WEMoLPEAAAAJ&hl=en&oi=ao)<sup>1,2,\*</sup>, [Fulvio Sanguigni](https://scholar.google.com/citations?user=tSpzMUEAAAAJ&hl=en)<sup>2,3,\*</sup>, [Bin Ren](https://scholar.google.com/citations?user=Md9maLYAAAAJ&hl=en)<sup>1,2</sup>, [Marcella Cornia](https://scholar.google.com/citations?user=DzgmSJEAAAAJ&hl=en)<sup>3</sup>, [Rita Cucchiara](https://scholar.google.com/citations?user=OM3sZEoAAAAJ&hl=en)<sup>3</sup>, [Nicu Sebe](https://scholar.google.com/citations?user=stFCYOAAAAAJ&hl=en)<sup>1</sup>   
> <sup>1</sup>University of Trento, <sup>2</sup>University of Pisa, <sup>3</sup>University of Modena and Reggio Emilia  
> <sup>*</sup> Equal contribution
</div>

<div align="center">
  <a href="https://arxiv.org/abs/2505.21062" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/Paper-arXiv-darkred.svg" alt="Paper">
  </a>
  <a href="https://arxiv.org/pdf/2505.21062" style="margin: 0 2px;">
    <img src="https://img.shields.io/badge/PDF-arXiv-darkred.svg" alt="PDF">
  </a>
  <a href='https://temu-vtoff-page.github.io/' style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/Webpage-Project-silver?style=flat&logo=&logoColor=orange' alt='webpage'>
  </a>
  <a href="https://huggingface.co/davidelobba/TEMU-VTOFF">
    <img src="https://img.shields.io/badge/HuggingFace-Model-FFB000.svg" alt="Project">
  </a>
  <a href="https://raw.githubusercontent.com/davidelobba/TEMU-VTOFF/refs/heads/main/LICENSE" style="margin: 0 2px;">
    <img src='https://img.shields.io/badge/License-CC BY--NC 4.0-lightgreen?style=flat&logo=Lisence' alt='License'>
  </a>
</div>



<!-- TABLE OF CONTENTS -->
<details>
<summary>Table of Contents</summary>
<ol>
  <li><a href="#-about-the-project">About The Project</a></li>
  <li><a href="#-key-features">Key Features</a></li>
  <li><a href="#-getting-started">Getting Started</a>
    <ul>
      <li><a href="#prerequisites">Prerequisites</a></li>
      <li><a href="#installation">Installation</a></li>
    </ul>
  </li>
  <li><a href="#inference">Inference</a></li>
  <li><a href="#dataset-inference">Dataset Inference</a></li>
  <ul>
      <li><a href="#dataset-captioning">Dataset Captioning</a></li>
      <li><a href="#feature-extraction">Feature Extraction</a></li>
      <li><a href="#generate-images">Generate Images</a></li>
    </ul>
  <li><a href="#-contact">Contact</a></li>
  <li><a href="#citation">Citation</a></li>
</ol>
</details>



<!-- ABOUT THE PROJECT -->
## 💡 About The Project

TEMU-VTOFF is a novel dual-DiT (Diffusion Transformer) architecture designed for the Virtual Try-Off task: generating clean, in-shop images of garments worn by a person. By combining a pretrained feature extractor with a text-enhanced generation module, our method can handle occlusions, multiple garment categories, and ambiguous appearances. It further refines generation fidelity via a feature alignment module based on DINOv2.

## ✨ Key Features
Our contribution can be summarized as follows:
- **🎯 Multi-Category Try-Off**. We present a unified framework capable of handling multiple garment types (upper-body, lower-body, and full-body clothes) without requiring category-specific pipelines.
- **🔗 Multimodal Hybrid Attention**. We introduce a novel attention mechanism that integrates garment textual descriptions into the generative process by linking them with person-specific features. This helps the model synthesize occluded or ambiguous garment regions more accurately.
- **⚡ Garment Aligner Module**. We design a lightweight aligner that conditions generation on clean garment images, replacing conventional denoising objectives. This leads to better alignment consistency on the overall dataset and preserves more precise visual retention.
- **📊 Extensive experiments**. Experiments on the Dress Code and VITON-HD datasets demonstrate that TEMU-VTOFF outperforms prior methods in both the quality of generated images and alignment with the target garment, highlighting its strong generalization capabilities.

<!-- GETTING STARTED -->
## 💻 Getting Started
### Prerequisites

Clone the repository:
  ```sh
  git clone https://github.com/davidelobba/TEMU-VTOFF.git
  ```

### Installation

1. We recommend installing the required packages using Python's native virtual environment (venv) as follows:
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```
2. Upgrade pip and install dependencies
   ```sh
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Create a .env file like the following:
   ```js
   export WANDB_API_KEY="ENTER YOUR WANDB TOKEN"
   export HF_TOKEN="ENTER YOUR HUGGINGFACE TOKEN"
   export HF_HOME="PATH WHERE YOU WANT TO SAVE THE HF MODELS"
   ```
🧠 Note: Access to Stable Diffusion 3 Medium must be requested via [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers).

## Inference
Let's generate the in-shop garment image.
```sh
source venv/bin/activate
source .env

python inference.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
    --seed 42 \
    --width "768" \
    --height "1024" \
    --output_dir "put here the output path" \
    --mixed_precision "bf16" \
    --example_image "examples/example1.jpg" \
    --guidance_scale 2.0 \
    --num_inference_steps 28
```

## Dataset Inference
### Dataset Captioning
Generate textual descriptions for each sample using a multimodal VLM (e.g., Qwen2.5-VL).

```sh
python precompute_utils/captioning_qwen.py \
            --pretrained_model_name_or_path "Qwen/Qwen2.5-VL-7B-Instruct" \
            --dataset_name "dresscode" \
            --dataset_root "path to your dataset" \
            --filename "qwen_captions_2_5.json"\
            --temperatures 0.2
```

### Feature extraction
Extract textual features using OpenCLIP, CLIP and T5 text encoders.

```sh
phases=("test" "train")
for phase in "${phases[@]}"; do
    python precompute_text_features.py \
               --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
               --dataset_name "dresscode" \
               --dataset_root "path to your dataset" \
               --phase $phase \
               --order "paired" \
               --category "all" \
               --height 1024 \
               --width 768 \
               --batch_size 4 \
               --mixed_precision "fp16" \
               --num_workers 8 \
               --text_encoders "T5" "CLIP" \
               --captions_type "qwen_text_embeddings"
done
```

Extract visual features using OpenCLIP and CLIP vision encoders.
```sh
phases=("test" "train")
for phase in "${phases[@]}"; do
   python precompute_utils/precompute_image_features.py \
               --dataset "dresscode" \
               --dataroot "path to your dataset" \
               --phase $phase \
               --order "paired" \
               --category "all" \
               --height 1024 \
               --width 768 \
               --batch_size 4 \
               --mixed_precision "fp16" \
               --num_workers 8
done
```
### Generate Images
Let's generate the in-shop garment images of DressCode or VITON-HD using the TEMU-VTOFF model.
```sh
source venv/bin/activate
source .env

python inference_dataset.py \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
    --pretrained_model_name_or_path_sd3_tryoff "davidelobba/TEMU-VTOFF" \
    --dataset_name "dresscode" \
    --dataset_root "path to your dataset" \
    --output_dir "path to your output folder" \
    --coarse_caption_file "qwen_captions_2_5_0_2.json" \
    --phase "test" \
    --order "paired" \
    --height "1024" \
    --width "768" \
    --mask_type "bounding_box" \
    --category "all" \
    --batch_size 4 \
    --mixed_precision "bf16" \
    --seed 42 \
    --num_workers 8 \
    --fine_mask \
    --guidance_scale 2.0 \
    --num_inference_steps 28
```

## Training
### Feature Extractor training
Let's train the feature extractor.
```sh
source venv/bin/activate
source .env

accelerate launch --config_file "path to your accelerate config" \
  --num_processes=$NUM_PROCESSES \
  --num_machines=$SLURM_NNODES \
  --machine_rank=$SLURM_NODEID \
  --main_process_ip=$MAIN_PROCESS_IP \
  training_feature_extractor.py \
      --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
      --dataset_name "dresscode" \
      --dataset_root "path to your dataset" \
      --phase "train" \
      --order "paired" \
      --category "all" \
      --output_dir "path to your output folder" \
      --seed 42 \
      --height 1024 \
      --width 768 \
      --train_batch_size 8 \
      --max_train_steps 30000  \
      --checkpointing_steps 5000 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_scheduler "cosine" \
      --lr_warmup_steps 1500 \
      --weighting_scheme "logit_normal" \
      --optimizer "adamw" \
      --report_to "wandb" \
      --mixed_precision "bf16" \
      --num_workers 8 \
      --mask_type "bounding_box" \
```

### TEMU-VTOFF training
Let's train the final model.
```sh
source venv/bin/activate
source .env

accelerate launch --config_file "path to your accelerate config" \
  --num_processes=$NUM_PROCESSES \
  --num_machines=$SLURM_NNODES \
  --machine_rank=$SLURM_NODEID \
  --main_process_ip=$MAIN_PROCESS_IP \
  training_TEMU_VTOFF.py \
      --pretrained_model_name_or_path "stabilityai/stable-diffusion-3-medium-diffusers" \
      --pretrained_model_name_or_path_sd3 "path to pretrained feature extractor" \
      --dataset_name "dresscode" \
      --dataset_root "path to your dataset" \
      --phase "train" \
      --order "paired" \
      --category "all" \
      --coarse_caption_file "qwen_captions_2_5_0_2.json" \
      --output_dir "path to your output folder" \
      --seed 42 \
      --height 1024 \
      --width 768 \
      --train_batch_size 8 \
      --max_train_steps 30000 \
      --checkpointing_steps 6000 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_scheduler "cosine" \
      --lr_warmup_steps 3000 \
      --weighting_scheme "logit_normal" \
      --optimizer "adamw" \
      --report_to "wandb" \
      --mixed_precision "bf16" \
      --cfg_dropout_probability 0.1 \
      --cfg_text_dropout_probability 0.1 \
      --num_workers 8 \
      --repa_loss_weight 0.5 \
      --probing_method "average_pool" \
      --resume_from_checkpoint "latest"
```


## 📬 Contact
**Lead Authors:**
- 📧 **Davide Lobba**: [davide.lobba@unitn.it](mailto:davide.lobba@unitn.it) | 🎓 [Google Scholar](https://scholar.google.com/citations?user=WEMoLPEAAAAJ&hl=en&oi=ao)
- 📧 **Fulvio Sanguigni**: [fulvio.sanguigni@unimore.it](mailto:fulvio.sanguigni@unimore.it) | 🎓 [Google Scholar](https://scholar.google.com/citations?user=tSpzMUEAAAAJ&hl=en)

For questions about the project, feel free to reach out to any of the lead authors!


## Citation
Please cite our paper if you find our work helpful:
```bibtex
@article{lobba2025inverse,
  title={Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals},
  author={Lobba, Davide and Sanguigni, Fulvio and Ren, Bin and Cornia, Marcella and Cucchiara, Rita and Sebe, Nicu},
  journal={arXiv preprint arXiv:2505.21062},
  year={2025}
}
```
