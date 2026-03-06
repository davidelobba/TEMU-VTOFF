from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os
from accelerate.utils import set_seed
import torch
import json
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset.dresscode import DressCodeDataset
from dataset.vitonhd import VitonHDDataset
from tqdm import tqdm

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=False,
        choices=["dresscode", "vitonhd"],
        help="",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        required=False,
        help="",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        required=False,
        help="Path to visual attributes annotations for RAG enhanced VLM captioning",
    )
    parser.add_argument("--height",type=int,default=1024)
    parser.add_argument("--width",type=int,default=768)
    parser.add_argument("--temperatures",type=float, nargs="+", default=[0.2])

    
    args = parser.parse_args()
    return args

def qwen_captioning(args):
    set_seed(42)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    processor = AutoProcessor.from_pretrained(args.pretrained_model_name_or_path, use_fast=True)
    visual_attributes = {"dresses": ["Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length", "Cloth Length"],
                                "upper_body": ["Cloth Type", "Waist", "Fit", "Hem", "Neckline", "Sleeve Length", "Cloth Length"], 
                                "lower_body": ["Cloth Type", "Waist", "Fit", "Cloth Length"]}
        
    dataset_kwargs = dict(
        dataroot_path=args.dataset_root,
        order="paired",
        radius=5,
        outputlist=["c_name", "category"],
        sketch_threshold_range=(20, 20),
        size=(args.height, args.width),
    )
    if args.dataset_name == "dresscode":
        DatasetClass = DressCodeDataset
        dataset_kwargs["category"] = ["upper_body", "lower_body", "dresses"]
    else:
        DatasetClass = VitonHDDataset

    train_dataset = DatasetClass(phase="train", **dataset_kwargs)
    test_dataset = DatasetClass(phase="test", **dataset_kwargs)

    dataroot_names = train_dataset.dataroot_names + test_dataset.dataroot_names
    c_names = train_dataset.c_names + test_dataset.c_names
    phases = ["train"] * len(train_dataset) + ["test"] * len(test_dataset)

    category_label_map = {"upper_body": "upper body", "lower_body": "lower body", "dresses": "dress"}

    if args.dataset_name == "vitonhd":
        categories = ["upper_body"] * len(dataroot_names)
    else:
        categories = [dataroot.split("/")[-1] for dataroot in dataroot_names]
    temperatures = args.temperatures
    vlm_captions_dicts = [{} for _ in range(len(temperatures))]

    for i, t in enumerate(temperatures):
        t_str = str(t).replace(".", "_")
        path = os.path.join(args.dataset_root, args.filename.replace(".json", f"_{t_str}.json"))
        if os.path.exists(path):
            with open(path) as f:
                vlm_captions_dicts[i] = json.load(f)

    for idx, (dataroot, phase, c_name, category) in enumerate(tqdm(zip(dataroot_names, phases, c_names, categories)), 1):
        item_id = c_name.split("_")[0]
        if item_id in vlm_captions_dicts[0]:
            continue

        category_to_predict = category_label_map[category]
        if args.dataset_name == "dresscode":
            img_path = os.path.join(dataroot, 'images', item_id + "_0.jpg")
        else:
            img_path = os.path.join(dataroot, phase, 'image', c_name)

        messages = [
            {   "role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {   "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"{img_path}",
                    },
                    {"type": "text", "text": 
                    "Use only visual attributes that are present in the image. "
                    f"Predict values of the following attributes: {visual_attributes[category]}. Do it only for the {category_to_predict} garment. "
                    "It's forbidden to generate the following visual attributes: colors, background and textures/patterns. "
                    "It's forbidden to generate unspecified predictions. It's forbidden to generate newlines characters. "
                    f"Generate in this way: a <cloth type> with <attributes description>"
                    },
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        for i, temperature in enumerate(temperatures):
            generated_ids = model.generate(**inputs, max_new_tokens=75, temperature=temperature, do_sample=True,
                                           pad_token_id=processor.tokenizer.eos_token_id)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            vlm_captions_dicts[i][item_id] = output_text
        
        if idx % 1000 == 0:
            save_captions(args, temperatures, vlm_captions_dicts)

    save_captions(args, temperatures, vlm_captions_dicts)


def save_captions(args, temperatures, vlm_captions_dicts):
    for i, vlm_captions_dict in enumerate(vlm_captions_dicts):
        t_str = str(temperatures[i]).replace(".", "_")
        path = os.path.join(args.dataset_root, args.filename.replace(".json", f"_{t_str}.json"))
        with open(path, "w") as f:
            json.dump(vlm_captions_dict, f)

if __name__ == "__main__":
    args = parse_args()
    qwen_captioning(args)
