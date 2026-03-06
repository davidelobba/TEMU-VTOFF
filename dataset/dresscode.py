import json
import os
import random
from pathlib import Path
from typing import Tuple, Literal
import shutil

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from torchvision.ops import masks_to_boxes

from utils.labelmap import label_map
from utils.posemap import kpoint_to_heatmap
import random

torch.multiprocessing.set_sharing_strategy('file_system')


class DressCodeDataset(data.Dataset):
    """
    Dataset class for the Dress Code Multimodal Dataset
    im_parse: inpaint mask
    """

    def __init__(self,
                 dataroot_path: str,
                 phase: Literal["train", "test"],
                 radius: int = 5,
                 caption_file: str = 'fine_captions.json',
                 coarse_caption_file: Literal['coarse_captions.json', 'qwen_captions_2_5_0_2.json', 'all_captions_structural.json'] = 'coarse_captions.json',
                 sketch_threshold_range: Tuple[int, int] = (20, 127),
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'image', 'im_cloth', 'cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'im_sketch', 'greyscale_im_sketch', 'captions', 'category', 'stitch_label',
                                           'texture', 'cloth_embeddings', 'dwpose', 'vton_image_embeddings'),
                 category: Literal['dresses', 'upper_body', 'lower_body'] = ('dresses', 'upper_body', 'lower_body'),
                 size: Tuple[int, int] = (512, 384),
                 mask_type: Literal["mask", "bounding_box"] = "bounding_box",
                 texture_order: Literal["shuffled", "retrieved", "original"] = "original",
                 mask_items: Tuple[str] = ["original"],
                 mask_category: Literal['upper_body', 'lower_body'] = None,
                 n_chunks=3,
                 ):

        super().__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.sketch_threshold_range = sketch_threshold_range
        self.category = category
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order
        self.mask_type = mask_type
        self.texture_order = texture_order
        self.mask_items = mask_items
        self.mask_category = mask_category

        im_names = []
        c_names = []
        dataroot_names = []
        category_names = []

        possible_outputs = ['c_name', 'im_name', 'cpath', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton', 'im_mask', 
                            'im_mask_fine', 'inshop_mask',
                            'inpaint_mask', 'greyscale_im_sketch', 'parse_mask_total', 'cloth_sketch', 'im_sketch',
                            'captions', 'category', 'hands', 'parse_head_2', 'stitch_label', 'texture', 'parse_cloth',
                            'cloth_embeddings', 'dwpose', 'vton_image_embeddings', 'clip_pooled', 'clip_embeds', 't5_embeds']

        assert all(x in possible_outputs for x in outputlist)

        needs_captions = any(x in outputlist for x in ("captions", "clip_pooled", "clip_embeds", "t5_embeds"))
        self.captions_dict = {}
        self.caption_type = None
        if needs_captions:
            with open(os.path.join(self.dataroot, caption_file)) as f:
                self.captions_dict = json.load(f)
            self.captions_dict = {k: v for k, v in self.captions_dict.items() if len(v) >= 3}
            if coarse_caption_file is not None:
                with open(os.path.join(self.dataroot, coarse_caption_file)) as f:
                    coarse_captions_dict = json.load(f)
                    self.captions_dict.update(coarse_captions_dict)

            if coarse_caption_file == 'coarse_captions.json':
                self.caption_type = 'text_embeddings'
            elif coarse_caption_file == 'all_captions_structural.json':
                self.caption_type = 'struct_text_embeddings'
            elif 'demo_qwen' in coarse_caption_file:
                self.caption_type = 'demo_qwen_text_embeddings'
                self.captions_dict = coarse_captions_dict
            elif 'qwen' in coarse_caption_file:
                self.caption_type = 'qwen_text_embeddings'
            else:
                raise NotImplementedError("not supported caption file")
            del coarse_captions_dict

        for c in sorted(category):
            assert c in ['dresses', 'upper_body', 'lower_body']

            dataroot = os.path.join(self.dataroot, c)
            if phase == 'train':
                filename = os.path.join(dataroot, f"{phase}_pairs.txt")
            else:
                filename = os.path.join(dataroot, f"{phase}_pairs_{order}.txt")


            with open(filename, 'r') as f:
                for line in f.readlines():
                    im_name, c_name = line.strip().split()
                    if needs_captions and c_name.split('_')[0] not in self.captions_dict:
                        continue
                    if im_name in im_names:
                        continue

                    im_names.append(im_name)
                    c_names.append(c_name)
                    dataroot_names.append(dataroot)
                    category_names.append(c)

        self.im_names = im_names
        self.c_names = c_names
        self.dataroot_names = dataroot_names
        self.category_names = category_names
        self.cloth_complete_names = {c_name: os.path.relpath(os.path.join(dataroot_names[idx], "cleaned_inshop_imgs", \
            c_name.replace(".jpg", "_cleaned.jpg")), self.dataroot) for idx, c_name in enumerate(c_names)}
        self.complete_dict = {}
        if needs_captions:
            if not os.path.isfile(os.path.join(dataroot, "complete_dict.json")):
                self.complete_dict = {self.cloth_complete_names[c_name]: list(dict.fromkeys(self.captions_dict[c_name.split('_')[0]])) \
                    for c_name in sorted(c_names)}
                with open(os.path.join(dataroot, f"complete_dict_{phase}.json"), "w") as f:
                    json.dump(self.complete_dict, f, indent=2)
            else:
                with open(os.path.join(dataroot, f"complete_dict_{phase}.json")) as f:
                    self.complete_dict = json.load(f)

    def __getitem__(self, index: int) -> dict:
        """For each index return the corresponding element in the dataset

        Args:
            index (int): element index

        Raises:
            NotImplementedError

        Returns:
            dict:
                c_name: filename of inshop cloth
                im_name: filename of model with cloth
                cloth: img of inshop cloth
                image: img of the model with that cloth
                im_cloth: cut cloth from the model
                im_mask: black mask of the cloth in the model img
                cloth_sketch: sketch of the inshop cloth
                im_sketch: sketch of "im_cloth"
        """

        c_name = self.c_names[index]
        im_name = self.im_names[index]
        dataroot = self.dataroot_names[index]
        category = dataroot.split('/')[-1]
        item_id = os.path.splitext(im_name)[0]

        sketch_threshold = random.randint(
            self.sketch_threshold_range[0], self.sketch_threshold_range[1])

        if "captions" in self.outputlist:
            captions = self.captions_dict[c_name.split('_')[0]]
            if self.phase == 'train':
                random.shuffle(captions)
            captions = ", ".join(captions)

        if "cloth" in self.outputlist:
            cloth_path = os.path.join(
                dataroot, 'cleaned_inshop_imgs', c_name.replace(".jpg", "_cleaned.jpg"))
            if not os.path.exists(cloth_path): cloth_path = os.path.join(dataroot, 'images', c_name)
            cloth = Image.open(cloth_path)
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]
            
        if "cloth_embeddings" in self.outputlist:
            feat_path = os.path.join(str(Path(dataroot).parent), 'cloth_embeddings', 'CLIP_VIT_L_VIT_G_concat', category, f"{c_name.split('.')[0]}.pt")
            cloth_embeddings = torch.load(feat_path, map_location="cpu")

        if "vton_image_embeddings" in self.outputlist:
            feat_path = os.path.join(str(Path(dataroot).parent), 'vton_embeddings', 'CLIP_VIT_L_VIT_G_concat', category, f"{im_name.split('.')[0]}.pt")
            vton_image_embeddings = torch.load(feat_path, map_location="cpu")
        
        
        if "clip_pooled" in self.outputlist:
            feat_path = Path(dataroot).parent / self.caption_type / "CLIP_VIT_L_VIT_G_concat_pooled" / category / f"{c_name.split('.')[0]}.pt"
            clip_pooled = torch.load(feat_path, map_location="cpu")
        
        if "clip_embeds" in self.outputlist:
            feat_path = Path(dataroot).parent / self.caption_type / "CLIP_VIT_L_VIT_G_concat" / category / f"{c_name.split('.')[0]}.pt"
            clip_embeds = torch.load(feat_path, map_location="cpu")
        
        if "t5_embeds" in self.outputlist:
            feat_path = Path(dataroot).parent / self.caption_type / "T5_XXL" / category / f"{c_name.split('.')[0]}.pt"
            t5_embeds = torch.load(feat_path, map_location="cpu")
        
        if "dwpose" in self.outputlist:
            dwpose = Image.open(os.path.join(dataroot, 'dwpose', f"{im_name.split('_')[0]}_2.png"))
        
        if "inshop_mask" in self.outputlist:
            mask_filename = os.path.join(
                dataroot, 'cleaned_inshop_masks', c_name.replace(".jpg", "_cleaned_mask.jpg"))
            mask_image = Image.open(mask_filename)
            mask_image = transforms.ToTensor()(mask_image)
            inshop_mask = mask_image[0, :, :]
        
        if "image" in self.outputlist or "im_head" in self.outputlist \
                or "im_cloth" in self.outputlist:
            image = Image.open(os.path.join(dataroot, 'images', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]

        if "im_sketch" in self.outputlist or "greyscale_im_sketch" in self.outputlist:
            if "unpaired" == self.order and self.phase == 'test':
                greyscale_im_sketch = Image.open(os.path.join(dataroot, 'im_sketch_unpaired',
                                                              f'{im_name.replace(".jpg", "")}_{c_name.replace(".jpg", ".png")}'))
            else:
                greyscale_im_sketch = Image.open(os.path.join(dataroot, 'im_sketch', c_name.replace(".jpg", ".png")))

            greyscale_im_sketch = greyscale_im_sketch.resize((self.width, self.height))
            greyscale_im_sketch = ImageOps.invert(greyscale_im_sketch)
            im_sketch = greyscale_im_sketch.point(
                lambda p: 255 if p > sketch_threshold else 0)
            im_sketch = transforms.functional.to_tensor(im_sketch)  # [0,1]
            greyscale_im_sketch = transforms.functional.to_tensor(greyscale_im_sketch)  # [0,1]
            im_sketch = 1 - im_sketch
            greyscale_im_sketch = 1 - greyscale_im_sketch

        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_cloth" in self.outputlist \
                or "im_mask" in self.outputlist or "im_mask_fine" in self.outputlist or "parse_mask_total" in \
                self.outputlist or "parse_array" in self.outputlist or \
                "pose_map" in self.outputlist or "parse_array" in \
                self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist or "inpaint_mask" in self.outputlist:
            # Label Map
            parse_name = im_name.replace('_0.jpg', '_4.png')
            im_parse = Image.open(os.path.join(
                dataroot, 'label_maps', parse_name))
            im_parse = im_parse.resize(
                (self.width, self.height), Image.NEAREST)
            parse_array = np.array(im_parse)

            parse_shape = (parse_array > 0).astype(np.float32)

            parse_head = (parse_array == 1).astype(np.float32) + \
                         (parse_array == 2).astype(np.float32) + \
                         (parse_array == 3).astype(np.float32) + \
                         (parse_array == 11).astype(np.float32)

            parser_mask_fixed = (parse_array == label_map["hair"]).astype(np.float32) + \
                                (parse_array == label_map["left_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["right_shoe"]).astype(np.float32) + \
                                (parse_array == label_map["hat"]).astype(np.float32) + \
                                (parse_array == label_map["sunglasses"]).astype(np.float32) + \
                                (parse_array == label_map["scarf"]).astype(np.float32) + \
                                (parse_array == label_map["bag"]).astype(
                                    np.float32)

            parser_mask_changeable = (
                    parse_array == label_map["background"]).astype(np.float32)

            arms = (parse_array == 14).astype(np.float32) + \
                   (parse_array == 15).astype(np.float32)

            category = dataroot.split('/')[-1]
            if dataroot.split('/')[-1] == 'dresses':
                label_cat = 7
                parse_cloth = (parse_array == 7).astype(np.float32)
                parse_mask = (parse_array == 7).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)
                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))

            # upper body
            elif dataroot.split('/')[-1] == 'upper_body' or (dataroot.split('/')[-1] == 'lower_body' and self.mask_category=='upper_body'):
                label_cat = 4
                parse_cloth = (parse_array == 4).astype(np.float32)
                parse_mask = (parse_array == 4).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["skirt"]).astype(np.float32) + \
                                     (parse_array == label_map["pants"]).astype(
                                         np.float32)

                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))
            elif dataroot.split('/')[-1] == 'lower_body':
                label_cat = 6
                parse_cloth = (parse_array == 6).astype(np.float32)
                parse_mask = (parse_array == 6).astype(np.float32) + \
                             (parse_array == 12).astype(np.float32) + \
                             (parse_array == 13).astype(np.float32)

                parser_mask_fixed += (parse_array == label_map["upper_clothes"]).astype(np.float32) + \
                                     (parse_array == 14).astype(np.float32) + \
                                     (parse_array == 15).astype(np.float32)
                parser_mask_changeable += np.logical_and(
                    parse_array, np.logical_not(parser_mask_fixed))
            else:
                raise NotImplementedError

            parse_head = torch.from_numpy(parse_head)  # [0,1]
            parse_cloth = torch.from_numpy(parse_cloth)  # [0,1]
            parse_mask = torch.from_numpy(parse_mask)  # [0,1]
            parser_mask_fixed = torch.from_numpy(parser_mask_fixed)
            parser_mask_changeable = torch.from_numpy(parser_mask_changeable)

            parse_without_cloth = np.logical_and(
                parse_shape, np.logical_not(parse_mask))
            parse_mask = parse_mask.cpu().numpy()

            if "im_head" in self.outputlist:
                # Masked cloth
                im_head = image * parse_head - (1 - parse_head)
            if "im_cloth" in self.outputlist:
                im_cloth = image * parse_cloth + (1 - parse_cloth)
            if "im_mask_fine" in self.outputlist:
                im_mask_fine = image * (1-parse_cloth)

            # Shape
            parse_shape = Image.fromarray((parse_shape * 255).astype(np.uint8))
            parse_shape = parse_shape.resize(
                (self.width // 16, self.height // 16), Image.BILINEAR)
            parse_shape = parse_shape.resize(
                (self.width, self.height), Image.BILINEAR)
            shape = self.transform2d(parse_shape)  # [-1,1]

            # Load pose points
            pose_name = im_name.replace('_0.jpg', '_2.json')
            with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['keypoints']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 4))

            point_num = pose_data.shape[0]
            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)
                point_x = np.multiply(pose_data[i, 0], self.width / 384.0)
                point_y = np.multiply(pose_data[i, 1], self.height / 512.0)
                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r,
                                    point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle(
                        (point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2d(one_map)
                pose_map[i] = one_map[0]

            d = []
            for pose_d in pose_data:
                ux = pose_d[0] / 384.0
                uy = pose_d[1] / 512.0

                # scale posemap points
                px = ux * self.width
                py = uy * self.height

                d.append(kpoint_to_heatmap(
                    np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)

            # just for visualization
            im_pose = self.transform2d(im_pose)

            im_arms = Image.new('L', (self.width, self.height))
            arms_draw = ImageDraw.Draw(im_arms)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body' or dataroot.split('/')[
                -1] == 'lower_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    shoulder_right = np.multiply(
                        tuple(data['keypoints'][2][:2]), self.height / 512.0)
                    shoulder_left = np.multiply(
                        tuple(data['keypoints'][5][:2]), self.height / 512.0)
                    elbow_right = np.multiply(
                        tuple(data['keypoints'][3][:2]), self.height / 512.0)
                    elbow_left = np.multiply(
                        tuple(data['keypoints'][6][:2]), self.height / 512.0)
                    wrist_right = np.multiply(
                        tuple(data['keypoints'][4][:2]), self.height / 512.0)
                    wrist_left = np.multiply(
                        tuple(data['keypoints'][7][:2]), self.height / 512.0)
                    if wrist_right[0] <= 1. and wrist_right[1] <= 1.:
                        if elbow_right[0] <= 1. and elbow_right[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((wrist_left, elbow_left, shoulder_left, shoulder_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    elif wrist_left[0] <= 1. and wrist_left[1] <= 1.:
                        if elbow_left[0] <= 1. and elbow_left[1] <= 1.:
                            arms_draw.line(
                                np.concatenate((shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                    np.uint16).tolist(), 'white', 45, 'curve')
                        else:
                            arms_draw.line(np.concatenate(
                                (elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                                np.uint16).tolist(), 'white', 45, 'curve')
                    else:
                        arms_draw.line(np.concatenate(
                            (wrist_left, elbow_left, shoulder_left, shoulder_right, elbow_right, wrist_right)).astype(
                            np.uint16).tolist(), 'white', 45, 'curve')

                hands = np.logical_and(np.logical_not(im_arms), arms)

                if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                    parse_mask += im_arms
                    if not ("hands" in self.mask_items or "all" in self.mask_items):
                        parser_mask_fixed += hands

            # delete neck
            parse_head_2 = torch.clone(parse_head)
            if dataroot.split('/')[-1] == 'dresses' or dataroot.split('/')[-1] == 'upper_body':
                with open(os.path.join(dataroot, 'keypoints', pose_name), 'r') as f:
                    data = json.load(f)
                    points = []
                    points.append(np.multiply(
                        tuple(data['keypoints'][2][:2]), self.height / 512.0))
                    points.append(np.multiply(
                        tuple(data['keypoints'][5][:2]), self.height / 512.0))
                    x_coords, y_coords = zip(*points)
                    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                    m, c = np.linalg.lstsq(A, y_coords, rcond=None)[0]
                    for i in range(parse_array.shape[1]):
                        y = i * m + c
                        parse_head_2[int(
                            y - 20 * (self.height / 512.0)):, i] = 0

            parser_mask_fixed = np.logical_or(
                parser_mask_fixed, np.array(parse_head_2, dtype=np.uint16))
            parse_mask += np.logical_or(parse_mask, np.logical_and(np.array(parse_head, dtype=np.uint16),
                                                                   np.logical_not(
                                                                       np.array(parse_head_2, dtype=np.uint16))))

            parse_mask = cv2.dilate(parse_mask, np.ones(
                (5, 5), np.uint16), iterations=5)
            parse_mask = np.logical_and(
                parser_mask_changeable, np.logical_not(parse_mask))
            parse_mask_total = np.logical_or(parse_mask, parser_mask_fixed)
            im_mask = image * parse_mask_total
            inpaint_mask = 1 - parse_mask_total

            if self.mask_type == 'bounding_box':
                bboxes = masks_to_boxes(inpaint_mask.unsqueeze(0))
                bboxes = bboxes.type(torch.int32)
                xmin = bboxes[0, 0]
                xmax = bboxes[0, 2]
                ymin = bboxes[0, 1]
                ymax = bboxes[0, 3]

                inpaint_mask[ymin:ymax + 1, xmin:xmax + 1] = torch.logical_and(
                    torch.ones_like(inpaint_mask[ymin:ymax + 1, xmin:xmax + 1]),
                    torch.logical_not(parser_mask_fixed[ymin:ymax + 1, xmin:xmax + 1]))

                inpaint_mask = inpaint_mask.unsqueeze(0)
                im_mask = image * np.logical_not(inpaint_mask.repeat(3, 1, 1))
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            elif self.mask_type == "mask":
                inpaint_mask = inpaint_mask.unsqueeze(0)
                parse_mask_total = parse_mask_total.numpy()
                parse_mask_total = parse_array * parse_mask_total
                parse_mask_total = torch.from_numpy(parse_mask_total)
            else:
                raise ValueError("Unknown mask type")

        if "stitch_label" in self.outputlist:
            stitch_labelmap = Image.open(os.path.join(
                self.dataroot, 'test_stitchmap', im_name.replace(".jpg", ".png")))
            stitch_labelmap = transforms.ToTensor()(stitch_labelmap) * 255
            stitch_label = stitch_labelmap == 13

        cpath = self.cloth_complete_names[c_name]
        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self) -> int:
        """Return dataset length

        Returns:
            int: dataset length
        """

        return len(self.c_names)