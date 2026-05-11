import io
import re
import json
import base64
import random
import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

from ..core.common import any_type

def image2base64(image):
    img = Image.fromarray(image)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

def parse_json(json_str):
    json_output = json_str.strip().removeprefix("```json").removesuffix("```")
    try:
        parsed = json.loads(json_output)
    except Exception as e:
        raise ValueError(f"Unable to load JSON data!\n{e}")
    return parsed

def scale_image(image: torch.Tensor, max_size: int = 128):
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    w, h = img_pil.size
    scale = min(max_size / max(w, h), 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return np.array(img_resized)

def qwen3bbox(image, json_data):
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    bboxes = []
    for item in json_data:
        x0, y0, x1, y1 = item["bbox_2d"]
        size = 1000
        x0 = x0 / size * img.width
        y0 = y0 / size * img.height
        x1 = x1 / size * img.width
        y1 = y1 / size * img.height
        bboxes.append((x0, y0, x1, y1))
    return bboxes

def draw_bbox(image, json_data, mode):
    label_colors = {}
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    draw = ImageDraw.Draw(img)

    for item in json_data:
        try:
            label = item["label"]
        except Exception:
            try:
                label = item["text_content"]
            except Exception:
                label = "bbox"
        x0, y0, x1, y1 = item["bbox_2d"]
        if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
            size = 1000
            x0 = x0 / size * img.width
            y0 = y0 / size * img.height
            x1 = x1 / size * img.width
            y1 = y1 / size * img.height
        bbox = (x0, y0, x1, y1)

        if label not in label_colors:
            label_colors[label] = tuple(random.randint(80, 180) for _ in range(3))
        color = label_colors[label]
        draw.rectangle(bbox, outline=color, width=4)
        text_y = max(0, y0 - 10)
        text_size = draw.textbbox((x0, text_y), label)
        draw.rectangle([text_size[0], text_size[1]-2, text_size[2]+4, text_size[3]+2], fill=color)
        draw.text((x0+2, text_y), label, fill=(255,255,255))
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

def extract_thought(text):
    thought_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()
        output = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return output, thought

    if '</think>' in text:
        parts = text.split('</think>', 1)
        thought = parts[0].strip()
        thought = thought.replace('<think>', '').strip()
        output = parts[1].strip()
        return output, thought

    return text, ""

class json_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json": ("STRING", {"forceInput": True}),
                "mode": (["simple","Qwen3-VL", "Qwen2.5-VL"], {"default": "simple"}),
                "label": ("STRING", {"default":"", "multiline": False}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("BBOX", "IMAGE")
    RETURN_NAMES = ("bboxes", "image_list")
    OUTPUT_IS_LIST = (True, True)
    INPUT_IS_LIST = True
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, json_input, mode_input, label_input, image=None):
        mode = mode_input[0]
        label = label_input[0]

        flat_images_list = []
        original_structure = []

        if image is not None:
            for img_batch in image:
                if img_batch.ndim == 3:
                    flat_images_list.append(img_batch.unsqueeze(0))
                    original_structure.append(1)
                else:
                    count = img_batch.shape[0]
                    original_structure.append(count)
                    for n in range(count):
                        flat_images_list.append(img_batch[n:n+1])

        total_images = len(flat_images_list)
        output_bboxes = []
        processed_flat_results = []

        for i, j in enumerate(json_input):
            bboxes = parse_json(j)

            if label != "":
                try:
                    bboxes = [item for item in bboxes if item["label"] == label]
                except Exception:
                    bboxes = [item for item in bboxes if item.get("text_content") == label]

            if total_images > 0:
                curr_idx = i if i < total_images else (total_images - 1)
                curr_img = flat_images_list[curr_idx]

                try:
                    res_img = draw_bbox(curr_img[0], bboxes, mode)
                    if res_img.ndim == 3:
                        res_img = res_img.unsqueeze(0)
                    elif res_img.ndim == 4 and res_img.shape[0] > 1:
                        res_img = res_img[0:1]

                    processed_flat_results.append(res_img)
                except Exception as e:
                    print(f"Error drawing on image {curr_idx}: {e}")
                    processed_flat_results.append(curr_img)

            if mode in ["Qwen3-VL", "Qwen2.5-VL"]:
                if total_images == 0:
                    raise ValueError("Image required for Qwen mode")
                curr_idx = i if i < total_images else (total_images - 1)
                bbox = qwen3bbox(flat_images_list[curr_idx][0], bboxes)
            else:
                bbox = [tuple(item["bbox_2d"]) for item in bboxes]

            output_bboxes.append(bbox)

        restructured_images_list = []
        cursor = 0
        for count in original_structure:
            chunk = processed_flat_results[cursor : cursor + count]
            if chunk:
                restructured_images_list.append(torch.cat(chunk, dim=0))
            cursor += count

        return (output_bboxes, restructured_images_list)

class SEG:
    def __init__(self, cropped_image, cropped_mask, confidence, crop_region, bbox, label, control_net_wrapper=None):
        self.cropped_image = cropped_image
        self.cropped_mask = cropped_mask
        self.confidence = confidence
        self.crop_region = crop_region
        self.bbox = bbox
        self.label = label
        self.control_net_wrapper = control_net_wrapper

class bbox_to_segs:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        mask_shape = (height, width)
        seg_list = []
        image_for_cropping = image[0]

        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1_exp, y1_exp = x1 - dilation, y1 - dilation
            x2_exp, y2_exp = x2 + dilation, y2 + dilation

            crop_region = [x1_exp, y1_exp, x2_exp, y2_exp]
            crop_w, crop_h = x2_exp - x1_exp, y2_exp - y1_exp

            if crop_h <= 0 or crop_w <= 0:
                continue

            local_mask_np = np.zeros((crop_h, crop_w), dtype=np.float32)
            local_mask_np[dilation:dilation+(y2-y1), dilation:dilation+(x2-x1)] = 1.0

            if feather > 0:
                local_mask_np = gaussian_filter(local_mask_np, sigma=feather)

            cropped_img_padded = torch.zeros((crop_h, crop_w, 3), dtype=image.dtype, device=image.device)
            src_x1, src_y1 = max(0, x1_exp), max(0, y1_exp)
            src_x2, src_y2 = min(width, x2_exp), min(height, y2_exp)
            dst_x1, dst_y1 = src_x1 - x1_exp, src_y1 - y1_exp
            dst_x2, dst_y2 = src_x2 - x1_exp, src_y2 - y1_exp

            if src_x2 > src_x1 and src_y2 > src_y1:
                cropped_img_padded[dst_y1:dst_y2, dst_x1:dst_x2, :] = image_for_cropping[src_y1:src_y2, src_x1:src_x2, :]

            seg = SEG(
                cropped_image=cropped_img_padded.permute(2, 0, 1).unsqueeze(0),
                cropped_mask=local_mask_np,
                confidence=np.array([0.9], dtype=np.float32),
                crop_region=crop_region,
                bbox=np.array(bbox, dtype=np.float32),
                label="bbox"
            )
            seg_list.append(seg)

        return ((mask_shape, seg_list),)

class bbox_to_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
                "dilation": ("INT", {"default": 10, "min": 0, "max": 200}),
                "feather": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, bboxes, image, dilation, feather):
        _batch_size, height, width, _channels = image.shape
        combined_full_mask = torch.zeros((height, width), dtype=torch.float32, device=image.device)

        for bbox in bboxes:
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            current_full_mask_np = np.zeros((height, width), dtype=np.float32)
            x1_c, y1_c = max(0, x1 - dilation), max(0, y1 - dilation)
            x2_c, y2_c = min(width, x2 + dilation), min(height, y2 + dilation)

            if x2_c > x1_c and y2_c > y1_c:
                current_full_mask_np[y1_c:y2_c, x1_c:x2_c] = 1.0

            if feather > 0:
                current_full_mask_np = gaussian_filter(current_full_mask_np, sigma=feather)

            combined_full_mask = torch.maximum(combined_full_mask, torch.from_numpy(current_full_mask_np).to(image.device))

        return (combined_full_mask.unsqueeze(0),)

class bboxes_to_bbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image_index": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "bbox_index": ("INT", {"default": 0, "min": -998, "max": 999}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    RETURN_NAMES = ("bbox",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, bboxes, image_index, bbox_index):
        if bbox_index != 999:
            return ([bboxes[image_index][bbox_index]],)
        return (bboxes[image_index],)

class parse_json_node:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"input": ("STRING", {"forceInput": True})},
            "optional": {"key": ("STRING",), "default": ("STRING",)},
        }

    RETURN_TYPES = (any_type, "STRING", "INT", "FLOAT", "BOOLEAN")
    RETURN_NAMES = ("any", "string", "int", "float", "boolean")
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, input_data, key=None, default=None):
        if isinstance(input_data, str):
            input_data = [input_data]

        result = {"any": [], "string": [], "int": [], "float": [], "boolean": []}
        for json_str in input_data:
            val = get_nested_value(json_str.strip().removeprefix("```json").removesuffix("```"), key, default)
            result["any"].append(val)
            result["string"].append(str(val))
            try: result["int"].append(int(val))
            except: result["int"].append(0)
            try: result["float"].append(float(val))
            except: result["float"].append(0.0)
            result["boolean"].append(str(val).lower() == "true")

        if len(result["any"]) == 1:
            return (result["any"][0], result["string"][0], result["int"][0], result["float"][0], result["boolean"][0])
        return (result["any"], result["string"], result["int"], result["float"], result["boolean"])

def get_nested_value(data, dotted_key, default=None):
    keys = dotted_key.split('.')
    for key in keys:
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return default
    return data

class remove_code_block:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"input": ("STRING", {"forceInput": True})},
            "optional": {"label": ("STRING",)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llama-cpp-vlm"

    def process(self, input_data, label):
        if isinstance(input_data, str):
            input_data = [input_data]
        output = [v.strip().removeprefix(f"```{label}").removesuffix("```") for v in input_data]
        return (output[0],) if len(output) == 1 else (output,)
