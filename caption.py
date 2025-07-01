import glob
import json
import os
import random
import shutil

import torch
from accelerate.utils import set_seed
from PIL import Image
from tqdm import tqdm, trange
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

import argparse


def center_crop(im: Image) -> Image:
    # Get dimensions
    width, height = im.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def load_im_from_path(im_path: str) -> Image:
    image = Image.open(im_path).convert("RGB")
    image = center_crop(image)
    image = image.resize((image_resolution, image_resolution), Image.LANCZOS)
    return image


def get_random_image(folder_path):
    x = random.choice(folder_path)
    while x.split("/")[-1] in [
        "n02085936_7574.JPEG",
        "n02105855_16006.JPEG",
        "n02107574_133.JPEG",
        "n02107683_1677.JPEG",
        "n02107683_5973.JPEG",
        "n02108915_5290.JPEG",
        "n02105641_6159.JPEG",
        "n02123159_630.JPEG",
        "n02124075_7446.JPEG",
        "n02108915_2468.JPEG",
        "n02085936_8369.JPEG",
        "n02108915_2789.JPEG",
        "n02124075_2958.JPEG",
        "n02105855_10368.JPEG",
        "n02123394_2784.JPEG",
        "n02107683_3886.JPEG",
        "n02123394_6318.JPEG",
        "n02105855_13529.JPEG",
        "n02124075_4399.JPEG",
        "n02123597_6444.JPEG",
        "n02123597_2370.JPEG",
        "n02124075_7914.JPEG",
        "n02105855_10167.JPEG",
        "n02123597_1843.JPEG",
        "n02105855_13211.JPEG",
        "n02105855_15330.JPEG",
        "n02107683_5243.JPEG",
        "n02123159_8118.JPEG",
        "n02124075_1953.JPEG",
        "n02107683_3428.JPEG",
        "n02124075_14965.JPEG",
        "n02123597_12906.JPEG",
        "n02123597_8698.JPEG",
        "n02123597_27315.JPEG",
        "n02124075_13216.JPEG",
        "n02123394_2482.JPEG",
        "n02124075_6734.JPEG",
        "n02123394_8271.JPEG",
        "n02123394_4520.JPEG",
        "n02124075_7196.JPEG",
        "n02123597_4513.JPEG",
        "n02123597_2219.JPEG",
        "n02123597_14478.JPEG",
        "n02123597_4550.JPEG",
        "n02105855_5964.JPEG",
        "n02123394_6112.JPEG",
        "n02105855_3240.JPEG",
        "n02107683_2539.JPEG",
        "n02108915_10337.JPEG",
        "n02105855_2447.JPEG",
        "n02105855_16191.JPEG",
        "n02108915_4604.JPEG",
        "n02105855_16951.JPEG",
        "n02105855_18241.JPEG",
        "n02105855_17070.JPEG",
        "n02105855_11493.JPEG",
        "n02107574_3775.JPEG",
        "n02107574_282.JPEG",
        "n02108915_6366.JPEG",
        "n02105855_13382.JPEG",
        "n02123597_7798.JPEG",
        "n02107574_4950.JPEG",
        "n02107574_351.JPEG",
        "n02123597_5513.JPEG",
    ]:
        x = random.choice(folder_path)
    return x


def copy_image(source_folder, dest_folder, image_name):
    dest_path = os.path.join(dest_folder, image_name.split("/")[-1])
    shutil.copy2(image_name, dest_path)
    return dest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path', required=True, type=str, help="path to the image dir")
    parser.add_argument(
        '--json_path', required=True, type=str, help="path to the caption dir")
    args, extras = parser.parse_known_args()
    
    set_seed(42)
    image_resolution = 768

    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.to("cuda")

    json_output = []
    all_images = glob.glob("args.image_path")
    for i, image_path in enumerate(tqdm(all_images)):
        idx = image_path.split('.')[0]
        if idx.endswith('1'):
            continue
        image_path_0 = image_path
        image_path_1 = image_path.replace('_0', '_1')
        image_0 = Image.open(image_path_0)
        image_1 = Image.open(image_path_1)
        width, height = image_0.size
        prompt = "[INST] <image>\nDescribe the image using five phrases and separate the phrases using commas.[/INST]"
        inputs = processor(
            prompt, load_im_from_path(image_path_0), return_tensors="pt"
        ).to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=50)
        prompt1 = processor.decode(output[0], skip_special_tokens=True)
        prompt1 = prompt1.split("[/INST]")[-1].strip()
        prompt = "[INST] <image>\nDescribe the image using five phrases and separate the phrases using commas.[/INST]"
        inputs = processor(
            prompt, load_im_from_path(image_path_1), return_tensors="pt"
        ).to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=50)
        prompt2 = processor.decode(output[0], skip_special_tokens=True)
        prompt2 = prompt2.split("[/INST]")[-1].strip()
        json_output.append(
            {
                "exp_id": i,
                "image_paths": [
                    image_path_0,
                    image_path_1,
                ],
                "prompts": [prompt1, prompt2],
            }
        )

    with open("args.json_path/caption.json", "w") as f:
        for item in json_output:
            # Convert each dictionary to a JSON string and write it to the file
            f.write(json.dumps(item) + "\n")
