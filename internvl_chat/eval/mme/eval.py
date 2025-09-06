import argparse
import os
import re
from pathlib import Path
import cv2

import torch
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from tqdm import tqdm

from sfu_torch_lib import io

from multimodal_coding.utils.utils import compress_image_jpeg
from multimodal_coding.prefilter import tinyclip_filter
from multimodal_coding.utils.utils import is_debug_mode


FILE = Path(__file__).resolve()
ROOT = FILE.parents[4]


def load_image(
    image_file: str,
    prompt: str,
    args: argparse.Namespace,
    input_size: int = 224,
):
    orig_image = Image.open(image_file).convert('RGB')
    if args.verbose:
        orig_image.show()
    file_size_bytes = os.path.getsize(image_file)  # on-disk size
    width, height = orig_image.size
    num_pixels = width * height
    orig_bpp = (file_size_bytes * 8) / num_pixels

    if args.prefilter:
        image_filtered = tinyclip_filter(
            img_path=image_file,
            clip_arch=args.clip_arch,
            clip_pretrained=args.clip_pretrained,
            prompt=prompt,
            max_tiles=args.max_num*4,   # max_num for internvl 448x448 patches, but for tinyclip the tile size is 224x224
            prescale=args.prescale,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            method=args.method,
            ksize=args.ksize
        )
        # Encode (compress) to JPEG in memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # quality from 0–100
        success, encoded_img = cv2.imencode('.jpg', image_filtered, encode_param)
        assert success, "Image encoding failed"
        compressed_bpp = encoded_img.size * 8 / num_pixels
        # Decode (load back into cv2 NumPy array)
        decoded_img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        # Convert to PIL
        cv_img_rgb = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(cv_img_rgb)
        if args.verbose:
            image.show()
    else:
        image = orig_image

    if args.compress_ratio != 0:
        image = compress_image_jpeg(image, quality=args.compress_ratio)

    transform = build_transform(is_train=False, input_size=input_size)
    if args.dynamic:
        images = dynamic_preprocess(image, image_size=input_size,
                                    use_thumbnail=use_thumbnail,
                                    max_num=args.max_num)
        orig_images = dynamic_preprocess(orig_image, image_size=input_size,
                                         use_thumbnail=use_thumbnail,
                                         max_num=args.max_num)
    else:
        images = [image]
        orig_images = [orig_image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    orig_pixel_values = [transform(image) for image in orig_images]
    orig_pixel_values = torch.stack(orig_pixel_values)
    return pixel_values, orig_pixel_values


def post_processing(response):
    response = response.replace('\n', '').replace('不是', 'No').replace('是', 'Yes').replace('否', 'No')
    response = response.lower().replace('true', 'yes').replace('false', 'no')
    pattern = re.compile(r'[\u4e00-\u9fa5]')
    response = re.sub(pattern, '', response)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=str(ROOT / 'InternVL/pretrained/InternVL2_5-1B') if is_debug_mode() else \
            str(io.localize_cached(path="s3://models/InternVL/InternVL2_5-1B.zip")))
    parser.add_argument('--data-path', type=str, default=str((ROOT / 'InternVL/internvl_chat/data/mme/MME_Benchmark_release_version').resolve()) if is_debug_mode() else \
            str(io.localize_dataset(path="s3://datasets/multimodal", overwrite=False)))
    parser.add_argument('--root', type=str, default=str(ROOT / 'InternVL/internvl_chat/eval/mme/Your_Results'))
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--sample', type=bool, default=False)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--prefilter', action='store_true', help='Use prefiltering based on CLIP scores')
    parser.add_argument('--compress-ratio', type=int, default=0)
    parser.add_argument("--sigma_min", type=float, default=0.5, help="Minimum sigma value for Gaussian blur.")
    parser.add_argument("--sigma_max", type=float, default=10.0, help="Maximum sigma value for Gaussian blur.")
    parser.add_argument("--method", type=str, default="exponential", choices=["linear", "exponential", "inverse"], help="Method for sigma value calculation.")
    parser.add_argument("--ksize", type=int, default=11, help="Kernel size for Gaussian blur.")
    parser.add_argument("--prescale", type=float, default=50.0, help="Prescale factor for the scores before softmax.")
    parser.add_argument("--clip_arch", type=str, default="TinyCLIP-ViT-39M-16-Text-19M", help="Architecture of the CLIP model.")
    parser.add_argument("--clip_pretrained", type=str, default="YFCC15M", help="Pretrained weights for the CLIP model.")
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f'[test] total_params: {total_params}B, use num_beams: {args.num_beams}')
    else:
        print(f'[test] total_params: {total_params}B')
    print(f'[test] image_size: {image_size}')
    print(f'[test] template: {model.config.template}')
    print(f'[test] dynamic_image_size: {args.dynamic}')
    print(f'[test] use_thumbnail: {use_thumbnail}')
    print(f'[test] max_num: {args.max_num}')

    output = os.path.basename(args.checkpoint)
    os.makedirs(output, exist_ok=True)
    prompt = 'Answer the question using a single word or phrase.'

    for filename in os.listdir(args.root):
        fin = open(os.path.join(args.root, filename), 'r', encoding='utf-8')
        fout = open(os.path.join(output, filename), 'w', encoding='utf-8')
        lines = fin.readlines()
        filename = filename.replace('.txt', '')
        for line in tqdm(lines):
            img, text, gt = line.strip().split('\t')
            question = text + ' ' + prompt
            img_path = os.path.join(args.data_path, filename, img)
            assert os.path.exists(img_path), img_path
            pixel_values, orig_pixel_values = load_image(img_path, text, args, image_size)
            pixel_values = pixel_values.cuda().to(torch.bfloat16)
            orig_pixel_values = orig_pixel_values.cuda().to(torch.bfloat16)
            generation_config = dict(
                do_sample=args.sample,
                num_beams=args.num_beams,
                max_new_tokens=20,
                eos_token_id=tokenizer.eos_token_id,
            )
            if args.sample:
                generation_config["top_k"] = args.top_k
                generation_config["top_p"] = args.top_p
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=False
            )
            orig_response = model.chat(
                tokenizer=tokenizer,
                pixel_values=orig_pixel_values,
                question=question,
                generation_config=generation_config,
                verbose=False
            )
            response = post_processing(response)
            orig_response = post_processing(orig_response)
            print(img, question, gt, response, sep='\t', file=fout)
            if args.verbose:
                print(text)
                print("original response:", orig_response)
                print("prefiltered image response:", response)
        fin.close()
        fout.close()
